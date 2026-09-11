"""Correctness gate for _C::fused_experts_cpu (SGL CPU MoE kernel).

Compares the fused kernel against a pure-PyTorch fp32 reference at the
Qwen3.6-35B-A3B shapes (hidden 2048, moe_intermediate 512, 256 experts,
top-k 8, bf16, silu). Runs inside the vllm_artemis:cpu image.

Exit code 0 = pass, 1 = fail.
"""

import sys

import torch

import vllm._C  # noqa: F401  (registers torch.ops._C)

HIDDEN = 2048
INTER = 512
NUM_EXPERTS = 256
TOP_K = 8
DTYPE = torch.bfloat16
# bf16 kernel vs fp32 reference: elementwise agreement is loose, the
# relative error of the whole output tensor is the robust signal.
MAX_REL_ERR = 0.02


def reference_moe(x, w13, w2, topk_weights, topk_ids):
    """Pure-torch fp32 reference using the unpacked weights."""
    x = x.float()
    w13 = w13.float()
    w2 = w2.float()
    m = x.size(0)
    out = torch.zeros(m, HIDDEN, dtype=torch.float32)
    for t in range(m):
        for k in range(TOP_K):
            e = int(topk_ids[t, k])
            gate_up = x[t] @ w13[e].t()  # [2*INTER]
            gate, up = gate_up[:INTER], gate_up[INTER:]
            act = torch.nn.functional.silu(gate) * up
            out[t] += float(topk_weights[t, k]) * (act @ w2[e].t())
    return out


def run_case(m: int) -> bool:
    torch.manual_seed(1234 + m)
    x = torch.randn(m, HIDDEN, dtype=DTYPE) * 0.5
    w13 = torch.randn(NUM_EXPERTS, 2 * INTER, HIDDEN, dtype=DTYPE) * 0.05
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTER, dtype=DTYPE) * 0.05

    router = torch.randn(m, NUM_EXPERTS, dtype=torch.float32)
    weights, ids = torch.topk(torch.softmax(router, dim=-1), TOP_K, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    ids = ids.to(torch.int32)

    packed_w13 = torch.ops._C.convert_weight_packed(w13)
    packed_w2 = torch.ops._C.convert_weight_packed(w2)

    got = torch.ops._C.fused_experts_cpu(
        x.clone(), packed_w13, packed_w2, weights, ids,
        False,  # inplace
        0,      # moe_comp_method: UNQUANT
        None, None, None, None,  # scales / zeros
        None,   # block_size
        None, None,  # biases
        None, None,  # alpha / limit
        True,   # is_vnni
    ).float()

    want = reference_moe(x, w13, w2, weights, ids)

    if not torch.isfinite(got).all():
        print(f"M={m}: FAIL — non-finite values in kernel output")
        return False

    rel_err = (got - want).norm() / want.norm()
    ok = rel_err < MAX_REL_ERR
    print(f"M={m}: rel_err={rel_err:.5f} ({'ok' if ok else 'FAIL'})")
    return bool(ok)


def main() -> int:
    torch.set_num_threads(torch.get_num_threads())
    results = [run_case(m) for m in (1, 8, 64)]
    if all(results):
        print("MOE KERNEL CORRECTNESS: PASS")
        return 0
    print("MOE KERNEL CORRECTNESS: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
