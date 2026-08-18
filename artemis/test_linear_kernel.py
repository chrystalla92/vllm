"""Correctness gate for _C::weight_packed_linear (SGL CPU dense GEMM).

Compares the packed-weight linear kernel against a pure-torch fp32
reference at Qwen3.6-35B-A3B decode shapes (K=2048 projections, small M).
Runs inside the vllm_artemis:cpu image.

Exit code 0 = pass, 1 = fail.
"""

import sys

import torch

import vllm._C  # noqa: F401  (registers torch.ops._C)

MAX_REL_ERR = 0.02
# (M, K, N): decode/prefill shapes around the model's dense projections
CASES = [
    (1, 2048, 2048),
    (8, 2048, 4096),
    (8, 2048, 512),
    (64, 512, 2048),
    (256, 2048, 2048),
]


def run_case(m: int, k: int, n: int, with_bias: bool) -> bool:
    torch.manual_seed(m * 7 + k + n)
    x = torch.randn(m, k, dtype=torch.bfloat16) * 0.5
    w = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    bias = torch.randn(n, dtype=torch.float32) * 0.1 if with_bias else None

    packed = torch.ops._C.convert_weight_packed(w)
    got = torch.ops._C.weight_packed_linear(x, packed, bias, True).float()

    want = x.float() @ w.float().t()
    if bias is not None:
        want = want + bias.float()

    if not torch.isfinite(got).all():
        print(f"M={m} K={k} N={n} bias={with_bias}: FAIL — non-finite output")
        return False

    rel_err = (got - want).norm() / want.norm()
    ok = rel_err < MAX_REL_ERR
    print(f"M={m} K={k} N={n} bias={with_bias}: rel_err={rel_err:.5f} "
          f"({'ok' if ok else 'FAIL'})")
    return bool(ok)


def main() -> int:
    results = [run_case(m, k, n, wb)
               for m, k, n in CASES for wb in (False, True)]
    if all(results):
        print("LINEAR KERNEL CORRECTNESS: PASS")
        return 0
    print("LINEAR KERNEL CORRECTNESS: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
