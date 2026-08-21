"""Microbenchmark for the MoE and dense-GEMM CPU kernels at PRODUCTION shapes.

Why this exists
---------------
End-to-end throughput on the production trace has a ~3% noise floor, while a
realistic kernel win is 10-20% of a 10-20% slice, i.e. 1-3% end to end. The
instrument cannot resolve the effect. This calls the kernels directly, which
removes the server, arrival-rate replay, prefix cache, lazy Inductor
compilation and concurrency jitter, so a 15% kernel win reads as ~15%.

Shapes are taken from the measured production profile, not invented:
  Qwen3.6-35B-A3B: hidden 2048, moe_intermediate 512, 256 experts, top-k 8.
  Prefill chunk M=2048 (vLLM default max_num_batched_tokens, not overridden).
    -> MoE avg_M = 2048*8/256 = 64 > 4  -> AMX brgemm path
  Decode  M=42   (observed concurrency at benchmark saturation).
    -> MoE avg_M = 42*8/256 = 1        -> AVX512 tinygemm path
  Dense dispatches on M directly (M > 4), so BOTH regimes use brgemm.

VALIDATION: the profiled per-call medians on the real trace are
  fused_experts_cpu  p50 4.49 ms, p90 19.70 ms  (decode / prefill modes)
  weight_packed_linear p50 0.07 ms, with 352 calls >1ms holding 61% of time
If the timings below are far from those, the shapes are unrepresentative and
the benchmark is measuring fiction - check before trusting any result.

Fitness: total weighted CPU time in ms per model layer-step, weighting each
regime by its share of calls in the production profile. LOWER IS BETTER.
Both prefill and decode are included so that a prefill win which wrecks
decode is penalised rather than hidden.
"""

import json
import os
import sys
import time

import torch

import vllm._C  # noqa: F401  (registers torch.ops._C)

H, I, E, TOPK = 2048, 512, 256, 8
DTYPE = torch.bfloat16
M_PREFILL, M_DECODE = 2048, 42

# Share of each regime's calls in the production profile. MoE is bimodal with
# roughly half its time in each mode; dense is dominated by its large calls.
WEIGHTS = {"moe_prefill": 0.5, "moe_decode": 0.5, "dense_prefill": 0.61, "dense_decode": 0.39}

REPS = int(os.environ.get("MICROBENCH_REPS", "5"))
WARMUP = int(os.environ.get("MICROBENCH_WARMUP", "2"))


def _time(fn, reps=REPS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best  # ms; min over reps is the least noisy estimator here


def bench_moe(m):
    torch.manual_seed(0)
    x = torch.randn(m, H, dtype=DTYPE) * 0.5
    w13 = torch.randn(E, 2 * I, H, dtype=DTYPE) * 0.05
    w2 = torch.randn(E, H, I, dtype=DTYPE) * 0.05
    router = torch.randn(m, E, dtype=torch.float32)
    weights, ids = torch.topk(torch.softmax(router, dim=-1), TOPK, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    ids = ids.to(torch.int32)
    pw13 = torch.ops._C.convert_weight_packed(w13)
    pw2 = torch.ops._C.convert_weight_packed(w2)

    def run():
        torch.ops._C.fused_experts_cpu(
            x.clone(), pw13, pw2, weights, ids,
            False, 0, None, None, None, None, None, None, None, None, None, True,
        )

    return _time(run)


def bench_dense(m):
    """Sum over the per-layer dense projections at this M."""
    torch.manual_seed(1)
    # (K, N) for the projections this model issues per layer
    shapes = [(H, 4096), (4096, H), (H, 2 * I), (I, H), (H, 512)]
    packed = []
    for k, n in shapes:
        w = torch.randn(n, k, dtype=DTYPE) * 0.05
        packed.append((torch.randn(m, k, dtype=DTYPE) * 0.5,
                       torch.ops._C.convert_weight_packed(w)))

    def run():
        for a, b in packed:
            torch.ops._C.weight_packed_linear(a, b, None, True)

    return _time(run)


def main() -> int:
    res = {
        "moe_prefill": bench_moe(M_PREFILL),
        "moe_decode": bench_moe(M_DECODE),
        "dense_prefill": bench_dense(M_PREFILL),
        "dense_decode": bench_dense(M_DECODE),
    }
    fitness = sum(res[k] * WEIGHTS[k] for k in res)

    print(f"{'regime':16s} {'ms/call':>10s} {'weight':>8s}")
    for k, v in res.items():
        print(f"{k:16s} {v:10.3f} {WEIGHTS[k]:8.2f}")
    print(f"\nweighted_total_ms {fitness:.3f}   (LOWER IS BETTER)")

    # Sanity against the production profile
    print("\n-- shape validation vs production profile --")
    print(f"  moe_prefill {res['moe_prefill']:.2f} ms   profile p90 ~19.70 ms")
    print(f"  moe_decode  {res['moe_decode']:.2f} ms   profile p50 ~4.49 ms")

    out = {f"micro_{k}_ms": v for k, v in res.items()}
    out["weighted_total_ms"] = fitness
    out["fitness_inverse"] = 1000.0 / fitness  # higher-is-better form
    json.dump(out, open("artemis_results.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
