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

# Each regime's share of TOTAL step CPU, from the production profile.
# fused_experts_cpu is 46.10%, split ~50/50 between its two modes;
# weight_packed_linear is 18.02%, split 61/39 (its large calls dominate).
SHARE = {
    "moe_prefill": 23.05,
    "moe_decode": 23.05,
    "dense_prefill": 10.99,
    "dense_decode": 7.03,
}

# Per-regime times on STOCK code. Used to convert each measurement into a
# FRACTION of its own baseline, so the composite predicts end-to-end impact.
#
# WHY THIS EXISTS - a real mistake, do not repeat it. The original scored
# sum(time * share), which weights by ABSOLUTE MILLISECONDS and so
# over-represents whichever regime is slowest per call. In the attention/GDN
# companion that made a kernel worth 6.16% of step CPU hold 37% of fitness: a
# 26% cut scored -9.6% and read as a major win, while its true worth was 1.6%
# of step CPU, under the ~3% trace noise floor. It measured ZERO end-to-end
# over an ABABAB at n=3 (exp-374). Same flaw, same fix, applied here.
#
# Fixed reference, so host drift shifts the absolute value; only the DIFFERENCE
# between two arms measured close together is meaningful.
BASELINE_MS = {
    "moe_prefill": 21.02,
    "moe_decode": 7.99,
    "dense_prefill": 5.02,
    "dense_decode": 0.281,
}

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
    # Percent of TOTAL step CPU these four regimes consume, at the measured
    # speeds. Each contributes (its time / its stock time) * its share, so a
    # candidate's gain is directly comparable to an end-to-end result.
    fitness = sum(res[k] / BASELINE_MS[k] * SHARE[k] for k in res)
    stock = sum(SHARE.values())

    print(f"{'regime':16s} {'ms/call':>10s} {'stock':>8s} {'share%':>8s} {'cpu%':>7s}")
    for k, v in res.items():
        print(f"{k:16s} {v:10.3f} {BASELINE_MS[k]:8.2f} {SHARE[k]:8.2f} "
              f"{v / BASELINE_MS[k] * SHARE[k]:7.2f}")
    print(f"\ncpu_share_pct {fitness:.3f}   (LOWER IS BETTER; stock = {stock:.2f})")
    print(f"  => predicted end-to-end saving {stock - fitness:+.2f}% of step CPU")
    if abs(stock - fitness) < 3.0:
        print("  NOTE: under the ~3% end-to-end noise floor - would NOT be")
        print("        provable on the production trace even if real.")

    # Sanity against the production profile
    print("\n-- shape validation vs production profile --")
    print(f"  moe_prefill {res['moe_prefill']:.2f} ms   profile p90 ~19.70 ms")
    print(f"  moe_decode  {res['moe_decode']:.2f} ms   profile p50 ~4.49 ms")

    # Emit ONE metric only. Every numeric key in artemis_results.json becomes a
    # fitness schema entry, and the score is a weighted blend across all of
    # them: a previous run's schema had 21 entries at importances
    # [0.6, 0.0267, 0.0133], so ~40% of fitness came from ancillary signals
    # (compile_runtime, unit_test_runtime, memory). Rewarding low
    # compile_runtime actively biases toward smaller, simpler changes.
    # Concentrating on the single composite avoids that. The per-regime
    # numbers are still printed above for humans and for the agent to read.
    # (The alternative, metrics-schema set, requires metric UUIDs which the
    # metrics endpoint does not currently expose.)
    json.dump({"cpu_share_pct": fitness}, open("artemis_results.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
