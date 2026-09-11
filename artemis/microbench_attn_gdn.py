"""Microbenchmark for the CPU attention and GDN kernels at PRODUCTION shapes.

Companion to microbench_moe_dense.py, covering the other 27% of the
production-trace profile: csrc/cpu/cpu_attn.cpp and
csrc/cpu/sgl-kernels/fla.cpp.

Why this exists
---------------
End-to-end throughput on the production trace has a ~3% noise floor and costs
~17 min per candidate, while a realistic kernel win is 1-3% end to end. The
instrument cannot resolve the effect. Calling the kernels directly removes the
server, arrival-rate replay, prefix cache, lazy Inductor compilation and
concurrency jitter, so a 15% kernel win reads as ~15%, in ~90 seconds.

Shapes are taken from the measured production profile, not invented
-------------------------------------------------------------------
Qwen3.6-35B-A3B: 40 layers on a 3-linear / 1-full repeating pattern, so 10
full-attention and 30 linear-attention (GDN) layers.
  full attention : 16 q heads, 2 kv heads, head_dim 256
  GDN            : 16 key heads / 128, 32 value heads / 128, conv kernel 4
Prefill chunk M=2048 (vLLM default max_num_batched_tokens, not overridden).
Decode concurrency 42, mean prompt 6271 tokens (median 2813, max 23907).

Call counts observed in a 25-forward-pass profiling window corroborate the
layer split: attention 250 = 25 x 10; gating update 750 = 25 x 30; chunk 150
and conv1d_fwd 150 = 5 prefill steps x 30.

Profiled per-call costs to validate against:
  cpu_attention_with_kv_cache            10.65%  p50 3.87 ms, p90 6.05, p99 38.8
  fused_sigmoid_gating_delta_rule_update  7.34%  p50 1.49 ms (strikingly uniform)
  chunk_gated_delta_rule_cpu              6.16%  p25 4.80, p50 5.85, p90 12.68
If measured times are far from these, the shapes are unrepresentative and the
benchmark is measuring fiction - check before trusting any result.

Measured calibration and repeatability (16 cores, quiet host, 3 runs)
--------------------------------------------------------------------
  regime         measured   profile   ratio   run-to-run spread
  gdn_chunk        5.85 ms   5.85      1.00x   1.6%
  attn_prefill     4.96 ms   6.05(p90) 0.82x   2.1%
  attn_decode      3.03 ms   3.87      0.78x   3.1%  (post kv-split fix)
  gdn_update       0.98 ms   1.49      0.66x   4.7%
  weighted total  101 ms       -         -     1.35%
All four land within ~1.3x of production, and the composite resolves changes
well under the 3% end-to-end noise floor. (An earlier revision measured
attn_decode at 6.06ms/1.57x: that was enable_kv_split=False disabling the
scheduler's KV rebalancing - remainder sequences tail-load the last thread,
up to 2.5x inflation at N % threads != 0. Production defaults the split ON;
so does this benchmark now.)

Two measurement decisions were forced by data, not taste:
  - The cache flush (see _flush_cache). Without it gdn_update read 0.42 ms,
    3.5x faster than profile, because one tensor stayed LLC-resident.
  - Median rather than min (see _time). Under flushing, min selected whichever
    rep stayed warmest and gdn_update swung 24% run to run; median gives 4.7%.

Fitness: weighted_total_ms, LOWER IS BETTER, weighting each regime by its share
of the production profile. Attention's 10.65% is split 57/43 between prefill and
decode, matching the observation that its largest quartile of calls holds 56.9%
of its time. Prefill and decode are both included so a prefill win that wrecks
decode is penalised rather than hidden.

REQUIRES A QUIET MACHINE. Contention destroys the signal: in the MoE/dense
companion, the same code measured 34.8 / 25.9 / 21.5 ms for one regime purely
depending on what else was running.

Emits ONE metric. Every numeric key in artemis_results.json becomes a fitness
schema entry and the score is a blend across all of them; a previous run's
schema had 21 entries at importances [0.6, 0.0267, 0.0133], so ~40% of fitness
came from compile_runtime, unit_test_runtime and memory. Rewarding low
compile_runtime biases toward trivial changes.
"""

import json
import os
import sys
import time

import torch

from vllm import _custom_ops as ops
from vllm._custom_ops import (
    cpu_attention_with_kv_cache,
    cpu_attn_get_scheduler_metadata,
    cpu_attn_reshape_and_cache,
)
from vllm.v1.attention.backends.cpu_attn import _get_attn_isa

if torch.cpu._is_amx_tile_supported():
    torch.cpu._init_amx()

DTYPE = torch.bfloat16

# full-attention layers
Q_HEADS, KV_HEADS, HEAD_SIZE = 16, 2, 256
BLOCK_SIZE = 128
# GDN (linear-attention) layers
QK_HEADS, V_HEADS = 16, 32
QK_DIM, V_DIM = 128, 128

M_PREFILL = 2048          # vLLM default chunked-prefill budget
N_DECODE = 42             # observed concurrency at saturation
KV_LEN_DECODE = 6271      # mean production prompt length
KV_LEN_PREFILL = 2813     # median prompt; prefill chunks attend over accumulated context
STATE_POOL = 256          # max_num_seqs: production sizes the GDN state pool
                          # for every slot, not just the active batch. With
                          # 32 heads x 128 x 128 fp32 that is ~536 MB, and the
                          # kernel gathers scattered rows out of it - a very
                          # different cache regime from a compact 42-row pool.

# Profile shares (% of total CPU time). Attention split 57/43 prefill/decode.
# Each regime's share of TOTAL step CPU, from the production profile.
SHARE = {
    "attn_prefill": 6.07,
    "attn_decode": 4.58,
    "gdn_update": 7.34,
    "gdn_chunk": 6.16,
}

# Per-regime times on STOCK code, measured on this host from a clean baseline
# image. Used to convert each regime's measurement into a FRACTION of its own
# baseline, so the composite predicts end-to-end impact.
#
# WHY THIS EXISTS - a real mistake, do not repeat it. The first version scored
# `sum(time * share)`, which weights by ABSOLUTE MILLISECONDS and therefore
# over-represents whichever regime is slowest per call. gdn_chunk is 6.16% of
# step CPU but held 37% of that composite. A change that cut gdn_chunk 26%
# scored -9.6% on the composite and looked like a major win; its true worth is
# 0.26 * 6.16 = 1.6% of step CPU, below the ~3% trace-benchmark noise floor.
# It duly measured ZERO end-to-end over an ABABAB at n=3 (exp-374). Two hours
# of trace replay to discover the metric was lying about magnitude.
#
# These are a FIXED REFERENCE, so host drift shifts the absolute cpu_share_pct
# (stock code read 22.86 against a nominal 24.15 on a fast evening). Only the
# DIFFERENCE between two arms measured close together is meaningful - which is
# the same discipline ABBA already enforces.
BASELINE_MS = {
    "attn_prefill": 5.21,
    # 3.03, not the earlier 6.02: that figure was measured with
    # enable_kv_split=False (see the metadata call), which tail-loaded the
    # scheduler and inflated this regime 2x. With the production setting the
    # regime matches the profiled p50 (~3.87ms, mean-length contexts here).
    "attn_decode": 3.03,
    "gdn_update": 1.02,
    "gdn_chunk": 6.23,
}

# 20, not 8: at 8 reps the weighted total spread was 3.2% - no better than the
# end-to-end harness this replaces. 20 brings it to 1.35%.
REPS = int(os.environ.get("MICROBENCH_REPS", "20"))
WARMUP = int(os.environ.get("MICROBENCH_WARMUP", "2"))
ESTIMATOR = os.environ.get("MICROBENCH_ESTIMATOR", "median")
# gdn_update is the shortest kernel (~0.7 ms), so cold-miss variance dominates
# it; it also carries the largest weight. Give it more reps than the rest.
GDN_UPDATE_REPS = REPS * 3


# ~512 MB, comfortably larger than this class of CPU's shared LLC.
_FLUSH = torch.empty(128 * 1024 * 1024, dtype=torch.float32)


def _flush_cache():
    """Evict the LLC between timed reps.

    Production never calls these kernels back-to-back on warm state: 30 GDN
    layers each touch ~88 MB of state per step, so by the time a layer is
    revisited its state has been evicted. Repeating one call in a loop leaves
    everything cache-resident and measured gdn_update ~3.5x faster than the
    profile (0.42 ms vs p50 1.49 ms). Flushing restores the cold-start regime,
    so optimisations that target cold-state access can actually register.
    Runs outside the timed region.
    """
    _FLUSH.uniform_(0.0, 1.0)


def _time(fn, reps=None):
    """Cold-start time per call, in ms.

    Estimator is median, not min. With a cache flush every rep should be
    equally cold, so min would pick whichever rep the flush happened to leave
    warmest - a bias back toward the cache-resident regime the flush exists to
    remove. Median is robust to that and to occasional scheduler hits.
    """
    reps = reps or REPS
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(reps):
        _flush_cache()
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    if ESTIMATOR == "min":
        return samples[0]
    return samples[len(samples) // 2]


def _build_attn(query_lens, kv_lens):
    """Paged-attention inputs, mirroring tests/kernels/attention/test_cpu_attn.py."""
    num_seqs = len(query_lens)
    total_q = sum(query_lens)
    max_kv = max(kv_lens)
    num_blocks = max(4, (max_kv + BLOCK_SIZE - 1) // BLOCK_SIZE * num_seqs + 8)

    query = torch.randn(total_q, Q_HEADS, HEAD_SIZE, dtype=DTYPE) * 0.1
    kv = torch.randn(2, num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_SIZE, dtype=DTYPE) * 0.1
    key_cache, value_cache = kv.unbind(0)

    packed = torch.empty(num_blocks, KV_HEADS, BLOCK_SIZE, HEAD_SIZE * 2, dtype=DTYPE)
    packed = packed.view((num_blocks, KV_HEADS, BLOCK_SIZE * 2, -1))
    packed_k, packed_v = packed.chunk(2, dim=2)

    cu_q = torch.tensor([0] + list(query_lens), dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    max_blocks_per_seq = (max_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_table = torch.randint(
        0, num_blocks, (num_seqs, max_blocks_per_seq), dtype=torch.int32
    )

    isa = _get_attn_isa(DTYPE, BLOCK_SIZE)
    cpu_attn_reshape_and_cache(
        key=key_cache.view(-1, KV_HEADS, HEAD_SIZE),
        value=value_cache.view(-1, KV_HEADS, HEAD_SIZE),
        key_cache=packed_k,
        value_cache=packed_v,
        slot_mapping=torch.arange(0, num_blocks * BLOCK_SIZE, dtype=torch.int64),
        isa=isa,
    )
    metadata = cpu_attn_get_scheduler_metadata(
        num_reqs=num_seqs,
        num_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_SIZE,
        seq_lens=kv_lens_t,
        dtype=DTYPE,
        query_start_loc=cu_q,
        causal=True,
        sliding_window_size=-1,
        isa=isa,
        # Production default (VLLM_CPU_ATTN_SPLIT_KV=1). With False the
        # scheduler cannot rebalance KV across threads: remainder sequences
        # tail-load the last thread and attn_decode inflates up to 2.5x at
        # N % threads != 0 (measured: N=42 6.29ms vs 3.31ms; a phantom
        # "sawtooth bug" that was really this flag).
        enable_kv_split=True,
    )
    out = torch.empty_like(query)
    scale = HEAD_SIZE**-0.5

    def run():
        cpu_attention_with_kv_cache(
            query=query, key_cache=packed_k, value_cache=packed_v, output=out,
            query_start_loc=cu_q, seq_lens=kv_lens_t, scale=scale, causal=True,
            alibi_slopes=None, sliding_window=-1, block_table=block_table,
            softcap=0, scheduler_metadata=metadata, s_aux=None,
        )

    return run


def _gdn_inputs(num_tokens):
    q = torch.randn(1, num_tokens, QK_HEADS, QK_DIM, dtype=DTYPE) * 0.1
    k = torch.randn(1, num_tokens, QK_HEADS, QK_DIM, dtype=DTYPE) * 0.1
    v = torch.randn(1, num_tokens, V_HEADS, V_DIM, dtype=DTYPE) * 0.1
    a = torch.randn(num_tokens, V_HEADS, dtype=DTYPE) * 0.1
    b = torch.randn(num_tokens, V_HEADS, dtype=DTYPE) * 0.1
    A_log = torch.randn(V_HEADS, dtype=torch.float32)
    dt_bias = torch.randn(V_HEADS, dtype=DTYPE)
    return q, k, v, a, b, A_log, dt_bias


def _build_gdn_update(batch):
    """Decode path: one token per sequence, state updated in place."""
    q, k, v, a, b, A_log, dt_bias = _gdn_inputs(batch)
    # Full-size pool with scattered indices, as production has: a compact
    # 42-row pool with sequential indices is far more cache-friendly than the
    # real thing and measured ~3.5x too fast against the profile.
    state = torch.randn(STATE_POOL, V_HEADS, QK_DIM, V_DIM, dtype=torch.float32) * 0.1
    idx = torch.randperm(STATE_POOL, dtype=torch.int64)[:batch].to(torch.int32)
    cu = torch.arange(batch + 1, dtype=torch.int32)

    def run():
        ops.fused_sigmoid_gating_delta_rule_update_cpu(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state, initial_state_indices=idx,
            cu_seqlens=cu, use_qk_l2norm_in_kernel=True,
        )

    return run


def _build_gdn_chunk(num_tokens):
    """Prefill path: one long sequence through the chunked kernel."""
    q, k, v, a, b, A_log, dt_bias = _gdn_inputs(num_tokens)
    cu = torch.tensor([0, num_tokens], dtype=torch.int32)
    state = torch.randn(STATE_POOL, V_HEADS, QK_DIM, V_DIM, dtype=torch.float32) * 0.1
    idx = torch.randint(0, STATE_POOL, (1,), dtype=torch.int32)
    # gating: g and beta derived as the model does, shapes [1, tokens, heads]
    # Mirrors ref_gdn_gating in tests/kernels/mamba/cpu/test_cpu_gdn_ops.py:
    # g stays float32, beta is cast back to b's dtype (bf16). Passing beta as
    # float32 makes the kernel raise "Input tensor dtype mismatch".
    g = (-torch.exp(A_log.float()) * torch.nn.functional.softplus(
        a.float() + dt_bias.float(), beta=1.0, threshold=20.0)).unsqueeze(0)
    beta = torch.sigmoid(b.float()).to(dtype=b.dtype).unsqueeze(0)

    def run():
        ops.chunk_gated_delta_rule_cpu(
            query=q, key=k, value=v, g=g, beta=beta,
            initial_state=state, output_final_state=True, cu_seqlens=cu,
            head_first=False, use_qk_l2norm_in_kernel=True,
            initial_state_indices=idx,
        )

    return run


def main() -> int:
    torch.manual_seed(0)
    res = {
        # chunked prefill: one 2048-token chunk attending over its own context
        "attn_prefill": _time(_build_attn([M_PREFILL], [KV_LEN_PREFILL])),
        # decode: 42 concurrent single-token queries over mean-length contexts
        "attn_decode": _time(_build_attn([1] * N_DECODE, [KV_LEN_DECODE] * N_DECODE)),
        "gdn_update": _time(_build_gdn_update(N_DECODE), reps=GDN_UPDATE_REPS),
        "gdn_chunk": _time(_build_gdn_chunk(M_PREFILL)),
    }
    # Percent of TOTAL step CPU these four regimes consume, at the measured
    # speeds. Each regime contributes (its time / its stock time) * its share,
    # so a 26% cut in a regime worth 6.16% of the step moves this by 1.6 -
    # directly comparable to what an end-to-end benchmark could show.
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

    print("\n-- shape validation vs production profile --")
    print(f"  attn_prefill {res['attn_prefill']:6.2f} ms   profile p90 ~6.05 ms (p99 38.8)")
    print(f"  attn_decode  {res['attn_decode']:6.2f} ms   profile p50 ~3.87 ms")
    print(f"  gdn_update   {res['gdn_update']:6.2f} ms   profile p50 ~1.49 ms")
    print(f"  gdn_chunk    {res['gdn_chunk']:6.2f} ms   profile p50 ~5.85 ms")

    json.dump({"cpu_share_pct": fitness}, open("artemis_results.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
