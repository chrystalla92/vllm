#!/usr/bin/env bash
set -euo pipefail

# FAST fitness signal for the CPU attention and GDN kernels: calls them
# directly at production shapes instead of replaying the trace through a
# server. Companion to benchmark_micro.sh, covering the other 27% of the
# profile (cpu_attn.cpp 10.65% + fla.cpp family ~16.4%).
#
# Why: the end-to-end trace benchmark has a ~3% noise floor and costs ~17 min
# per candidate, while a realistic kernel win is 1-3% end to end. This costs
# ~3 min and measures 1.35% run-to-run on the weighted total, so it resolves
# effects the trace benchmark cannot.
#
# Shapes come from the measured production profile: prefill chunk M=2048 over
# a median-length 2813-token context; decode 42 concurrent single-token queries
# over mean-length 6271-token contexts; GDN over a full 256-slot state pool
# with scattered indices. Calibration against profiled per-call medians is
# printed by the script itself on every run - read it.
#
# TWO NON-OBVIOUS MEASUREMENT DECISIONS, both forced by data:
#   1. The LLC is flushed between reps. Production never calls these kernels on
#      warm state: 30 GDN layers each touch ~88 MB per step, so a layer's state
#      is always evicted before it is revisited. Without the flush gdn_update
#      read 0.42 ms against a profiled 1.49 ms, and any optimisation targeting
#      cold-state access would have registered as worthless.
#   2. Timing is the MEDIAN of reps, not the min. Under flushing, min selects
#      whichever rep the flush left warmest - biasing back toward the very
#      cache-resident regime the flush exists to remove - and gdn_update swung
#      24% run to run. Median holds it to 4.7%.
#
# CAVEAT: this is a SEARCH SIGNAL, not proof. attn_decode reads ~1.6x the
# profiled median because it uses mean rather than median prompt length, and no
# microbenchmark reproduces inter-layer cache interference exactly. Winners MUST
# be confirmed end to end on the real trace (ABBA) before being believed.
#
# REQUIRES A QUIET MACHINE. Contention destroys it: in the MoE/dense companion
# the same code measured 34.8 / 25.9 / 21.5 ms for one regime purely depending
# on what else was running. Check `docker ps` and the process list first -
# load average lags and will read low while a runner is still busy.

docker run --rm --entrypoint python3 \
  --user "$(id -u):$(id -g)" \
  -e VLLM_CPU_SGL_KERNEL=1 -e OMP_NUM_THREADS=16 \
  -e MICROBENCH_REPS="${MICROBENCH_REPS:-20}" \
  -e MICROBENCH_ESTIMATOR="${MICROBENCH_ESTIMATOR:-median}" \
  --cpuset-cpus 0-15 \
  -v "$(pwd)/artemis/microbench_attn_gdn.py:/mb.py:ro" \
  -v "$(pwd):/out" -w /out \
  vllm_artemis:cpu /mb.py
