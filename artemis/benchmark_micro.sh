#!/usr/bin/env bash
set -euo pipefail

# FAST fitness signal: calls the MoE and dense CPU kernels directly at
# production shapes instead of replaying the full trace through a server.
#
# Why: the end-to-end trace benchmark has a ~3% noise floor and costs ~17 min
# per candidate, while a realistic kernel win is 1-3% end to end. This costs
# ~1.5 min and measures <1% run-to-run (weighted total spread 0.96% over three
# reps on an idle host), so it resolves effects the trace benchmark cannot.
#
# Shapes come from the measured production profile: prefill M=2048 (vLLM's
# default max_num_batched_tokens) -> MoE avg_M=64 -> AMX brgemm; decode M=42
# (observed saturation concurrency) -> MoE avg_M=1 -> AVX512 tinygemm. Dense
# dispatches on M>4 so both regimes use brgemm.
#
# CAVEAT: absolute decode cost runs ~1.9x the profiled median because this uses
# uniform random routing (~57 distinct experts) while production routing
# concentrates on a hot subset. Relative improvements should still transfer.
# Winners MUST be confirmed with the end-to-end trace benchmark (ABBA) before
# being believed - this is a search signal, not proof.
#
# REQUIRES A QUIET MACHINE. Contention destroys it: under a competing workload
# the same code measured 34.8 / 25.9 / 21.5 ms for the same regime.

docker run --rm --entrypoint python3 \
  --user "$(id -u):$(id -g)" \
  -e VLLM_CPU_SGL_KERNEL=1 -e OMP_NUM_THREADS=16 \
  -e MICROBENCH_REPS="${MICROBENCH_REPS:-8}" \
  --cpuset-cpus 0-15 \
  -v "$(pwd)/artemis/microbench_moe_dense.py:/mb.py:ro" \
  -v "$(pwd):/out" -w /out \
  vllm_artemis:cpu /mb.py
