#!/usr/bin/env bash
set -uo pipefail

# Artemis benchmark against the COMPANY production trace (nw-benchmark).
#
# Why this exists: artemis/benchmark.sh measures a synthetic 3000/300 conc-8
# shape which is DECODE-dominated. The production trace is PREFILL-dominated
# (518 input tok/s vs 40 output tok/s at saturation, a 13:1 ratio), so
# optimising against the synthetic shape optimises the wrong thing.
#
# Design decisions, each measured rather than assumed:
#  - --rate-multiplier 3: at rate 1.0 the CPU server keeps up and drains its
#    queue, so Output tok/s is pinned by the trace and reads 0.00% for ANY
#    engine change. 3x makes the server the bottleneck.
#  PREFIX CACHING IS ON HERE, DELIBERATELY, AND THAT CHANGES HOW TO READ THIS.
#  The sibling benchmark_nw.sh disables it for measurement hygiene. This
#  variant exists to optimise the cache itself, which is worth ~+14% on this
#  trace (46.16 vs 40.44 tok/s) - an order of magnitude more than any kernel
#  win found in four campaigns.
#
#  THE CATCH: hit ratio is ENDOGENOUS. It swung 0.905-1.486 across runs of the
#  SAME code on the same trace and seed, because a faster server interleaves
#  requests differently, which changes block residency, which changes the hit
#  rate. So throughput here is noisier than the ~3% of the caching-off harness.
#  Duration is raised to 900s to average over more requests, and cache_hit_ratio
#  is printed alongside throughput. READ IT: if a candidate's throughput gain
#  is accompanied by a large hit-ratio swing, suspect the lottery rather than
#  the change, and re-run before believing it.
#
#  Historical note on the original comment this replaced:
#  - --enable-prefix-caching: cache hit ratio swings 0.905-1.486 across
#    runs of the same trace+seed (endogenous: engine speed changes request
#    interleaving changes prefix residency). That variance would drown the
#    signal. Deterministic proxy here; validate winners against the real
#    production config separately.
#  - Inductor ON (no --enforce-eager).
#  - The torch.compile cache is mounted persistently so Inductor work is
#    reused across candidates; C++ kernel edits do not invalidate the graph,
#    so this saves the compile on every candidate after the first.
#
#    *** SUPERSEDED: a shared persistent cache is NOT safe. ***
#    A discovery run measures its baseline FIRST. On an empty cache the
#    baseline pays the whole Inductor compile and every candidate inherits
#    it, which is a systematic ~2% penalty against the baseline -- the
#    benchmark then rewards doing nothing. This voided an entire 10-version
#    run (2026-08-21): a control on unmodified code with a warm cache scored
#    46.24 against a 45.34 cold-cache baseline, beating six of seven
#    "optimised" candidates. Warm it with one throwaway invocation of this
#    script before `artemis discovery create`, or delete $COMPILE_CACHE so
#    every arm is equally cold.
#  - nw-benchmark needs GLIBC 2.38; this host has 2.36, so it runs inside
#    ubuntu:24.04 with --network host.

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
NW_DIR="${NW_DIR:-/home/chrystalla/nw-benchmark-v1.0.0}"
TRACE="${TRACE:-traces/Qwen3.6-35B-A3B_prod_2026-06-12_2h_filtered_24k}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/home/chrystalla/optimisation-orchestrator/.local/models}"
COMPILE_CACHE="$(mktemp -d /tmp/vllm-compile-XXXXXX)"   # FRESH per arm - see note above
DURATION="${DURATION:-900}"
RATE="${RATE:-3}"
WARMUP="${WARMUP:-150}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
# Overridable so an ABBA can point successive arms at two different images
# without rebuilding (and without clobbering the baseline tag that the
# Artemis harness itself depends on).
IMAGE="${IMAGE:-vllm_artemis:cpu}"

mkdir -p "$COMPILE_CACHE"
cleanup() { docker rm -f nw-artemis-server 2>/dev/null >/dev/null; rm -rf "$COMPILE_CACHE" 2>/dev/null; }
trap cleanup EXIT
cleanup

docker run -d --name nw-artemis-server --network host --ipc=host --privileged --shm-size 16g \
  -e HF_HOME=/hf -e VLLM_CPU_KVCACHE_SPACE=20 -e VLLM_CPU_OMP_THREADS_BIND=0-15 \
  -e VLLM_CPU_SGL_KERNEL=1 -e VLLM_CACHE_ROOT=/compile-cache \
  ${ADAPTIVE_BUDGET:+-e VLLM_ADAPTIVE_PREFILL_BUDGET=$ADAPTIVE_BUDGET} \
  ${FAST_GREEDY:+-e VLLM_CPU_FAST_GREEDY=$FAST_GREEDY} \
  ${FUSED_ROUTER:+-e VLLM_CPU_FUSED_ROUTER=$FUSED_ROUTER} \
  ${FUSED_AR_ADD:+-e VLLM_CPU_FUSED_AR_ADD=$FUSED_AR_ADD} \
  ${CANDIDATE_SAMPLE:+-e VLLM_CPU_CANDIDATE_SAMPLE=$CANDIDATE_SAMPLE} \
  ${V2_FAST_SAMPLE:+-e VLLM_CPU_V2_FAST_SAMPLE=$V2_FAST_SAMPLE} \
  -v "$HF_CACHE_DIR:/hf" -v "$COMPILE_CACHE:/compile-cache" \
  "$IMAGE" --model "$MODEL" --host 0.0.0.0 --port 8000 \
  ${PREFIX_MATCH_UNIT:+--prefix-match-unit $PREFIX_MATCH_UNIT} \
  ${MAX_BATCHED:+--max-num-batched-tokens $MAX_BATCHED} \
  ${SSM_DTYPE:+--mamba-ssm-cache-dtype $SSM_DTYPE} \
  --max-model-len 32768 --enable-prefix-caching \
  --enable-prompt-tokens-details --language-model-only >/dev/null 2>&1

echo "waiting for server (Inductor compile may be slow on a cold cache)..."
START=$(date +%s)
while true; do
  ELAPSED=$(( $(date +%s) - START ))
  if [ $ELAPSED -ge $HEALTH_TIMEOUT ]; then
    echo "FAILURE: server not healthy within ${HEALTH_TIMEOUT}s"; docker logs nw-artemis-server 2>&1 | tail -60; exit 1
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' nw-artemis-server 2>/dev/null)" != "true" ]; then
    echo "FAILURE: server container exited after ${ELAPSED}s"; docker logs nw-artemis-server 2>&1 | tail -60; exit 1
  fi
  curl -sf http://localhost:8000/health >/dev/null 2>&1 && { echo "server healthy after ${ELAPSED}s"; break; }
  sleep 5
done

# Warm-up replay. Its results are discarded. Its purpose is to trigger every
# lazy Inductor shape compilation this trace provokes, so that the MEASURED
# replay below runs compile-free. Without this, compilation leaks into the
# measured window and depresses throughput by ~2% - which is larger than the
# effects being measured and, because a run benchmarks its baseline first,
# lands asymmetrically on the baseline.
echo "warm-up replay (${WARMUP}s, results discarded)..."
docker run --rm --network host -v "$NW_DIR:/nw" -w /nw ubuntu:24.04 \
  ./nw-benchmark --trace-dir "$TRACE" --base-url http://localhost:8000/v1 \
  --duration-seconds "$WARMUP" --seed 42 --rate-multiplier "$RATE" --no-primer \
  --out /nw/artemis_nw_warmup.json >/dev/null 2>&1 || true

echo "replaying production trace (${DURATION}s at ${RATE}x)..."
docker run --rm --network host -v "$NW_DIR:/nw" -w /nw ubuntu:24.04 \
  ./nw-benchmark --trace-dir "$TRACE" --base-url http://localhost:8000/v1 \
  --duration-seconds "$DURATION" --seed 42 --rate-multiplier "$RATE" --no-primer \
  --out /nw/artemis_nw_raw.json
BENCH_EXIT=$?
# Preserve the server's own log before the exit trap removes the container.
# Run 11 lost two versions to exactly this: agent-added instrumentation logged
# to the server's stdout, which vanished with the container, so the agent
# could never read its own instrument.
docker logs nw-artemis-server > artemis_nw_server.log 2>&1 || true
[ $BENCH_EXIT -ne 0 ] && { echo "FAILURE: benchmark exit $BENCH_EXIT"; exit $BENCH_EXIT; }

python3 - "$NW_DIR/artemis_nw_raw.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
out = {}
for section in ("metrics", "replay"):
    for k, v in (d.get(section) or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v

# Emit ONE metric. Every numeric key here becomes a fitness schema entry and
# the score is a weighted blend across all of them: an earlier run's schema had
# 21 entries at importances [0.6, 0.0267, 0.0133], so ~40% of fitness came from
# compile_runtime, unit_test_runtime and memory. Rewarding low compile_runtime
# biases the search toward trivial changes. The rest are printed for humans.
metric = "output_tokens_per_second"
if metric not in out:
    raise SystemExit(f"FAILURE: {metric} missing from benchmark output")
json.dump({metric: out[metric]}, open("artemis_results.json", "w"), indent=2)

keys = (metric, "avg_ttft_ms", "avg_tpot_ms", "cache_hit_ratio",
        "total_requests", "total_errors")
print(json.dumps({k: out[k] for k in keys if k in out}, indent=2))
PY
