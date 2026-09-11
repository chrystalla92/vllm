#!/usr/bin/env bash
set -uo pipefail

# Artemis benchmark for the vLLM CPU image on mc-010-gqzxu
# (GCP c4-standard-32, Emerald Rapids, 16 physical cores, single socket).
#
# Mirrors the exp-360..364 measurement config: Qwen3.6-35B-A3B bf16, TP=1,
# OMP bind 0-15, SGL kernels on, chunked prefill, block 128, kvcache 20 GB,
# mbt 8192, random 3000-in/300-out at concurrency 8. Writes numeric metrics
# to artemis_results.json (higher output_throughput = better).

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-3000}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-300}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/home/chrystalla/optimisation-orchestrator/.local/models}"

cleanup() {
  docker kill vllm-artemis-server 2>/dev/null
  docker rm -f vllm-artemis-server 2>/dev/null
  docker network rm vllm-artemis-net 2>/dev/null || true
}
trap cleanup EXIT

cleanup
docker network create vllm-artemis-net 2>/dev/null || true

docker run -d --name vllm-artemis-server --network vllm-artemis-net \
  --ipc=host --privileged \
  -e HF_HOME=/hf \
  -e VLLM_CPU_KVCACHE_SPACE=20 \
  -e VLLM_CPU_OMP_THREADS_BIND=0-15 \
  -e VLLM_CPU_SGL_KERNEL=1 \
  -v "$HF_CACHE_DIR:/hf" \
  vllm_artemis:cpu \
  --model "$MODEL" \
  --dtype bfloat16 \
  --block-size 128 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --enable-chunked-prefill \
  --language-model-only \
  --max-model-len 4096

docker logs -f vllm-artemis-server > /tmp/vllm-artemis-server.log 2>&1 &

TIMEOUT=1800
START_TIME=$(date +%s)
echo "Starting vLLM server..."
while true; do
  ELAPSED=$(( $(date +%s) - START_TIME ))
  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "FAILURE: server not healthy within ${TIMEOUT}s"
    tail -100 /tmp/vllm-artemis-server.log
    exit 1
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' vllm-artemis-server 2>/dev/null)" != "true" ]; then
    echo "FAILURE: server container exited after ${ELAPSED}s"
    tail -100 /tmp/vllm-artemis-server.log
    exit 1
  fi
  if docker run --rm --network vllm-artemis-net --entrypoint curl vllm_artemis:cpu \
       -sf http://vllm-artemis-server:8000/health > /dev/null 2>&1; then
    echo "Server healthy after ${ELAPSED}s"
    break
  fi
  sleep 5
done

echo "Warmup pass..."
docker run --rm --network vllm-artemis-net \
  -e HF_HOME=/hf -v "$HF_CACHE_DIR:/hf" \
  --entrypoint vllm vllm_artemis:cpu \
  bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --base-url http://vllm-artemis-server:8000 \
  --model "$MODEL" \
  --num-prompts 4 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 32 \
  --max-concurrency 4 \
  --request-rate inf \
  --ignore-eos > /dev/null 2>&1

echo "Measured benchmark: ${NUM_PROMPTS}x ${RANDOM_INPUT_LEN}/${RANDOM_OUTPUT_LEN} conc ${MAX_CONCURRENCY}..."
docker run --rm --network vllm-artemis-net \
  -e HF_HOME=/hf -v "$HF_CACHE_DIR:/hf" \
  -v "$(pwd):/results" \
  --entrypoint vllm vllm_artemis:cpu \
  bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --base-url http://vllm-artemis-server:8000 \
  --model "$MODEL" \
  --num-prompts "$NUM_PROMPTS" \
  --dataset-name random \
  --random-input-len "$RANDOM_INPUT_LEN" \
  --random-output-len "$RANDOM_OUTPUT_LEN" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --request-rate inf \
  --ignore-eos \
  --save-result \
  --result-dir /results \
  --result-filename artemis_results_raw.json
BENCH_EXIT=$?
[ $BENCH_EXIT -ne 0 ] && exit $BENCH_EXIT

python3 -c "
import json
d = json.load(open('artemis_results_raw.json'))
filtered = {k: v for k, v in d.items() if isinstance(v, (int, float))}
json.dump(filtered, open('artemis_results.json', 'w'), indent=2)
print(json.dumps({k: filtered[k] for k in ('output_throughput', 'mean_tpot_ms', 'mean_ttft_ms') if k in filtered}, indent=2))
"
