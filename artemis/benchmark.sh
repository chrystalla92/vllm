#!/usr/bin/env bash
set -uo pipefail

# Run an Artemis benchmark against the vLLM CPU image built by build.sh.
#
# Starts the vLLM OpenAI server in a container, waits for it to become
# healthy, runs `vllm bench serve` against it, filters the raw results down
# to numeric metrics, and tears everything down.

# Benchmark parameters (override via environment variables).
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-512}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-512}"

docker network create vllm-bench-net 2>/dev/null || true

CONTAINER_ID=$(docker run -d --name vllm-server --network vllm-bench-net -p 8000:8000 --ipc=host --privileged \
  -e VLLM_CPU_KVCACHE_SPACE=40 \
  -e VLLM_CPU_OMP_THREADS_BIND=auto \
  vllm_artemis:cpu \
  --model "$MODEL" \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --block-size 128 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 256 \
  --enable-chunked-prefill \
  --language-model-only \
  --max-model-len 4096)

docker logs -f vllm-server > /tmp/vllm-server.log 2>&1 &

TIMEOUT=1800
START_TIME=$(date +%s)
echo "Starting vLLM server (container: $CONTAINER_ID)..."

while true; do
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))

  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "FAILURE: Server did not become healthy within $TIMEOUT seconds"
    tail -200 /tmp/vllm-server.log
    docker kill vllm-server 2>/dev/null
    docker rm -f vllm-server 2>/dev/null
    docker network rm vllm-bench-net 2>/dev/null || true
    exit 1
  fi

  if [ "$(docker inspect -f '{{.State.Running}}' vllm-server 2>/dev/null)" != "true" ]; then
    EXITCODE=$(docker inspect -f '{{.State.ExitCode}}' vllm-server 2>/dev/null)
    echo "FAILURE: container exited (exitCode=$EXITCODE) after $ELAPSED seconds"
    tail -200 /tmp/vllm-server.log
    docker rm -f vllm-server 2>/dev/null
    docker network rm vllm-bench-net 2>/dev/null || true
    exit 1
  fi

  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "SUCCESS: Server is healthy after $ELAPSED seconds"
    break
  fi

  echo "Waiting for server... ($ELAPSED/$TIMEOUT seconds)"
  sleep 5
done

echo "Running vLLM benchmark from container..."
docker run --rm --network vllm-bench-net \
  -v "$(pwd):/results" \
  --entrypoint vllm vllm_artemis:cpu \
  bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --base-url http://vllm-server:8000 \
  --model "$MODEL" \
  --num-prompts "$NUM_PROMPTS" \
  --dataset-name random \
  --random-input-len "$RANDOM_INPUT_LEN" \
  --random-output-len "$RANDOM_OUTPUT_LEN" \
  --max-concurrency 32 \
  --request-rate inf \
  --ignore-eos \
  --save-result \
  --result-dir /results \
  --result-filename artemis_results_raw.json
BENCH_EXIT=$?

python3 -c "
import json
d = json.load(open('artemis_results_raw.json'))
filtered = {k: v for k, v in d.items() if isinstance(v, (int, float))}
json.dump(filtered, open('artemis_results.json', 'w'), indent=2)
"

docker kill vllm-server 2>/dev/null; docker rm -f vllm-server 2>/dev/null; docker network rm vllm-bench-net 2>/dev/null || true

exit $BENCH_EXIT
