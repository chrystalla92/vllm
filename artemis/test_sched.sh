#!/usr/bin/env bash
set -euo pipefail

# Correctness gate for scheduler / batching changes. Uses vLLM's own v1 core
# suites. Scheduling may change HOW work is batched but must never change what
# is computed, so these tests are the line a candidate cannot cross.

docker run --rm --entrypoint bash -e VLLM_CPU_SGL_KERNEL=1 \
  -v "$(pwd)/tests:/tests:ro" "${IMAGE:-vllm_artemis:cpu}" -c \
  "pip install -q pytest 2>/dev/null; python3 -m pytest \
     /tests/v1/core/test_scheduler.py \
     /tests/v1/core/test_async_scheduler.py \
     /tests/v1/core/test_mamba_align_chunk_split.py \
     /tests/v1/core/test_output.py \
     -q --no-header -p no:cacheprovider"
