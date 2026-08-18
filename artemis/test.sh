#!/usr/bin/env bash
set -euo pipefail

# Correctness gate: run the fused_experts_cpu kernel test inside the image
# built by artemis/build.sh. Fast (~1 min), no model download needed.

docker run --rm \
  --entrypoint python3 \
  -e VLLM_CPU_SGL_KERNEL=1 \
  -v "$(pwd)/artemis/test_moe_kernel.py:/test_moe_kernel.py:ro" \
  vllm_artemis:cpu \
  /test_moe_kernel.py
