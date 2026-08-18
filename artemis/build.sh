#!/usr/bin/env bash
set -euo pipefail

# Build the vLLM CPU image used for Artemis compile/test/benchmark.
# Run from the repository root (the Artemis runner's checkout dir).

DOCKER_BUILDKIT=1 docker build \
  --build-arg VLLM_VERSION_OVERRIDE=0.27.1+cpu \
  --target vllm-openai \
  -t vllm_artemis:cpu \
  -f docker/Dockerfile.cpu .
