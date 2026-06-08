#!/usr/bin/env bash
set -euo pipefail

# Build the vLLM CPU image used for Artemis benchmarks.
#
# Run from the repository root so the Docker build context picks up the
# vLLM source tree.

DOCKER_BUILDKIT=1 docker build \
  --build-arg VLLM_VERSION_OVERRIDE=0.1.0+cpu \
  --target vllm-openai \
  -t vllm_artemis:cpu \
  -f docker/Dockerfile.cpu .
