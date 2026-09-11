#!/usr/bin/env bash
set -euo pipefail

# Correctness gate for the CPU attention and GDN (linear-attention) kernels.
# Uses vLLM's own upstream suites, which cover csrc/cpu/cpu_attn.cpp and
# csrc/cpu/sgl-kernels/fla.cpp. Baseline on this image: 1677 passed,
# 990 skipped, ~200s.

docker run --rm --entrypoint bash -e VLLM_CPU_SGL_KERNEL=1 \
  -v "$(pwd)/tests:/tests:ro" "${IMAGE:-vllm_artemis:cpu}" -c \
  "pip install -q pytest 2>/dev/null; python3 -m pytest \
     /tests/kernels/attention/test_cpu_attn.py \
     /tests/kernels/mamba/cpu/test_cpu_gdn_ops.py \
     -q --no-header -p no:cacheprovider"
