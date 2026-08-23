#!/usr/bin/env bash
set -euo pipefail

# Correctness gate for prefix-caching / KV block-management changes.
#
# This gate matters more than usual. The metric for that run is end-to-end
# throughput WITH prefix caching enabled, and the cheapest way to "win" is to
# reuse cached blocks that should not be reused - which raises the hit rate,
# raises throughput, and silently corrupts output. These suites are what
# separates a real caching improvement from a wrong one.

docker run --rm --entrypoint bash -e VLLM_CPU_SGL_KERNEL=1 \
  -v "$(pwd)/tests:/tests:ro" "${IMAGE:-vllm_artemis:cpu}" -c \
  "pip install -q pytest 2>/dev/null; python3 -m pytest \
     /tests/v1/core/test_prefix_caching.py \
     /tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py \
     /tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py \
     /tests/v1/core/test_kv_cache_utils.py \
     /tests/v1/core/test_deferred_block_free.py \
     -q --no-header -p no:cacheprovider"
