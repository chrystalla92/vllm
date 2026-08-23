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
     -k 'not test_mla_with_incompatible_swa_uses_one_full_allocation_group' \
     -q --no-header -p no:cacheprovider"

# The deselected test ERRORS on UNMODIFIED code in this environment - it covers
# MLA with sliding-window attention, neither of which this model uses (it is
# GDN + full attention). pytest exits non-zero on a collection error, so with
# set -e it failed the baseline and killed an entire discovery run at 0/8
# before a single candidate was measured. Filtered by test NAME rather than
# --deselect because pytest resolves nodeids against its own rootdir here, so a
# path-qualified deselect silently fails to match. The filter names the single
# test, so any OTHER regression in this file still fails the gate.
# Baseline: 202 passed, 1 deselected.
