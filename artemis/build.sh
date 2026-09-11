#!/usr/bin/env bash
set -euo pipefail

# Build the vLLM CPU image used for Artemis compile/test/benchmark.
# Run from the repository root (the Artemis runner's checkout dir).
#
# v0.29's docker/Dockerfile.cpu bind-mounts `.git` for the Rust build
# (setuptools-scm version derivation):
#     RUN --mount=type=bind,source=.git,target=.git ... bash build_rust.sh
# The Artemis runner unpacks a PLAIN DIRECTORY with no .git, so BuildKit
# fails before any compilation with:
#     failed to compute cache key: "/.git": not found
# Synthesise a throwaway repo in that case: one empty commit tagged with
# the release is all setuptools-scm needs to derive a version. Nothing is
# added to the index, so this costs milliseconds regardless of checkout
# size.
CREATED_GIT=0
if [ ! -d .git ]; then
  git init -q .
  git -c user.email=artemis@local -c user.name=artemis \
      commit -q --allow-empty -m "artemis build context"
  git tag -f v0.29.0 >/dev/null
  CREATED_GIT=1
fi
cleanup_git() { [ "$CREATED_GIT" = "1" ] && rm -rf .git; }
trap cleanup_git EXIT

DOCKER_BUILDKIT=1 docker build \
  --target vllm-openai \
  -t vllm_artemis:cpu \
  -f docker/Dockerfile.cpu .
