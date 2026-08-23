#!/usr/bin/env bash
set -euo pipefail

# FAST build for PYTHON-ONLY changes.
#
# docker/Dockerfile.cpu does `COPY . .` (line 151) before building the wheel,
# so editing a single .py file invalidates that layer and triggers a full C++
# recompile - roughly 10-15 minutes to change one line of scheduler logic.
# That makes Python-level discovery runs (scheduler, KV-cache manager, block
# pool) impractically slow.
#
# This instead starts FROM an already-built image and overwrites the installed
# vllm Python package in place. Takes ~30 s.
#
# ONLY VALID FOR PYTHON-ONLY CHANGES. If the candidate touches anything under
# csrc/ the compiled extension will NOT be rebuilt and the change will be
# silently ignored - which would read as "no effect" rather than as an error.
# The guard below fails loudly in that case rather than reporting a false
# neutral, which is the failure mode that matters: a silently-ignored change
# looks exactly like an honest null result.

BASE_IMAGE="${BASE_IMAGE:-vllm-nw:base64}"
OUT_IMAGE="${OUT_IMAGE:-vllm_artemis:cpu}"
SITE=/opt/venv/lib/python3.12/site-packages/vllm

# Refuse to run if compiled sources differ from what the base image was built
# from. Checksum manifest rather than `git diff`: the Artemis runner unpacks a
# plain directory with NO .git, so any git-based check fails closed and blocks
# every build.
MANIFEST="artemis/csrc.sha256"
csrc_hash() {
  find csrc CMakeLists.txt setup.py -type f 2>/dev/null \
    | LC_ALL=C sort | xargs sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1
}
if [ -f "$MANIFEST" ]; then
  WANT="$(cut -d' ' -f1 < "$MANIFEST")"
  GOT="$(csrc_hash)"
  if [ "$WANT" != "$GOT" ]; then
    echo "FAILURE: build_py.sh cannot be used - csrc/, CMakeLists.txt or setup.py"
    echo "         differs from the base image ($GOT != $WANT)."
    echo "         Compiled sources need the full build (artemis/build.sh);"
    echo "         this fast path would compile them out, and the change would"
    echo "         read as a false null rather than as an error."
    exit 1
  fi
else
  echo "FAILURE: $MANIFEST missing - cannot verify compiled sources are unchanged."
  exit 1
fi

echo "fast Python-only build: $BASE_IMAGE -> $OUT_IMAGE"
BUILD_DIR="$(mktemp -d /tmp/vllm-pybuild-XXXXXX)"
trap 'rm -rf "$BUILD_DIR"' EXIT
cp -r vllm "$BUILD_DIR/vllm"

cat > "$BUILD_DIR/Dockerfile" <<EOF
FROM $BASE_IMAGE
COPY vllm $SITE
EOF

DOCKER_BUILDKIT=1 docker build -q -t "$OUT_IMAGE" "$BUILD_DIR" >/dev/null
echo "built $OUT_IMAGE"

# Prove the interpreter actually loads and the native extension still resolves.
docker run --rm --entrypoint python3 "$OUT_IMAGE" -c \
  "import vllm, vllm._C; from vllm.v1.core.sched.scheduler import Scheduler; print('import ok')"
