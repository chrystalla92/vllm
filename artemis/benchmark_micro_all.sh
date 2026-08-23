#!/usr/bin/env bash
set -euo pipefail

# Combined fast fitness signal covering ~88% of the production-trace step CPU:
#   moe_prefill + moe_decode + dense_prefill + dense_decode   64.12%
#   attn_prefill + attn_decode + gdn_update + gdn_chunk       24.15%
#
# Emits ONE metric, cpu_share_pct = percent of TOTAL step CPU these eight
# regimes consume at the measured speeds. LOWER IS BETTER; stock is ~88.27.
# Because each regime contributes (its time / its stock time) * its profile
# share, a 1.0 drop here means ~1% of total step CPU saved - directly
# comparable to what an end-to-end trace run could show.
#
# READ THIS BEFORE TRUSTING A BIG NUMBER. The earlier version of these scripts
# scored sum(time * share), weighting by absolute milliseconds, which
# over-represented whichever regime was slowest per call: a kernel worth 6.16%
# of step CPU held 37% of fitness. A change that cut it 26% scored -9.6% and
# looked like a major win; it measured ZERO end-to-end over an ABABAB at n=3
# (exp-374). Direction was right, magnitude wrong by ~6x. The current scoring
# would have reported 0.86% and flagged it as sub-noise-floor.
#
# Anything under ~3% here is NOT provable on the production trace, whatever
# the unit tests say. Prefer candidates whose predicted saving clears that.
#
# REQUIRES A QUIET MACHINE. Under contention the same code has measured
# 34.8 / 25.9 / 21.5 ms for one regime. Check `docker ps` first - load average
# lags and reads low while a runner is still busy.

IMG="${IMAGE:-vllm_artemis:cpu}"
run_one() {
  docker run --rm --entrypoint python3 \
    --user "$(id -u):$(id -g)" \
    -e VLLM_CPU_SGL_KERNEL=1 -e OMP_NUM_THREADS=16 \
    -e MICROBENCH_REPS="${MICROBENCH_REPS:-20}" \
    --cpuset-cpus 0-15 \
    -v "$(pwd)/artemis/$1:/mb.py:ro" \
    -v "$(pwd):/out" -w /out \
    "$IMG" /mb.py
}

echo "=== MoE + dense (64.12% of step CPU) ==="
OUT_MOE="$(run_one microbench_moe_dense.py)"
echo "$OUT_MOE"

echo
echo "=== attention + GDN (24.15% of step CPU) ==="
OUT_ATTN="$(run_one microbench_attn_gdn.py)"
echo "$OUT_ATTN"

python3 - "$OUT_MOE" "$OUT_ATTN" <<'PY'
import json, re, sys
tot = 0.0
for blob in sys.argv[1:]:
    m = re.search(r"cpu_share_pct\s+([0-9.]+)", blob)
    if not m:
        raise SystemExit("FAILURE: no cpu_share_pct in a microbenchmark's output")
    tot += float(m.group(1))
stock = 88.27
print(f"\n{'='*58}")
print(f"cpu_share_pct {tot:.3f}   (LOWER IS BETTER; stock = {stock:.2f})")
print(f"  => predicted end-to-end saving {stock - tot:+.2f}% of step CPU")
if abs(stock - tot) < 3.0:
    print("  NOTE: under the ~3% end-to-end noise floor - would NOT be")
    print("        provable on the production trace even if real.")
print("="*58)
json.dump({"cpu_share_pct": tot}, open("artemis_results.json", "w"), indent=2)
PY
