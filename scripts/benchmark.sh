#!/usr/bin/env bash
# Sweep (kernel x size) for one exercise, capture RESULT lines to CSV.
#
# Usage:
#   EX=matmul ./scripts/benchmark.sh
#   EX=matmul KERNELS="0 1 2" SIZES="128 256 512 1024 2048 4096" ./scripts/benchmark.sh
#
# Outputs:
#   benchmark_results/<EX>.log   (full stdout for debugging)
#   benchmark_results/<EX>.csv   (one row per RESULT line)

set -euo pipefail

EX="${EX:?set EX=<exercise> e.g. matmul}"
OUT="${OUT:-benchmark_results}"
BUILD="${BUILD:-build}"
KERNELS="${KERNELS:-0 1 2}"
SIZES="${SIZES:-128 256 512 1024 2048 4096}"

BIN="$BUILD/$EX"
if [[ ! -x "$BIN" ]]; then
    echo "binary $BIN not found — run 'make build' first" >&2
    exit 1
fi

mkdir -p "$OUT"
LOG="$OUT/${EX}.log"
CSV="$OUT/${EX}.csv"
: > "$LOG"

for k in $KERNELS; do
    for s in $SIZES; do
        {
            echo "# EX=$EX kernel=$k size=$s"
            "$BIN" "$k" "$s" "$s" "$s" || echo "(run failed)"
            echo
        } | tee -a "$LOG"
    done
done

python3 "$(dirname "$0")/parse_results.py" "$LOG" "$CSV"
echo "wrote $CSV"
