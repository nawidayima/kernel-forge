#!/usr/bin/env bash
# Sweep (kernel x size) for one exercise, capture RESULT lines to CSV.
# Output filename is tagged with GPU model, git short SHA, and UTC date,
# so prior runs are never overwritten and rows can be matched back to a
# specific kernel state.
#
# Usage:
#   EX=matmul ./scripts/benchmark.sh
#   EX=matmul KERNELS="1 2" SIZES="1024 2048 4096" ./scripts/benchmark.sh
#
# Outputs:
#   benchmark_results/<EX>_<GPU>_<SHA>_<DATE>.log
#   benchmark_results/<EX>_<GPU>_<SHA>_<DATE>.csv
#
# If the working tree is dirty, SHA is suffixed with -dirty so dirty runs
# are obviously labeled and not committed as if they reflect HEAD.

set -euo pipefail

EX="${EX:?set EX=<exercise> e.g. matmul}"
OUT="${OUT:-benchmark_results}"
BUILD="${BUILD:-build}"
KERNELS="${KERNELS:-1 2}"
SIZES="${SIZES:-128 256 512 1024 2048 4096}"

BIN="$BUILD/$EX"
if [[ ! -x "$BIN" ]]; then
    echo "binary $BIN not found, run cmake build first" >&2
    exit 1
fi

# --- run tag ----------------------------------------------------------
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
        | head -1 | sed 's/NVIDIA //; s/ /_/g; s/[()]//g')"
GPU="${GPU:-unknownGPU}"

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    SHA="${SHA}-dirty"
fi

DATE="$(date -u +%Y-%m-%d)"
TAG="${EX}_${GPU}_${SHA}_${DATE}"

mkdir -p "$OUT"
LOG="$OUT/${TAG}.log"
CSV="$OUT/${TAG}.csv"
: > "$LOG"

echo "[benchmark] EX=$EX GPU=$GPU SHA=$SHA DATE=$DATE"
echo "[benchmark] writing $LOG and $CSV"

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
echo "[benchmark] wrote $CSV"
