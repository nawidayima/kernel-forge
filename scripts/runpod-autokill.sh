#!/usr/bin/env bash
# Auto-terminate the current RunPod pod after N hours.
#
# Run once per SSH session from inside a pod:
#   bash scripts/runpod-autokill.sh 4
#
# Default: 4 hours. The watchdog survives SSH disconnect (nohup + disown).
# Won't double-arm if already running.
#
# Relies on:
#   - $RUNPOD_POD_ID (auto-set in every RunPod pod)
#   - runpodctl (pre-installed with pod-scoped API key)

set -euo pipefail

HOURS="${1:-4}"
POD_ID="${RUNPOD_POD_ID:?RUNPOD_POD_ID not set — this script must run inside a RunPod pod}"
LOCK=/tmp/runpod-autokill.pid
LOG=/tmp/runpod-autokill.log

if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
    echo "[autokill] already armed (pid $(cat "$LOCK")). tail -f $LOG"
    exit 0
fi

nohup bash -c "
echo \"[\$(date -Is)] autokill armed: pod=$POD_ID lifetime=${HOURS}h\" >> '$LOG'
sleep $((HOURS * 3600))
echo \"[\$(date -Is)] lifetime elapsed; terminating $POD_ID\" >> '$LOG'
runpodctl remove pod '$POD_ID' >> '$LOG' 2>&1
rm -f '$LOCK'
" > /dev/null 2>&1 &

echo "$!" > "$LOCK"
disown

echo "[autokill] armed for ${HOURS}h (pid $!). Monitor: tail -f $LOG"
echo "[autokill] to cancel: kill \$(cat $LOCK) && rm -f $LOCK"
