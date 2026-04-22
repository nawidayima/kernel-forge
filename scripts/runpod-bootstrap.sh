#!/usr/bin/env bash
# One-shot pod setup. Source (don't execute) once per SSH session:
#   source /workspace/kernel-forge/scripts/runpod-bootstrap.sh
#
# Does two things:
#   1. Ensures cmake is on PATH — installed into a venv on the network
#      volume so it survives pod termination (unlike `pip install cmake`
#      to the pod's ephemeral /usr/local).
#   2. Arms the 4-hour autokill watchdog.
#
# Idempotent: re-running is safe and cheap.

set -u

# --- 1. cmake (persistent via venv on /workspace) ------------------------
if [[ ! -x /workspace/venv/bin/cmake ]]; then
    echo "[bootstrap] creating /workspace/venv and installing cmake (one-time, ~30s)"
    rm -rf /workspace/venv
    python3 -m venv /workspace/venv
    /workspace/venv/bin/pip install --quiet --upgrade pip
    /workspace/venv/bin/pip install --quiet cmake
fi
export PATH="/workspace/venv/bin:${PATH}"

# --- 2. Autokill watchdog -----------------------------------------------
bash /workspace/kernel-forge/scripts/runpod-autokill.sh 4

# --- Summary ------------------------------------------------------------
echo "[bootstrap] ready"
echo "  cmake : $(command -v cmake) ($(cmake --version | head -1))"
echo "  nvcc  : $(command -v nvcc)"
echo "  pod   : ${RUNPOD_POD_ID:-unknown}"
