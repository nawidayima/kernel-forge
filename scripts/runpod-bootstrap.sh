#!/usr/bin/env bash
# One-shot pod setup. Source (don't execute) once per SSH session:
#   source /workspace/kernel-forge/scripts/runpod-bootstrap.sh
#
# Does three things:
#   1. Initializes /workspace/kernel-forge as a git repo if it isn't one
#      already, so benchmark.sh can read HEAD's SHA for tagging. Working
#      tree is left untouched; any divergence from upstream main shows
#      up as 'modified' in `git status`.
#   2. Ensures cmake is on PATH, installed into a venv on the network
#      volume so it survives pod termination (unlike `pip install cmake`
#      to the pod's ephemeral /usr/local).
#   3. Arms the 4-hour autokill watchdog.
#
# Idempotent: re-running is safe and cheap.

set -u

# --- 1. Git repo bootstrap ----------------------------------------------
KF_REPO_URL="https://github.com/nawidayima/kernel-forge.git"
KF_DIR="/workspace/kernel-forge"

if [[ -d "$KF_DIR" && ! -d "$KF_DIR/.git" ]]; then
    echo "[bootstrap] $KF_DIR is not a git repo, initializing in-place"
    pushd "$KF_DIR" >/dev/null
    git init -q -b main
    git remote add origin "$KF_REPO_URL"
    git fetch -q origin main
    git reset FETCH_HEAD   # --mixed default: sync HEAD/index to upstream main, leave working tree
    popd >/dev/null
    echo "[bootstrap] git initialized; run 'git -C $KF_DIR status' to see what diverges from main"
fi

# --- 2. cmake (persistent via venv on /workspace) -----------------------
if [[ ! -x /workspace/venv/bin/cmake ]]; then
    echo "[bootstrap] creating /workspace/venv and installing cmake (one-time, ~30s)"
    rm -rf /workspace/venv
    python3 -m venv /workspace/venv
    /workspace/venv/bin/pip install --quiet --upgrade pip
    /workspace/venv/bin/pip install --quiet cmake
fi
export PATH="/workspace/venv/bin:${PATH}"

# --- 3. Autokill watchdog -----------------------------------------------
bash /workspace/kernel-forge/scripts/runpod-autokill.sh 4

# --- Summary ------------------------------------------------------------
echo "[bootstrap] ready"
echo "  cmake : $(command -v cmake) ($(cmake --version | head -1))"
echo "  nvcc  : $(command -v nvcc)"
echo "  git   : $(git -C "$KF_DIR" rev-parse --short HEAD 2>/dev/null || echo 'not a repo')"
echo "  pod   : ${RUNPOD_POD_ID:-unknown}"
