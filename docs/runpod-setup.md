# RunPod Setup for kernel-forge (from macOS / M4 Max)

> **Goal:** shortest path from signup to first `make run EX=matmul K=1`.
> **Scope:** `make run` and `make bench` only. RunPod does **not** allow `ncu` profiling — see `docs/lambda-setup.md`. Profiling decision is deferred.

## Hardware plan (per the project's own spec)

| Curriculum phase | Exercises | GPU | sm | ~$/hr |
|---|---|---|---|---|
| Phases 1–4 | matmul (naive, tiled), reduction, cross_entropy | **L4 24GB** | 89 | ~$0.45 |
| Phase 5+ | flash_attention, quantized_gemm, moe_dispatch | **H100 PCIe 80GB** | 90 | ~$2.50 |

**Start on L4.** Switch to H100 when you reach flash_attention (phase 5) — the later exercises exist *because* of Hopper tensor cores (`wgmma`, TMA, FP8), and running them on L4 misses the point. The GPU switch is cheap: terminate the L4 pod, keep the network volume, attach it to a new H100 pod in the same datacenter.

See README §"Compute capability" and `docs/specs/2026-04-19-kernel-forge-design.md` §"Compute capability reference" for the authoritative table.

---

## 1. Account + SSH key

```bash
# local (macOS)
test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -C "amiya@arbiter.ai"
pbcopy < ~/.ssh/id_ed25519.pub
```

Sign up at [runpod.io](https://runpod.io), add a payment method, then **Settings → SSH Public Keys → paste and save**.

### Cost enforcement — set this up before launching your first pod

**Disable auto-pay + pre-fund a budget.** Settings → Billing → turn **Auto-Pay OFF**. Top up once with your monthly budget (e.g., $30). RunPod will auto-stop pods when balance can't cover 10 more minutes of runtime — so your deposit is a hard cap. Their card cannot be charged beyond it.

**Gotcha:** if your balance stays at $0 for >48 hours, network volumes may be deleted and are unrecoverable. Top up again before that window closes, or back up anything you care about to GitHub.

**Spending limit (hourly rate cap).** Default is $80/hr across all resources. For a learner running one L4 at $0.45/hr this is not a meaningful brake, but it's worth knowing about — it catches catastrophic misconfigurations.

## 2. Create a network volume (before launching a pod)

Pod disk is ephemeral — **terminate wipes it**. Network volumes persist across pods in the same datacenter and attach at `/workspace`.

- Dashboard → **Storage → Network Volumes → New Volume**.
- Size: **50 GB** is enough (repo + build artifacts + a few `.ncu-rep` files later). Cost ~$5/month.
- Pick a **Secure Cloud** datacenter with H100 availability (typically US-CA or US-TX). Write down the region — you must launch pods in the same one.

## 3. Launch the pod

Dashboard → **Pods → Deploy**:

- **GPU:** **L4 24GB, Secure Cloud** (for phases 1-4). Rate ~$0.43–0.49/hr. Switch to H100 PCIe 80GB (~$2.49–2.89/hr) when you reach flash_attention.
- **Template:** pick a **CUDA devel** image. As of 2026 the reliable pick is "RunPod PyTorch 2.x (CUDA 12.x devel)" or the official NVIDIA `nvidia/cuda:12.x.x-devel-ubuntu22.04`. **Must be `-devel`, not `-runtime`** — runtime images don't ship `nvcc`.
- **Volume:** attach the network volume from step 2, mount at `/workspace`.
- **Deploy.** Pod boots in ~30 seconds.

## 4. Connect

On the pod card, copy the **"SSH over exposed TCP"** connection string (looks like `ssh root@1.2.3.4 -p 54321`). Add to `~/.ssh/config`:

```sshconfig
# local (macOS) — ~/.ssh/config
Host runpod
    HostName 1.2.3.4
    Port 54321
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    TCPKeepAlive yes
```

Rotate `HostName` and `Port` every time you launch a new pod — they change.

```bash
# local
ssh runpod
```

## 5. Sanity checks on the pod

```bash
# remote
nvidia-smi                                                      # GPU visible, no Xid
nvcc --version                                                  # CUDA 12.x
nvidia-smi --query-gpu=compute_cap --format=csv,noheader        # expect 8.9 (L4) or 9.0 (H100)
df -h /workspace                                                # your 50GB network volume
apt list --installed 2>/dev/null | grep -E "cmake|git"          # check; apt-get install if missing
```

If `cmake` is missing (it will be on the stock CUDA-devel image — `apt-get install cmake` fails with "Unable to locate package" because the image ships a stripped apt sources list), don't try to fix apt. Use the bootstrap script in section 5a — it installs cmake into a venv on the **network volume**, so it survives pod termination.

## 6. Get the code to the pod

Since the repo lives on your Mac and likely isn't pushed anywhere yet, tar-pipe from local. **Don't use rsync** — it isn't installed on the stock CUDA-devel image and can't be apt-installed (see section 5).

```bash
# local (macOS) — initial sync and every re-sync
cd ~/Desktop/Karpathy/kernel-forge
tar czf - --exclude='.git' --exclude='build*' --exclude='benchmark_results' --exclude='archive' . \
  | ssh runpod 'mkdir -p /workspace/kernel-forge && tar xzf - --no-same-owner -C /workspace/kernel-forge'
```

The `--no-same-owner` flag suppresses harmless ownership errors (macOS uid 501 → root on pod). `LIBARCHIVE.xattr` warnings in the output are macOS extended attributes — ignore.

**Iterative edits:** the highest-leverage workflow is VS Code Remote-SSH (section 8) — edits save directly on the pod, no re-sync needed. If you're editing on the Mac instead, rerun the tar-pipe above; it's idempotent.

(If/when you push the repo to GitHub, swap to `git clone` on the pod.)

## 6a. Bootstrap the pod (every new pod, one command)

The stock CUDA-devel image doesn't ship `cmake`, and `pip install cmake` to the pod's local filesystem gets wiped when the pod terminates. `scripts/runpod-bootstrap.sh` fixes this by installing cmake into `/workspace/venv/` — which lives on the network volume and persists.

```bash
# remote — first command after every new SSH session
source /workspace/kernel-forge/scripts/runpod-bootstrap.sh
```

What it does:
- Creates `/workspace/venv/` and `pip install cmake` into it (only on first run — ~30s one-time, ~0s every subsequent run).
- Prepends `/workspace/venv/bin` to `PATH` for this shell.
- Arms the 4-hour autokill watchdog.

If you want it automatic: since `~/.bashrc` is ephemeral, add it to the autoexec on every new pod:
```bash
# remote — once per new pod
echo 'source /workspace/kernel-forge/scripts/runpod-bootstrap.sh' >> ~/.bashrc
```

## 7. Build and run

The repo's `CMakeLists.txt` defaults to sm_80. Override once per GPU:

```bash
# remote — L4 (sm_89), phases 1-4
cd /workspace/kernel-forge
CUDAARCHS=89 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/matmul 1
```

```bash
# remote — H100 (sm_90), phase 5+
# Use a separate build dir so the CMake cache doesn't collide with the L4 build:
CUDAARCHS=90 cmake -S . -B build-h100 -DCMAKE_BUILD_TYPE=Release
cmake --build build-h100 -j
./build-h100/flash_attention 1
```

You should see a `RESULT exercise=matmul kernel=1 M=4096 K=4096 N=4096 ms=... gflops=...` line.

Once that works, `make run` and `make bench` work normally (the CMake cache in `build/` holds your current `CUDAARCHS`).

## 8. Daily workflow: VS Code (or Cursor) Remote-SSH

This is the highest-leverage quality-of-life setup:

1. Install the **Remote - SSH** extension.
2. Command Palette → `Remote-SSH: Connect to Host` → `runpod`.
3. Open folder `/workspace/kernel-forge`.
4. Edits save directly on the pod — **no rsync step after the initial sync**. Terminal pane inside the editor runs builds and kernels.

## 9. Cost enforcement (not just discipline)

Three layers, ordered from hardest to softest:

### Layer 1: Pre-funded budget, auto-pay disabled
Already set up in Section 1. Your deposited balance is the absolute cap — RunPod cannot charge your card beyond it. When it runs out, pods auto-stop within 10 minutes. Top up again to resume.

### Layer 2: Per-pod auto-terminate watchdog
Pods with network volumes attached can only be terminated, not stopped — so you either terminate or keep paying. This script kills the pod automatically after N hours regardless of what you're doing. Run it once per SSH session:

```bash
# remote — arms a 4-hour hard lifetime
bash scripts/runpod-autokill.sh 4
```

The watchdog is detached from your shell and survives SSH disconnect. It calls `runpodctl remove pod $RUNPOD_POD_ID` when the timer fires. Cancel with `kill $(cat /tmp/runpod-autokill.pid)` if you're actively working past the limit.

**Make it automatic:** append this to `/root/.bash_profile` on the pod so it arms on every SSH login (idempotent — won't double-arm):

```bash
# remote — on each new pod, edit /root/.bash_profile once
echo 'bash /workspace/kernel-forge/scripts/runpod-autokill.sh 4 >/dev/null 2>&1 || true' >> ~/.bash_profile
```

### Layer 3 (optional): Local-side RunPod API check
If you want belt-and-suspenders, a cron on your Mac can terminate any running pods at a fixed hour:

```bash
# local (macOS) — install runpodctl
brew install runpod/runpodctl/runpodctl
runpodctl config --apiKey rpa_...   # from Settings → API Keys

# nightly termination cron at 23:30 (crontab -e)
30 23 * * * /opt/homebrew/bin/runpodctl stop pod $(/opt/homebrew/bin/runpodctl get pod | awk '/RUNNING/ {print $1}') 2>>/tmp/runpod-nightly.log
```

### Billing reference table

| Pod state | Billing | What persists |
|---|---|---|
| Running | Full GPU-hr (~$0.45/hr L4, ~$2.50/hr H100) | Everything |
| Stopped (no network volume) | Disk only (~$0.10–0.20/hr) | Container disk + config |
| Terminated | $0 | **Only** the network volume; pod's own `/root` is gone |

**Important**: with a network volume attached (our setup), RunPod will only offer **Terminate**, not Stop. That's fine — the volume preserves your work.

**When switching L4 → H100** at phase 5: terminate the L4 pod, deploy a new H100 in the same datacenter, attach the same volume. Code and build dirs survive unchanged.

## 10. When you hit the `ncu` wall

Expected ~2-4 weeks in, at the register-blocked matmul or vectorized-load variant, when wall-clock timing stops being enough to diagnose bottlenecks. See `docs/lambda-setup.md` section 2 for the full landscape. Realistic options at that point:

- **Hetzner GEX44 (RTX 4090 dedicated, bare metal)** — ~€186/mo, root on host, `NVreg_RestrictProfilingToAdminUsers=0` works.
- **Used RTX 3090 workstation** — ~$600–700 one-time, unlimited profiling; requires a separate Linux box on your network (M4 Max can't host a CUDA GPU).
- **Vast.ai verified-datacenter listing** — cheaper, per-host variance, run `ncu --section SpeedOfLight /bin/true` in the first 60 seconds and terminate if it errors.

That decision can wait. Get the kernel running first.
