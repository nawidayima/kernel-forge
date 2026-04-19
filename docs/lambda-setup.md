# Running & Profiling CUDA Kernels on Lambda Labs (from macOS / M4 Max)

> Target reader: you edit CUDA on a Mac, build/run/profile on a rented NVIDIA box.
> Scope: end-to-end loop for `make run` / `make profile` / `make bench` against an A100 (sm_80) or H100 (sm_90).

---

## WARNING — Read this before picking Lambda On-Demand (verified 2026-04)

The premise that **Lambda On-Demand instances allow Nsight Compute performance counters** is **no longer accurate** (and, per the 2025 Lambda staff replies below, may never have been for the current virtualization stack):

- On Lambda's on-demand (virtualized) instances, `ncu --set full` and friends fail with `ERR_NVGPUCTRPERM`. Lambda staff confirmed on DeepTalk in **Sep-2025** that "nsight compute is not currently supported" because "NVIDIA GPU performance counters aren't allowed in the virtual machines that Lambda is using" (Nvidia Profiling support thread, [deeptalk.lambda.ai/t/nvidia-profiling-support/4763](https://deeptalk.lambda.ai/t/nvidia-profiling-support/4763)).
- The usual `NVreg_RestrictProfilingToAdminUsers=0` modprobe fix **does not work** on Lambda On-Demand because the restriction is at the hypervisor, not the guest kernel module.
- This state was still current when verified in April 2026.

**What this means for a learner who *needs* `ncu`:**

| Option | Works for `ncu`? | $/hr (2026) | Notes |
|---|---|---|---|
| Lambda **On-Demand** 1xH100 / 1xA100-PCIe | **No** — hypervisor blocks counters | H100 PCIe $3.29, A100 40GB PCIe $1.99/GPU | Use only for `make run` / `make bench` |
| Lambda **1-Click Clusters / Private Cloud** (bare metal H100) | Yes (bare metal, no hypervisor) | From ~$3.29–$4.29/GPU-hr, usually monthly commit | Overkill & expensive for short learning sessions |
| **RunPod Secure Cloud** (H100/A100 containers) | **No** — containers unprivileged, `SYS_ADMIN` not granted | ~$1.99–$2.99/hr for H100 | Use only for `make run` / `make bench` |
| **RunPod Community Cloud** | **No** — same container model | Cheaper, same block | Use only for `make run` / `make bench` |
| **Vast.ai** (any listing) | **Per-host variance** — depends entirely on whether the operator set the modprobe flag | $1.50–$2.50/hr typical for H100 | Must test each host with the 60-second check below |
| **Hetzner / OVH / Vultr bare-metal GPU rentals** | Yes — you are root on host | Varies; hourly or monthly | Most reliable rent-for-profiling option |
| **University cluster** | Usually yes (already configured) | Free if you have access | The quiet answer most published CUDA journals rely on |
| **Personal workstation with NVIDIA GPU** | Yes | One-time hardware cost | A used 3090 is the cheapest route to unlimited profiling |

**Bottom line:** no mainstream virtualized/containerized cloud provider (Lambda, RunPod, most Colab-style services) grants `ncu` counter access in 2026. The reason is architectural, not a bug: counter access requires either host-kernel modprobe (not reachable from a guest VM) or `--cap-add=SYS_ADMIN` on the container (a security hole no shared-tenancy provider will open).

**Recommendation:** Split the workflow. Do `make run` / `make bench` on any cheap single-GPU On-Demand instance (Lambda H100 PCIe is fine). For `make profile`, either (a) use a university cluster if you have one, (b) buy/borrow an NVIDIA workstation, or (c) rent bare-metal from a provider where you're root on the host. Do **not** plan to profile on Lambda On-Demand or RunPod — the counters are unreachable regardless of what you try inside the guest.

**The 60-second counter check** to run on any new box before you trust it:
```bash
# remote
ncu --section SpeedOfLight /bin/true
# if this errors with ERR_NVGPUCTRPERM, profiling will not work here. Terminate.
```

Sources:
- Lambda DeepTalk "Nvidia Profiling support" ([link](https://deeptalk.lambda.ai/t/nvidia-profiling-support/4763)) — Lambda staff confirming no counter access on On-Demand.
- RunPod Answer Overflow "Profiling CUDA kernels in runpod" ([link](https://www.answeroverflow.com/m/1225015304682471454)) — RunPod confirming containers are unprivileged and `SYS_ADMIN` is not granted.
- RunPod Answer Overflow "Enable performance counter on runpod" ([link](https://www.answeroverflow.com/m/1317999276684476509)) — follow-up confirming same limitation.
- NVIDIA ERR_NVGPUCTRPERM KB ([link](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)) — the underlying permission model.

---

## 1. Account + Instance Selection

### Sign up
Create a Lambda account at `https://cloud.lambda.ai`, add a payment method, and — important — **add an SSH public key under "SSH Keys"** before launching anything. Lambda bakes the key into `~/ubuntu`'s `authorized_keys` at boot; there is no console password login.

### Pick an instance (verified 2026-04 at [lambda.ai/pricing](https://lambda.ai/pricing))

For this curriculum (one GPU at a time, sm_80 or sm_90, `ncu` required):

- **If you want to profile on Lambda:** none of the single-GPU On-Demand SKUs let you run `ncu`. You'd need a 1-Click Cluster / Private Cloud (bare metal). See alt-provider table above.
- **If you only need `make run` / `make bench` on Lambda (recommended Lambda usage):**
  - **1x H100 PCIe (80 GB)** — $3.29/GPU-hr. Best default. sm_90, matches the H100 section of your repo. Often available.
  - **1x H100 SXM (80 GB)** — $4.29/GPU-hr. Slightly faster, rarely worth the delta for single-kernel micro-benchmarks.
  - **A100 40 GB PCIe** — $1.99/GPU-hr, but only in **2x and 4x** bundles as of 2026. There is **no 1x A100 On-Demand** anymore; A100 SXM (80 GB) is **8x-only** at $3.99/GPU-hr (so $31.92/hr minimum). If you're on a budget and want sm_80, pick RunPod or Vast.ai for single-A100.

### On-Demand vs Reserved
For a learning loop, **always On-Demand**. Reserved/1-Click requires 1-week to monthly commits and is cheaper only at high utilization. On-Demand is billed by the second with no minimum (Lambda pricing page).

### Persistent storage
- Lambda filesystems mount at `/lambda/nfs/<filesystem-name>` (Lambda filesystem docs, [docs.lambda.ai/public-cloud/filesystems/](https://docs.lambda.ai/public-cloud/filesystems/)).
- **Must be attached at launch**; you cannot add one to a running instance.
- **Anything outside the filesystem is destroyed on terminate**, including your home directory, installed packages, and build artifacts. Per Lambda docs: *"Data not stored in [persistent storage] is erased once you terminate your instance and cannot be recovered."* ([recover-data-terminated-instance](https://beta-docs.lambdalabs.com/cloud/recover-data-terminated-instance/))
- Cost: a few cents per GB-month; check current rate at Lambda billing page. The filesystem bills even while no instance is attached.
- Put the repo, `benchmark_results/`, and any datasets on `/lambda/nfs/<fs-name>/kernel-forge`. Symlink from `~` for convenience.

---

## 2. Nsight Compute Counter Access — Exact State

**Expect failure on On-Demand.** If you run `make profile` on a stock Lambda On-Demand instance you will see:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
==ERROR== For instructions on enabling permissions and to get more information
see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

On a normal Linux host with driver-level access, the fix is:

```bash
# remote (sudo) — works on RunPod Secure Cloud, Vast.ai bare-metal, Lambda bare-metal
sudo tee /etc/modprobe.d/nvidia-profiler.conf <<< 'options nvidia NVreg_RestrictProfilingToAdminUsers=0'
sudo update-initramfs -u -k all
sudo reboot
```

(Steps from [NVIDIA ERR_NVGPUCTRPERM KB](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).)

**On Lambda On-Demand this will not help** — the counters are blocked at the hypervisor. Confirm before you spend time: if you see the error above on Lambda On-Demand, stop and move the profiling session to RunPod/Vast.ai. Do not burn hours on the modprobe fix; it's been tried and it doesn't reach the hypervisor layer (multiple NVIDIA forum threads, e.g. [ERR_NVGPUCTRPERM despite following instructions](https://forums.developer.nvidia.com/t/err-nvgpuctrperm-despite-following-instructions/176670)).

**Quick self-check on any new box** before you trust the loop:

```bash
# remote
ncu --query-metrics | head -5     # works anywhere ncu is installed
ncu --section SpeedOfLight /bin/true || echo "counters blocked"
```

---

## 3. SSH Setup from macOS

```bash
# local (macOS)
test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -C "amiya@arbiter.ai"
pbcopy < ~/.ssh/id_ed25519.pub
# Paste into Lambda Cloud → SSH Keys → Add SSH key
```

Add to `~/.ssh/config`:

```sshconfig
# local (macOS) — ~/.ssh/config
Host lambda
    HostName <paste-public-ip-from-lambda-dashboard>
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 10
    TCPKeepAlive yes
    ForwardAgent no
```

Rotate `HostName` when you re-launch (the IP changes).

### Session survival: use tmux, not mosh

On Lambda, **tmux is the right answer** because (a) it's preinstalled on Ubuntu 22.04 + Lambda Stack, (b) Lambda's firewall exposes port 22 but not the UDP range mosh needs (60000–61000), and you can't reconfigure the security group on On-Demand. Long-running `make bench` should always be inside tmux:

```bash
# remote
tmux new -s forge            # start
# ... work ...
# Ctrl-b d                   # detach, leave running
tmux attach -t forge         # re-attach after reconnect
```

---

## 4. Getting Code Remote — Pick One

Three realistic options:

**(a) Git push + pull.** Clean, but forces a commit on every edit, pollutes history with WIP, and makes the remote clone drift if you forget to pull. Good only if you're disciplined about squashing.

**(b) `rsync -avz --delete`.** Ugly but precise. One-shot, one-way, mirrors your working tree including uncommitted edits. Helper script:

```bash
# local (macOS) — save as scripts/sync-to-lambda.sh
#!/usr/bin/env bash
set -euo pipefail
rsync -avz --delete \
  --exclude '.git' --exclude 'build/' --exclude 'benchmark_results/' \
  ~/Desktop/Karpathy/kernel-forge/ lambda:/lambda/nfs/forge/kernel-forge/
```

**(c) VS Code Remote-SSH (or Cursor Remote-SSH).** Edits in the IDE happen directly on the remote FS over SSH. Zero sync step.

**Recommendation: (c) VS Code Remote-SSH as your daily driver, with (b) as a fallback** for scripted batch reruns. Reasons:
- Edit latency on an M4 Max over SSH is imperceptible.
- The IDE's terminal pane becomes your build/run shell — no context switch.
- `make profile` output lands in `benchmark_results/` on the remote, where it belongs; pull `.ncu-rep` files back with a one-liner (Section 6).
- Unison is fine but one more moving part to install on both ends.

Install: VS Code → Extensions → "Remote - SSH" → Command Palette → `Remote-SSH: Connect to Host` → `lambda`.

---

## 5. First-Run Checklist (on a fresh Lambda box)

```bash
# remote — sanity
nvidia-smi                              # driver + GPU visible, no Xid errors
nvcc --version                          # expect CUDA 12.x
ncu --version                           # Nsight Compute CLI present
nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # expect 8.0 (A100) or 9.0 (H100)
```

Your repo pins `CMAKE_CUDA_ARCHITECTURES=80` by default. If you landed on an H100 (cc 9.0), rebuild with `90`:

```bash
# remote
cd /lambda/nfs/forge/kernel-forge
CUDAARCHS=90 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
make run EX=matmul K=1
```

If `make run` prints kernel output and a timing, you're good.

---

## 6. The Profiling Loop (end-to-end)

Assuming you edit locally, sync via Remote-SSH, and profile on a **counter-enabled** host (university cluster, personal workstation, bare-metal rental, or a Vast.ai host you've verified with the 60-second check in Section 2 — **not** Lambda On-Demand and **not** RunPod):

```bash
# 1. remote — run the profile
cd /workspace/kernel-forge   # or /lambda/nfs/forge/kernel-forge
make profile EX=matmul K=1
# produces benchmark_results/matmul_k1.ncu-rep
```

```bash
# 2. local (macOS) — pull reports back
rsync -avz lambda:/lambda/nfs/forge/kernel-forge/benchmark_results/*.ncu-rep \
  ~/Desktop/Karpathy/kernel-forge/benchmark_results/
# or for RunPod:
# rsync -avz runpod:/workspace/kernel-forge/benchmark_results/*.ncu-rep ./benchmark_results/
```

```bash
# 3. local (macOS) — open in Nsight Compute GUI
open -a "NVIDIA Nsight Compute" ~/Desktop/Karpathy/kernel-forge/benchmark_results/matmul_k1.ncu-rep
```

The **macOS Nsight Compute GUI** is a free separate download from NVIDIA. The installer is universal (Apple Silicon native since 2023.3). Get it from [developer.nvidia.com/nsight-compute](https://developer.nvidia.com/nsight-compute) — you'll need a (free) NVIDIA developer account. The **CLI `ncu` is NOT available on macOS** — that's why we profile remote.

---

## 7. Cost Discipline

On Lambda:
- **Stop** doesn't exist for On-Demand — only **Terminate**. Terminate **wipes the ephemeral disk**. Put anything you care about in the filesystem at `/lambda/nfs/...` *before* terminating. ([recover-data-terminated-instance](https://beta-docs.lambdalabs.com/cloud/recover-data-terminated-instance/))
- Billing stops the second the instance enters terminated state.
- The filesystem itself continues to bill per-GB-month even with no instance attached — deprovision it too if you're done for more than a few days.

Add this to `~/.zshrc` so you see idle instances every shell:

```bash
# local (macOS) — ~/.zshrc
alias forge-check='curl -s -u $LAMBDA_API_KEY: https://cloud.lambda.ai/api/v1/instances | jq -r ".data[] | select(.status==\"active\") | \"RUNNING: \(.instance_type.name) — \(.instance_type.price_cents_per_hour/100)/hr\""'
[[ -n "$(forge-check 2>/dev/null)" ]] && echo "Lambda instance still running — forge-check for details."
```

(Generate an API key under Lambda Cloud → API keys; export `LAMBDA_API_KEY` in `~/.zprofile`.)

---

## 8. Common Gotchas

1. **Clock throttling distorts benchmarks.** A100/H100 will DVFS down under thermal or power pressure; two runs of the same kernel differ by ~5–15%. Lock clocks before benching:
   ```bash
   # remote (needs sudo — works on bare metal, not Lambda On-Demand)
   sudo nvidia-smi -pm 1
   sudo nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)
   # to undo:
   sudo nvidia-smi -rgc
   ```
   On Lambda On-Demand you can't sudo-lock clocks, so take N=5 runs and report the median for your `make bench` sweeps.

2. **`ncu --set full` is glacial.** `--set full` replays each kernel dozens of times to cover every metric — a matmul that runs in 200 µs can take 30+ seconds to profile. For quick iteration, use a targeted section:
   ```bash
   # remote
   ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
       -o benchmark_results/matmul_k1 ./build/matmul 1
   ```
   Switch to `--set full` only for the final "capture everything" pass. (Nsight Compute Profiling Guide, [docs.nvidia.com/nsight-compute/ProfilingGuide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html))

3. **Persistent storage gotchas.** (a) Must be attached at launch — cannot be added to a running instance. (b) `apt install` and pip installs go to `/` not to the filesystem, so they die on terminate. Keep a `scripts/bootstrap.sh` that reinstalls your tooling; run it on every new box.

4. **A100 On-Demand availability churn.** Lambda's 1x A100 SKU was retired; A100 SXM is 8x-only and A100 PCIe is 2x/4x only as of 2026. If your repo's "default = sm_80" assumption matters, either pay for a 2x A100 PCIe (~$4/hr) or use a single-H100 and set `CUDAARCHS=90`. H100 availability is better but also spiky — the Lambda dashboard shows region-by-region stock; try `us-west-3` / `us-east-1` first.

5. **Counter check on every new host.** Before you invest time, run the Section 2 two-liner (`ncu --section SpeedOfLight /bin/true`). If it errors with `ERR_NVGPUCTRPERM`, you're on a VM whose hypervisor blocks counters — move the profile session to a different provider/host rather than debugging the guest.

---

### Sources (all verified 2026-04)

- Lambda pricing page: [lambda.ai/pricing](https://lambda.ai/pricing)
- Lambda filesystem docs: [docs.lambda.ai/public-cloud/filesystems](https://docs.lambda.ai/public-cloud/filesystems/)
- Lambda: data recovery on terminate: [beta-docs.lambdalabs.com/cloud/recover-data-terminated-instance](https://beta-docs.lambdalabs.com/cloud/recover-data-terminated-instance/)
- Lambda DeepTalk — Nvidia profiling support (staff reply, Sep 2025): [deeptalk.lambda.ai/t/nvidia-profiling-support/4763](https://deeptalk.lambda.ai/t/nvidia-profiling-support/4763)
- Lambda DeepTalk — Nsight permissions issue: [deeptalk.lambda.ai/t/running-nvidia-nsight-permissions-issue/4433](https://deeptalk.lambda.ai/t/running-nvidia-nsight-permissions-issue/4433)
- NVIDIA ERR_NVGPUCTRPERM KB: [developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
- NVIDIA forum — fixes failing for non-admin: [forums.developer.nvidia.com/t/err-nvgpuctrperm-despite-following-instructions/176670](https://forums.developer.nvidia.com/t/err-nvgpuctrperm-despite-following-instructions/176670)
- Nsight Compute Profiling Guide 13.2 / 2026.1: [docs.nvidia.com/nsight-compute/ProfilingGuide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- Nsight Compute download (macOS GUI): [developer.nvidia.com/nsight-compute](https://developer.nvidia.com/nsight-compute)
