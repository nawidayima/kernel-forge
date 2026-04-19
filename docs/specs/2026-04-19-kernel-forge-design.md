# kernel-forge: Design Spec

**Date:** 2026-04-19
**Author:** Amiya Diwan
**Status:** Draft

## Purpose

kernel-forge is a hands-on CUDA kernel engineering curriculum. Each exercise isolates a transferable GPU programming concept, progressing from naive matmul through FlashAttention and MoE dispatch. The project provides build infrastructure, timing, profiling integration, and correctness verification so the learner can focus on writing kernels.

The research goal behind this curriculum is reducing the compute cost of the research loop for improving model intelligence. RLVF training runs, adaptive-depth architectures like Mixture-of-Recursions, and longer-context experiments are all bottlenecked by kernel throughput. Faster kernels translate directly into more experimental iterations per dollar, which compounds into better models. The target is making the pathway to better models cheaper to explore.

## Scope

### Included

- Build system (CMake + Makefile convenience targets)
- Shared runner infrastructure (timing, verification, GFLOP/s reporting)
- Per-exercise scaffolding with kernel stubs (bodies left as TODOs)
- Cloud GPU setup guide with profiling validation steps
- Revised study guide reflecting a trimmed critical path

### Not included

- Kernel implementations (the learner writes these)
- Cloud GPU provisioning scripts or Terraform
- Training loop integration (nanochat is a separate project)
- Popcorn CLI integration (archived)

## Critical Path

The exercises below form a dependency chain. Each builds on concepts from the previous.

| Order | Exercise | Transferable Principle | Study Guide Phase |
|-------|----------|----------------------|-------------------|
| 1 | matmul (naive) | Memory-bound bottleneck analysis | 2 |
| 2 | matmul (tiled) | Shared memory tiling reduces global memory traffic | 2 |
| 3 | reduction | Tree-based parallelism, warp-level primitives | 3 |
| 4 | cross_entropy (fused) | Kernel fusion eliminates intermediate materialization | 4/6 |
| 5 | flash_attention | Online algorithms replace materialization with bounded memory | 6 |
| 6 | quantized_gemm | Bandwidth optimization through reduced precision | 6 |
| 7 | moe_dispatch | Irregular parallelism, gather-compute-scatter | 6 |

Exercises from the original study guide that are not on this critical path (histogram, conv2d, prefixsum, sort, trimul) remain in the study guide as optional reading. They teach real concepts but are not prerequisites for the research direction.

## Literature Review (April 2026)

A lit review across arxiv, community sources, and tooling docs confirmed the exercise list is aligned with frontier kernel engineering needs. Key findings:

### Attention kernels

FlashAttention-2 remains the right pedagogical target. FA-3 (arXiv:2407.08608) targets Hopper with warp specialization and FP8; FA-4 (arXiv:2603.05451) targets Blackwell with async MMA. Both extend the same algorithmic core (tiling, online softmax, recomputation). Understanding FA-2 is prerequisite to all successors, Ring Attention, and PagedAttention.

### Fused cross-entropy

The naive fusion (softmax + log + nll in one pass) is a starting point, but the frontier is Liger-Kernel's Fused Linear Cross-Entropy (FLCE, arXiv:2410.10989). FLCE chunks the logit projection + softmax + CE + backward into one pass, never materializing the full vocab-sized logit tensor. ~60% memory reduction. torch.compile cannot discover this pattern. The cross_entropy exercise should build toward this.

### Quantized GEMM

INT4 with the Marlin kernel (arXiv:2408.11743) is the production standard for weight-only quantization. FP8 is default for training. NVFP4 on Blackwell is the emerging frontier with native hardware support. DeepSeek's DeepGEMM provides a unified FP8/FP4/BF16 kernel library. For the curriculum, INT4 dequantization fused with GEMM is the right starting target.

### MoE dispatch

MegaBlocks (arXiv:2211.15841) reformulates MoE as block-sparse GEMM. MegaScale-MoE (arXiv:2505.11432) introduces hybrid expert parallelism with warp-group-level dispatch. Main bottlenecks: all-to-all communication, irregular grouped GEMMs with load imbalance, and kernel launch overhead from separate routing/gating/FFN launches. Fused dispatch kernels (FusedXpert, AdaFuse) are the fix.

### RLVF training bottleneck

Rollout generation (autoregressive decode) consumes 60-80% of RLVF training wall time. The bottleneck is memory-bandwidth-bound attention during decode. FlashAttention and quantized GEMM are the highest-leverage exercises for the stated research goal.

### Compiler vs hand-written boundary

torch.compile/Inductor handles pointwise fusions, simple normalization, and matmul epilogues. Hand-written kernels are still required for: FLCE, FlashAttention forward+backward, sub-byte quantization fusions, communication-overlapped kernels, and activation recomputation inside fused kernels. The exercises in this curriculum target the hand-written side of this boundary.

### Tooling landscape

Hand-written CUDA remains the right foundation. Triton is a practical complement for fused ops (Liger-Kernel, DRTriton) but does not replace CUDA for peak attention/GEMM performance. CUTLASS 3/CuTe is the production path for GEMM optimization. ThunderKittens (Stanford) simplifies FlashAttention-style kernels. After completing the CUDA exercises, reimplementing select kernels in Triton is a natural extension.

### References

PDFs are downloaded to `~/Desktop/Karpathy/papers/kernel-forge/`. See `INDEX.md` there for the full catalog.

**Attention kernels:**
- Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) | [PDF](https://arxiv.org/pdf/2307.08691)
- Shah, J. et al. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," 2024. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608) | [PDF](https://arxiv.org/pdf/2407.08608)
- Zadouri, T. et al. "FlashAttention-4: Hardware-Friendly Attention with Producer-Consumer Pipelines," 2026. [arXiv:2603.05451](https://arxiv.org/abs/2603.05451) | [PDF](https://arxiv.org/pdf/2603.05451)

**Kernel fusion:**
- Hsu, P. et al. "Liger Kernel: Efficient Triton Kernels for LLM Training," 2024. [arXiv:2410.10989](https://arxiv.org/abs/2410.10989) | [PDF](https://arxiv.org/pdf/2410.10989)
- "Deep Kernel Fusion for Transformers," 2026. [arXiv:2602.11808](https://arxiv.org/abs/2602.11808) | [PDF](https://arxiv.org/pdf/2602.11808)

**Quantized GEMM:**
- Frantar, E. et al. "Marlin: Mixed-Precision Auto-Regressive Parallel INference on Large Language Models," 2024. [arXiv:2408.11743](https://arxiv.org/abs/2408.11743) | [PDF](https://arxiv.org/pdf/2408.11743)
- "FireQ: Fast INT4-FP8 Fused Kernel for Quantized LLM Inference," 2025. [arXiv:2505.20839](https://arxiv.org/abs/2505.20839) | [PDF](https://arxiv.org/pdf/2505.20839)
- "LiquidGEMM: W4A8 Kernel Pipelining for Quantized GEMM," 2025. [arXiv:2509.01229](https://arxiv.org/abs/2509.01229) | [PDF](https://arxiv.org/pdf/2509.01229)
- Xia, H. et al. "FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design," 2024. [arXiv:2401.14112](https://arxiv.org/abs/2401.14112) | [PDF](https://arxiv.org/pdf/2401.14112)
- "RaZeR: Redundant-Zero Remapping for NVFP4 Quantization," 2025. [arXiv:2501.04052](https://arxiv.org/abs/2501.04052) | [PDF](https://arxiv.org/pdf/2501.04052)

**MoE dispatch:**
- Gale, T. et al. "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts," 2022. [arXiv:2211.15841](https://arxiv.org/abs/2211.15841) | [PDF](https://arxiv.org/pdf/2211.15841)
- "MegaScale-MoE: Hybrid Expert Parallelism with Optimized Communication," 2025. [arXiv:2505.11432](https://arxiv.org/abs/2505.11432) | [PDF](https://arxiv.org/pdf/2505.11432)
- "AdaFuse: Adaptive Fused Switching Kernels for MoE," 2026. [arXiv:2603.11873](https://arxiv.org/abs/2603.11873) | [PDF](https://arxiv.org/pdf/2603.11873)

**Adaptive compute / research context:**
- Bae, S. et al. "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation," 2025. [arXiv:2507.10524](https://arxiv.org/abs/2507.10524) | [PDF](https://arxiv.org/pdf/2507.10524)

## Directory Structure

```
kernel-forge/
  CMakeLists.txt
  Makefile
  README.md
  common/
    runner.cuh            # Timing, verification, reporting
    utils.cuh             # Error macros, random init, alloc helpers
  kernels/
    matmul/
      1_naive.cuh
      2_tiled.cuh
      reference.cuh       # cuBLAS wrapper
    reduction/
      1_divergent.cuh
      2_coalesced.cuh
      reference.cuh
    cross_entropy/
      1_naive.cuh
      2_fused.cuh
      reference.cuh
    flash_attention/
      1_naive_attention.cuh
      2_flash_forward.cuh
      reference.cuh
    quantized_gemm/
      1_int4_dequant.cuh
      reference.cuh
    moe_dispatch/
      1_gather_scatter.cuh
      reference.cuh
  runners/
    matmul.cu
    reduction.cu
    cross_entropy.cu
    flash_attention.cu
    quantized_gemm.cu
    moe_dispatch.cu
  docs/
    study-guide.tex
    specs/
  wiki/                   # Learning knowledge base
  archive/                # Old popcorn repos
    pmpp/
    pmpp_v2/
    princeton/
  benchmark_results/      # ncu exports
```

### Design decisions

**One runner per exercise.** Each runner is a standalone `main()` that selects a kernel variant by CLI argument (`./build/matmul 1`). This keeps exercises independent and avoids shared state.

**Kernels as `.cuh` headers.** Each kernel is a header file included by its runner. No separate compilation units, which keeps the CMake configuration minimal.

**Adding new kernel variants** requires two changes: a new `.cuh` file and a new `case` in the runner's `switch` statement. Adding an entirely new exercise requires a new runner `.cu`, a new `kernels/` subdirectory, and one line in `CMakeLists.txt`.

## Build System

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.19)
project(kernel_forge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 80 CACHE STRING "GPU compute capability")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

set(EXERCISES matmul reduction cross_entropy flash_attention quantized_gemm moe_dispatch)

foreach(exercise ${EXERCISES})
    add_executable(${exercise} runners/${exercise}.cu)
    set_target_properties(${exercise} PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
    target_link_libraries(${exercise} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endforeach()
```

### Makefile

```makefile
BUILD_DIR := build

build:
	@mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build .

run: build
	@./$(BUILD_DIR)/$(EX) $(K)

profile: build
	@ncu --set full --export benchmark_results/$(EX)_kernel_$(K) --force-overwrite ./$(BUILD_DIR)/$(EX) $(K)

clean:
	@rm -rf $(BUILD_DIR)
```

Usage:
- `make build`
- `make run EX=matmul K=1` (run naive matmul)
- `make profile EX=matmul K=1` (profile with ncu)

## Shared Infrastructure

### common/runner.cuh

Provides three functions used by every runner:

- `run_kernel(kernel_fn, ...)`: Warmup (5 iterations), then timed execution (20 iterations) using CUDA events. Returns average milliseconds.
- `verify(float* result, float* reference, int N, float tolerance)`: Compares against reference using relative tolerance. Prints max absolute error and pass/fail. Uses approximate comparison, not exact equality.
- `report_performance(int M, int K, int N, float ms)`: Computes and prints GFLOP/s and percentage of theoretical peak for the target GPU.

### common/utils.cuh

- `CHECK_CUDA(err)`: Error-checking macro that prints file/line on failure.
- `init_random(float* data, int N)`: Fill with uniform random floats.
- `alloc_and_init(int N)`: cudaMalloc + init_random in one call.

## Runner Pattern

Each runner follows a consistent structure:

```cpp
#include "common/runner.cuh"
#include "common/utils.cuh"
#include "kernels/matmul/reference.cuh"
#include "kernels/matmul/1_naive.cuh"
#include "kernels/matmul/2_tiled.cuh"

int main(int argc, char **argv) {
    if (argc < 2) { printf("Usage: %s <kernel_num>\n", argv[0]); return 1; }
    int kernel = atoi(argv[1]);
    int M = 4096, K = 4096, N = 4096;

    // Allocate
    float *A, *B, *C, *C_ref;
    cudaMalloc(&A, M * K * sizeof(float));
    // ... (utils helpers handle this)

    // Reference
    cublas_matmul(A, B, C_ref, M, K, N);

    // Selected kernel
    float ms;
    switch (kernel) {
        case 0: ms = run_kernel(cublas_matmul, A, B, C, M, K, N); break;
        case 1: ms = run_kernel(sgemm_naive, A, B, C, M, K, N); break;
        case 2: ms = run_kernel(sgemm_tiled, A, B, C, M, K, N); break;
        default: printf("Unknown kernel %d\n", kernel); return 1;
    }

    verify(C, C_ref, M * N, 1e-3);
    report_performance(M, K, N, ms);

    // Cleanup
    cudaFree(A); // ...
    return 0;
}
```

## Kernel Stub Pattern

Each kernel file contains the function signature and a TODO marker:

```cpp
#pragma once
#include <cuda_runtime.h>

__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float *A, const float *B,
                            float beta, float *C) {
    // TODO: Your implementation here
    // Each thread computes one element of C.
    // C[row][col] = alpha * dot(A[row,:], B[:,col]) + beta * C[row][col]
}
```

Kernel stubs use float (SGEMM) for matmul exercises, matching Simon Boehm's approach. This avoids the dtype dispatch complexity encountered with popcorn's half-precision tests and lets the learner focus on memory hierarchy concepts rather than type plumbing.

## Cloud GPU Setup

The cloud GPU landscape for CUDA profiling is harsher than commonly assumed. `ncu` requires access to GPU hardware performance counters via either (a) host-kernel `NVreg_RestrictProfilingToAdminUsers=0` modprobe, or (b) `--cap-add=SYS_ADMIN` on the container. Neither is reachable on shared-tenancy cloud providers — VMs block counters at the hypervisor, containers refuse `SYS_ADMIN` for security reasons. Verified primary sources (April 2026):

| Provider | `make run` / `make bench` | `make profile` (ncu) | Notes |
|----------|-----|------|-------|
| Lambda Labs On-Demand | Works | **Blocked** — hypervisor-level | Staff-confirmed Sep 2025 ([DeepTalk](https://deeptalk.lambda.ai/t/nvidia-profiling-support/4763)) |
| Lambda 1-Click / Private Cloud | Works | Works (bare metal) | ≥16 GPUs / monthly commit — wrong sizing for a learner |
| RunPod Secure Cloud | Works | **Blocked** — unprivileged containers | Staff-confirmed ([answeroverflow](https://www.answeroverflow.com/m/1225015304682471454)) |
| RunPod Community Cloud | Works | **Blocked** — same container model | |
| Vast.ai | Works | **Per-host variance** | Must test each listing with the validation step below |
| Hetzner / OVH / Vultr bare-metal GPU | Works | Works | Most reliable rent-for-profiling; you are root on host |
| Personal workstation with NVIDIA GPU | Works | Works | Cheapest long-term; used 3090 ~$600 |
| University cluster | Works | Usually works | Already configured; requires institutional access |

### Recommended workflow

Split run/bench from profile. For phases 1-4 (matmul, reduction, cross_entropy), use any cheap container instance — `ncu` is not on the critical path this early. When the curriculum reaches phase 5+ and wall-clock timing stops being diagnostic, switch profiling to a bare-metal or personally-owned host.

Concrete setup docs:
- `docs/runpod-setup.md` — run/bench workflow on RunPod (L4 for phases 1-4, H100 for phase 5+).
- `docs/lambda-setup.md` — same story for Lambda On-Demand, plus the profiling landscape in full detail.

### Validation step

Before starting any profiling session, confirm counters are accessible:

```bash
make run EX=matmul K=1                          # confirm the kernel runs
ncu --section SpeedOfLight /bin/true            # confirm counters accessible
```

If `ncu` returns `ERR_NVGPUCTRPERM`, counters are blocked on this host and no amount of `NVreg_RestrictProfilingToAdminUsers=0` will help (because you're a guest, not the host). Terminate and move to a bare-metal provider.

### Compute capability reference

| GPU | Compute Capability | Typical use |
|-----|-------------------|-------------|
| L4 | 89 | Budget exercises (Phases 1-4) |
| A100 | 80 | Full profiling (Phase 5+) |
| H100 | 90 | Large-scale kernels, FlashAttention |

Set `CUDA_COMPUTE_CAPABILITY` in CMakeLists.txt to match your GPU.

## Migration Plan

### Archive

Move the following directories to `archive/` within the new repo:
- `pmpp/` (original popcorn problems)
- `pmpp_v2/` (v2 popcorn problems, including completed vectoradd and grayscale)
- `princeton/` (Princeton problem set)

These contain working code from earlier exercises and should be preserved for reference.

### Migrate

- `wiki/` moves into the new repo root. Existing pages (type-dispatch, blocks-and-threads, gpu-cpu-split-compilation, gpu-cpu-async, warps-and-hardware-constraints) are retained.
- `docs/study-guide.tex` moves into `docs/`. Will be revised separately to reflect the trimmed critical path.

### Git

Initialize as a new git repository. The old `gpumode/` directory was not a git repo, so there is no history to preserve.

## Acknowledgements

The README will include:

```markdown
## Acknowledgements

The build system, profiling workflow, and kernel progression in this project
are directly inspired by Simon Boehm's
[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
and its companion repository [siboehm/CUDA-MMM](https://github.com/siboehm/CUDA-MMM).

The curriculum follows *Programming Massively Parallel Processors* (4th ed.)
by Hwu, Kirk, and El Hajj, supplemented by exercises from the
[GPU Mode](https://discord.gg/gpumode) community.
```

## Resolved Questions

1. **Matrix sizes:** Configurable via CLI args, defaulting to 4096x4096 (Simon's approach). Usage: `./build/matmul 1` uses defaults; `./build/matmul 1 2048 2048 2048` overrides M, K, N.
2. **Benchmark chart generation:** Left as a TODO stub. Can adapt Simon's `plot_benchmark_results.py` when multiple kernel variants exist.
3. **Float precision:** Stubs use float32 (SGEMM). Mixed-precision handling is per-exercise (quantized_gemm will introduce its own types). No global type dispatch system.
