# GPU Performance Engineering Learning Plan

**Author:** Amiya Diwan
**Date:** 2026-04-13
**Status:** Approved

## Goal

Develop research taste in GPU performance engineering for frontier AI inference optimization. Not just kernel competence — the ability to diagnose which kernel matters, why it's slow, and what class of optimization helps, grounded in hardware reasoning that transfers across platforms and generations.

## North Star Metric

Policy improvement per dollar of rollout compute. Every kernel optimization matters if it increases rollout throughput per dollar. (From The Critical Path: RL at Scale)

## Guiding Principles

1. **Follow the problem, not the silo.** Cross-disciplinary thinking over specialized titles.
2. **Prioritize ruthlessly and test cheaply.** Find the riskiest assumption, test at smallest informative scale.
3. **Cultivate taste.** Predict -> validate -> explain -> prioritize. AI is your tool, not your crutch.
4. **Don't overfit to the stack.** Extract transferable reasoning about WHY systems behave as they do.

## Learning Method

The "Cultivate Taste" loop applied at every step. Each problem follows this protocol:

1. Write a naive version and explain its bottleneck
2. Write an optimized version and predict the speedup before measuring
3. Explain the gap between predicted and actual
4. Name the transferable principle (not just the CUDA syntax)

Theory is interleaved with coding — never more than 45 minutes of reading before writing a kernel.

## Resources

- **PMPP 4th Edition** (Hwu, Kirk, El Hajj) — Kindle
- **GPU Mode popcorn-cli** v1.3.12 — remote T4 GPU for submissions
- **The Critical Path: RL at Scale** (Diwan, 2026) — bridge to inference optimization
- **Cloud GPU** (Lambda/RunPod/Vast.ai) — for profiling with nsys/ncu
- **KernelBench** (Stanford) — 250-task benchmark for kernel generation
- **KernelLLM-8B** (Meta, HuggingFace) — open-weight model for capstone

## Tools

- `torch.utils.cpp_extension.load_inline()` for CUDA kernels in Python submissions
- `popcorn submit submission.py` for leaderboard submissions
- `nsys` / `ncu` for profiling (cloud GPU)
- Triton (learned in context during capstone, not as separate phase)

---

## Phase 1: Hello World (~2 hours)

**Goal:** Working CUDA toolchain, basic thread indexing.

### Step 1: Read PMPP Ch 1 (skim, ~15 min)
- Section 1.1 only: throughput-vs-latency mental model, host/device distinction
- Skip history, skip other programming interfaces

### Step 2: Read PMPP Ch 2 sections 2.1-2.5 (~30 min)
- 2.1 Data parallelism
- 2.2 CUDA C++ program structure (host code, kernel launches, grids)
- 2.3 A vector addition example
- 2.4 Device global memory and data transfer (cudaMalloc, cudaMemcpy)
- 2.5 Calling kernel functions (<<<blocks, threads>>>)

### Step 3: vectoradd_v2
- **Timebox:** 30 minutes max. If not submitted in 30 min, your toolchain is broken.
- Wire up `load_inline()`, write the kernel, submit via popcorn-cli.
- **Predict:** What happens if you launch fewer threads than elements?
- This is a toolchain smoke test, not a learning exercise.

### Step 4: Read PMPP Ch 3 (~30 min)
- Multidimensional grids and data
- Mapping threads to multidimensional data, row-major linearization
- 2D grid -> 1D memory mapping

### Step 5: grayscale_v2
- **Predict:** Does the weighted sum formula (0.2989R + 0.587G + 0.114B) affect memory access pattern, or just arithmetic?
- First real "predict before running" exercise.
- Cements 2D thread indexing with a concrete visual problem.

---

## Phase 2: The Architecture Turn (~4-5 hours)

**Goal:** Understand GPU hardware well enough to make informed predictions. Matmul is done twice — the most important exercise sequence in the entire plan.

### Step 6: Read PMPP Ch 4 sections 4.1-4.4 (~30 min)
- 4.1 Architecture of a modern GPU (SMs, SPs)
- 4.2 Block scheduling
- 4.3 Synchronization and transparent scalability
- 4.4 Warps and SIMD hardware (32-thread lockstep)

### Step 7: Read PMPP Ch 4 sections 4.5-4.6 (~20 min)
- 4.5 Control divergence (why branches are expensive)
- 4.6 Warp scheduling and latency tolerance (how GPU hides memory latency)

### Step 8: Read PMPP Ch 5 sections 5.1-5.3 (~35 min)
- 5.1 Memory access efficiency
- 5.2 CUDA memory types (global, shared, constant, registers)
- 5.3 Tiling for reduced memory traffic
- **Before coding:** Trace the tiled matmul algorithm by hand on a 4x4 example.

### Step 9: matmul_v2 (naive)
- **Predict:** What fraction of peak FLOPS will you hit? (Expect <5%)
- Profile it on cloud GPU if possible.
- **Explain:** Why is it slow, in terms of memory transactions per FLOP?

### Step 10: Read PMPP Ch 5 remaining + Ch 6 sections 6.1-6.4 (~40 min)
- Memory coalescing, occupancy, shared memory bank conflicts
- You now have a slow kernel. This reading should feel urgent — you're diagnosing a real problem.

### Step 11: matmul_v2 (tiled with shared memory)
- **Predict the speedup before measuring.**
- **Explain the gap between predicted and actual.**
- **This is the single most important exercise in the entire plan.** The memory hierarchy lesson lands here.

### Step 12: Read PMPP Ch 6 sections 6.5-6.9 (~30 min)
- 6.5 Thread coarsening
- 6.6 Loop unrolling
- 6.7 Double buffering
- 6.8 A checklist of optimizations
- 6.9 Optimization strategy

---

## Phase 3: Parallel Patterns (~6-8 hours)

**Goal:** Implement each fundamental parallel pattern. For each: read the chapter, do the problem, articulate the transferable principle.

### Step 13-14: Histogram
- Read PMPP Ch 9 sections 9.1-9.4 (~25 min): atomics, basic histogram, latency/throughput of atomics, privatization
- **histogram_v2** — write naive global atomics first, then privatize to shared memory. Do both versions.
- Read Ch 9 sections 9.5-9.6 (15 min): thread coarsening, thread-level privatization. Optimize further.
- **Transferable principle:** Contention management via privatization. Applies to gradient accumulation, token counting, anywhere concurrent writes collide.

### Step 15-16: Reduction
- Read PMPP Ch 10 sections 10.1-10.5 (~30 min): reduction trees, simple kernel, control divergence, memory access divergence
- **vectorsum_v2** — write the divergent version first, measure, then fix.
- Read Ch 10 sections 10.6-10.11 (25 min): reducing global memory accesses, synchronization overhead, thread coarsening.
- **Transferable principle:** Tree-based parallel algorithms. Reduction is a primitive in every ML framework (loss functions, norms, softmax denominators).

### Step 17-18: Convolution
- Read PMPP Ch 7 sections 7.1-7.4 (~25 min): convolution background, basic kernel, memory bandwidth, constant memory
- **conv2d_v2** — basic parallel convolution.
- Read Ch 7 sections 7.5-7.6 (20 min): tiled convolution with halo cells, using caches for halo cells. Optimize.
- **Transferable principle:** Halo exchange. Same pattern in distributed training, stencil computations, fluid dynamics.

### Step 19-20: Scan (Prefix Sum)
- Read PMPP Ch 11 sections 11.1-11.2 (~25 min): scan background, Kogge-Stone algorithm
- **prefixsum_v2** — implement Kogge-Stone.
- Read Ch 11 remaining (20 min): work efficiency, Brent-Kung, segmented scan.
- **Predict:** Kogge-Stone vs Brent-Kung — which wins at N=1024 vs N=1M?
- **Transferable principle:** Work-efficiency vs span tradeoff — the fundamental tension in parallel algorithm design.

### Step 21-22: Sort
- Read PMPP Ch 13 (merge) + Ch 14 sections 14.1-14.5 (~40 min): merge algorithm, co-rank, radix sort, parallel merge sort
- **sort_v2** — compose scan + merge into a sort.
- Read Ch 14 sections 14.6-14.8 (15 min): 14.6 Thread coarsening to improve memory coalescing, 14.7-14.8 other parallel sort methods.
- **Predict:** At what N does your GPU sort beat `torch.sort`?
- **Transferable principle:** Composing parallel primitives. Sort = scan + merge + scatter.

---

## Phase 4: Capstone Popcorn Problems (~3-4 hours)

**Goal:** Apply pattern composition to non-standard problems.

### Step 23: trimul (BioML)
- Triangle matrix multiplication. Adapts matmul optimization to a non-standard shape.
- Tests "follow the problem, not the silo" — can you modify your tiled matmul for triangular structure?

### Step 24: princeton_cross_entropy (Princeton)
- Cross-entropy = softmax (reduction) + log + element-wise ops.
- Directly relevant to inference workloads. Fusing these operations is a common production optimization.
- Foundation for the fused softmax+CE kernel in Phase 6.

---

## Phase 5: Bridge to Inference (~half day)

**Goal:** Connect PMPP hardware knowledge to real inference workloads. Set up profiling infrastructure.

### Step 25: Set up cloud GPU
- Provision a cloud instance with NVIDIA GPU (Lambda/RunPod/Vast.ai)
- Install nsys, ncu, FlashAttention, PyTorch
- Verify profiling workflow works

### Step 26: Think First — Autoregressive Decode Arithmetic Intensity (rl-at-scale p.17)
- Compute arithmetic intensity for single-head attention decode: I = 4dS / 4dS = 1 FLOP/byte
- Place it on the roofline for H100 (ridge point = 148 FLOP/byte)
- **Key insight:** Decode is deep in the memory-bound regime. Arithmetic units idle >99% of the time. No amount of faster compute fixes this — only bandwidth or data movement reduction helps.

### Step 27: FlashAttention Roofline Profiling (rl-at-scale p.19)
- Profile FlashAttention decode latency across batch sizes 1, 8, 32, 64
- For each batch size, compute: achieved HBM bandwidth, arithmetic intensity, attainable TFLOP/s from roofline
- Plot all four points. Identify the crossover batch size where decode transitions from memory-bound to compute-bound.
- **This exercise connects everything:** PMPP Ch 4-6 (architecture, memory, performance) applied to a real inference kernel.

### Step 28: Read rl-at-scale Ch 2 Move 2 (p.18-21)
- Attention kernels: why naive attention is unacceptable (O(N^2) materialization), FlashAttention insight (fuse everything, materialize nothing, online softmax)
- Quantized matmuls: arithmetic intensity by precision (FP16 -> FP8 -> FP4), decode is bandwidth-bound regardless
- MoE routing: gather-compute-scatter, load imbalance, sparse dispatch
- This is the "why" behind Phase 6. Read it before implementing.

---

## Phase 6: Critical Kernel Implementations (~4-6 weeks)

**Goal:** Build the kernels that dominate inference wall time. All simplified — focus on algorithmic insight, not micro-optimization.

### Step 29: Fused softmax + cross-entropy (~1 week)
- Single-pass reduction: compute max, subtract, exponentiate, sum, normalize, log, multiply by targets — all in one kernel pass
- No N-element intermediate tensor materialized to DRAM
- Builds on princeton_cross_entropy from Phase 4
- **Why it matters:** Kernel fusion is the #1 most common optimization in production inference. This is the simplest example that demonstrates the full pattern.

### Step 30: FlashAttention forward pass (~2 weeks)
- **Scope:** Single-head, FP32, no masking, no dropout, forward pass only
- **Core algorithm:** Online softmax (running max + denominator), tiled QKV streaming through SRAM, accumulate output without materializing N x N attention matrix
- **Not in scope:** Tensor cores (mma.sync), async copy (cp.async), multi-stage pipelines, register pressure optimization
- **Reference:** Tri Dao's Algorithm 1 pseudocode, rl-at-scale Ch 2 Move 2a
- **Why it matters:** The single most important inference kernel. HBM traffic drops from O(N^2 + Nd) to O(Nd) — a pure memory-hierarchy optimization. The algorithm is identical to what you learned in PMPP Ch 5 tiling, applied to attention.

### Step 31: Quantized GEMM — INT4 weight-only (~1-2 weeks)
- **Scope:** INT4 weights dequantized to FP32 on-the-fly, fused with matmul. Weight-only quantization (activations stay FP32).
- Load INT4 weights (0.5 bytes/element), dequantize in registers, accumulate in FP32
- **Why it matters:** For decode at batch=1, the weight matrix load from DRAM sets a hard floor on latency. INT4 quarters the bytes loaded, quartering the minimum latency. This is a bandwidth optimization, not a compute optimization. (rl-at-scale p.19)

### Step 32: MoE dispatch (stretch goal, ~2 weeks if attempted)
- Gather-compute-scatter for top-k expert routing
- Irregular memory access patterns, dynamic per-expert batch sizes
- **Why it matters:** MoE forward passes underperform their theoretical arithmetic intensity due to dispatch overhead, load imbalance, and communication overhead. An unsolved problem at frontier labs.

---

## Phase 7: KernelBench Challenge (~2-3 weeks)

**Goal:** Validate kernel skills against a standardized benchmark. Prepare for capstone.

- Stanford's KernelBench: 250 tasks across 3 difficulty levels
- **L1:** Simple operations — warmup, validate skills. Target: solve most.
- **L2:** Standard patterns — matmul variants, attention, norms. Best LLMs solve ~40%. Target: match or beat LLM baselines.
- **L3:** Complex fusions — multi-kernel pipelines. Best LLMs solve ~18%. Target: attempt, learn from failures.
- **Meta-goal:** Build intuition for what makes kernel generation hard — what patterns do LLMs struggle with? This directly informs the capstone.

---

## Phase 8: Capstone — KernelLLM Narrow Fine-Tune (~6-8 weeks)

**Goal:** Apply RL to improve LLM kernel generation on a narrow domain.

### Setup
- Base model: Meta's KernelLLM-8B (open-weight, HuggingFace)
- Target language: Triton (KernelLLM's native target — Triton learned here in context)
- Domain: Narrow — attention kernels OR reduction/scan patterns (pick one)
- Eval: KernelBench subset for chosen domain

### Method
- RL algorithm: GRPO (standard for RLVR in kernel generation)
- Reward signal: compilation success + correctness via test cases + wall-clock execution time
- Training loop: generate kernel -> compile -> execute -> measure -> reward -> update
- Cloud GPU: few A100s/H100s for RL training (generation + kernel execution)

### What Makes This Not Just Replication
- Existing work (CUDA-L1, DRTriton, Dr. Kernel) trains on broad benchmarks. A narrow-domain fine-tune with carefully designed reward shaping on a specific kernel family is a focused contribution.
- Your Phase 6 experience building these kernels by hand gives you unique insight into reward design — you know what makes a kernel fast vs. just correct.
- Anti-reward-hacking: DRTriton and Dr. Kernel both had to invent mechanisms for this. Your RL background (you wrote the textbook) is a genuine advantage.

### Deliverable
- Fine-tuned model weights
- Eval results vs KernelLLM baseline and other published results
- Analysis: what did RL learn? What optimization patterns does it discover? Where does it still fail?

---

## Phase 9: Cross-Platform + Portfolio

**Goal:** Demonstrate transferable understanding and end-to-end systems thinking.

### Step: Cross-Platform Attention Optimization (rl-at-scale p.23)
- Now that you've built FlashAttention in CUDA, map the same optimization to TPU/Pallas
- Focus on conceptual mapping (shared memory -> VMEM, __syncthreads -> implicit tile sync, double buffering -> Buffered(buffer_count=2))
- Tests principle 4: if you can only write it in CUDA, you learned syntax, not concepts

### Portfolio artifacts
- GitHub repo with all kernel implementations (popcorn submissions + Phase 6 kernels)
- KernelBench results and analysis
- KernelLLM fine-tune results, model weights, training curves
- Written analysis connecting hardware reasoning to optimization decisions

---

## Phase 10: Circular Read

Reread PMPP Ch 4-5 and rl-at-scale Ch 2. The understanding is circular — after building these kernels, the same pages will read differently.

---

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 1: Hello World | 2 hours | 2 hours |
| 2: Architecture Turn | 4-5 hours | ~7 hours |
| 3: Parallel Patterns | 6-8 hours | ~15 hours |
| 4: Capstone Popcorn | 3-4 hours | ~19 hours |
| 5: Bridge to Inference | half day | ~23 hours |
| 6: Critical Kernels | 4-6 weeks | ~5-7 weeks |
| 7: KernelBench | 2-3 weeks | ~8-10 weeks |
| 8: Capstone (KernelLLM) | 6-8 weeks | ~16-18 weeks |
| 9: Portfolio | 1-2 weeks | ~18-20 weeks |

Phases 1-5 can be done in a single focused week. Phases 6-9 are the real work.

---

## Stress-Test Results (2026-04-13)

Changes made after adversarial review:

1. **Cut v1 problems** — v2 only. Core algorithms identical; API differences don't justify doubling the work.
2. **Cherry-picked rl-at-scale exercises** — kept arithmetic intensity (p.17) and FA roofline profiling (p.19). Cut: quantization-for-RL (off-topic), decode chip (too abstract), cross-platform (premature — moved to Phase 9).
3. **Scoped Phase 6 kernels to simplified versions** — algorithmic insight over micro-optimization. MoE dispatch is stretch goal.
4. **Replaced capstone** — "train LLM to write kernels via RL" is a crowded field (10+ papers). Narrowed to KernelLLM fine-tune on specific domain. Declarative kernel language idea contradicted — no evidence a new DSL outperforms Triton.
5. **Replaced GPU Mode competitions with KernelBench** — hardware-agnostic, well-documented, directly relevant to capstone.
6. **Merged Triton learning into capstone** — no standalone Triton rewrite phase. Learn Triton in context via KernelLLM.
7. **Added cloud GPU setup** as explicit step. User is on macOS; profiling requires interactive GPU access.

## Key References

- PMPP 4th Edition (Hwu, Kirk, El Hajj)
- The Critical Path: RL at Scale (Diwan, 2026)
- FlashAttention-2 (Dao, 2023) / FlashAttention-4 (Zadouri et al., 2026)
- KernelBench (Stanford, 2025)
- KernelLLM (Meta, 2025) — HuggingFace: facebook/KernelLLM
- CUDA-L1 (arXiv 2507.14111) — contrastive RL for CUDA
- DRTriton (arXiv 2603.21465) — synthetic data RL for Triton
- Dr. Kernel (arXiv 2602.05885) — RL for Triton with anti-reward-hacking
