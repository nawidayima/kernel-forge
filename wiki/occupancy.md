# Occupancy

The fraction of an SM's maximum resident warps that are actually resident at a given moment. Intuitively: how much independent work the scheduler has on hand to hide memory latency. Occupancy is capped by whichever of three finite resources per SM runs out first.

## The three limiters

Each SM has hardware caps on:

| Resource | Typical scale | Consumed by |
|---|---|---|
| Threads (= warps × 32) | ~1500–2000 threads, ~48–64 warps | every resident thread |
| Register file | tens of thousands of 32-bit regs | regs/thread × threads |
| Shared memory | ~50–100 KB | SMEM allocated per resident block |

Maximum blocks resident per SM is bounded by the tightest of:

```
max_blocks_per_SM = min(
    threads_per_SM     / threads_per_block,
    registers_per_SM   / (regs_per_thread × threads_per_block),   // rounded to granularity
    smem_per_SM        / smem_per_block
)

occupancy = max_blocks_per_SM × warps_per_block / max_warps_per_SM
```

Compute this by hand before trusting the profiler. One of the three limiters is almost always binding, and knowing *which* one tells you what to optimize.

## Register file is bigger than SMEM

An underappreciated asymmetry: the register file on an SM is typically **several times larger** than shared memory (often ~5×). This is why register tiling is cheap in occupancy terms — doubling registers per thread rarely binds occupancy before SMEM or thread count does. Spending registers on a big per-thread accumulator is usually the right trade.

## Why more occupancy isn't always better

Occupancy exists to hide latency: while one warp waits for memory, the scheduler issues from another. But if the kernel already has

- enough **instruction-level parallelism** — independent FMAs the compiler can reorder to keep the ALU busy within a *single* warp, **and**
- high enough **arithmetic intensity** that it rarely waits on memory in the first place,

then additional resident warps contribute nothing. Volkov's thesis (2016) documented the canonical *cusp behavior*: GFLOP/s rises sharply with occupancy up to a knee (often ~30–50% for compute-bound kernels) and then flattens or even regresses. A hand-optimized SGEMM can run at ~50% occupancy and still hit 95%+ of a production BLAS, because the per-thread tile already holds dozens of independent FMAs and the SMEM traffic is amortized.

## The trade-off

Enlarging the per-thread tile:

- ✓ raises arithmetic intensity (each thread amortizes more memory traffic)
- ✓ more ILP per thread (hides latency without needing more warps)
- ✗ raises register pressure (may bound occupancy)
- ✗ raises SMEM per block (may bound occupancy)

The question is never "maximize occupancy." It's:

> **Which limiter is currently binding my throughput — bandwidth, memory latency, or ILP — and which knob relieves it without starving another?**

## Context

**Source:** siboehm.com/articles/22/CUDA-MMM (Kernel 3, referencing Volkov 2016)
**Question that led here:** "What hardware constraints does CUDA enforce on the tile and block dimensions I can pick?"
