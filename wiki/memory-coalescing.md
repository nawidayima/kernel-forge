# Memory Coalescing

How a warp's 32 thread-level loads become DRAM transactions. This is the first-order determinant of kernel performance on any memory-bound problem.

## How a warp forms from threadIdx

Blocks are written in 3D (`threadIdx.x/y/z`), but the hardware groups threads into warps along a **1D linearization**:

```
tid = threadIdx.z * (blockDim.y * blockDim.x)
    + threadIdx.y * blockDim.x
    + threadIdx.x
```

Warps are contiguous slices: `warp_id = tid / 32`, `lane = tid % 32`.

Consequence: `threadIdx.x` is the innermost, fastest-varying index. Within a single warp, `threadIdx.x` takes 32 consecutive values while `threadIdx.y/.z` stay fixed.

**Rule of thumb:** whichever array index you derive from `threadIdx.x` is the one that changes across lanes of a warp.

## Memory transactions

DRAM isn't loaded byte-at-a-time. The smallest unit is a **128-byte cache line** (= 32 floats), aligned to 128-byte boundaries.

When a warp issues a global load, hardware inspects the 32 addresses the lanes want and decides how many 128-byte transactions to issue. Fewer transactions = better bandwidth efficiency.

## Three access patterns

| Pattern | 32 lanes request | Transactions | Bandwidth efficiency |
|---|---|---|---|
| **Coalesced** | 32 contiguous floats, aligned | 1 | 100% |
| **Broadcast** | All the same address | 1 (fanned out) | 100%, plus free reuse |
| **Uncoalesced** | 32 scattered addresses | up to 32 | as low as 3% |

Uncoalesced at stride ≥ 32 pulls 32 × 128 B = 4 KiB when the warp actually uses 128 B. **97% of the fetched bandwidth is wasted.**

Stores follow the same rules.

## Applied to naive SGEMM

Inner loop: `sum += A[row*K + k] * B[k*N + col]`.

With block(32,32) and `threadIdx.x → col`, within one warp: `row` is constant, `col` varies 0–31. Therefore:

- `A[row*K + k]` — same address for all 32 lanes → **broadcast** (1 transaction)
- `B[k*N + col]` — 32 contiguous floats → **coalesced** (1 transaction)
- `C[row*N + col]` (once) — **coalesced store**

Every global access is optimal.

**The footgun:** if instead `threadIdx.x → row`, within a warp `row` varies 0–31, `col` is constant. Now `A[row*K + k]` asks for 32 floats at stride K=4096 → **32 transactions per warp per k**, 97% waste. This is Simon Boehm's "kernel 1 → kernel 2" jump (~8×). Avoiding it is a choice of which `threadIdx` component drives which matrix index.

## The DRAM-bound floor

For any naive memory-bound kernel, an instructive upper bound on runtime:

```
total_bytes = (total threads) × (bytes loaded per thread)
min_runtime = total_bytes / DRAM_bandwidth
max_gflops  = total_flops / min_runtime
```

For 4096³ SGEMM with **zero cache reuse**: 512 GB / 300 GB/s ≈ 1.71 s → **~80 GFLOP/s floor on L4**. The observed number is always higher than this (caches catch broadcasts and line-adjacent reuse), but never by much until you explicitly tile into shared memory — which is the whole point of the next exercise.

## Context

**Exercise:** matmul/1_naive (Phase 1)
**Question that led here:** "What does it mean that warps 'form along threadIdx.x', and why does that change memory traffic?"
