# Shared-Memory Tiling

Load a rectangular tile of A and B into on-chip shared memory once per block, then reuse each loaded element across every thread in the block that needs it. This is the first optimization that changes *how much* memory traffic the kernel does — coalescing only changed *how* the traffic was packed.

## The memory hierarchy

Three tiers, ordered by speed and proximity to the ALU:

```
GMEM (HBM, off-chip)         1×         shared across all SMs, GB scale
SMEM (on-chip, per-SM)       ~15-20×    per-block allocation, ~10s of KB per SM
registers (per-thread)       ~100×+     fastest, smallest, private per thread
```

SMEM is the only programmer-controlled cache — it's a scratchpad, not a transparent cache. You pay to move data in (`__shared__` declaration + explicit load), but then every thread in the block can read it for free.

## The tile

Assign one output tile of size BM×BN to each block. To compute it, we need the corresponding BM rows of A and BN columns of B — but only BK columns of A and BK rows of B at a time, because the inner K dimension is summed:

```
                        K
      ┌─────────────────────────────────┐
      │   ┌────┐ slide ──►              │
   BM │   │ As │                        │   A: BM × K
      │   │BM×BK│                       │
      │   └────┘                        │
      └─────────────────────────────────┘

      ┌────┐
      │    │
      │ Bs │  ┐
   BK │BK×BN│  │ slide down with As
      │    │  ▼
      ├────┤
      │    │
      │    │                              B: K × N
      │    │
      │    │
      └────┘
        BN
```

One outer iteration: every thread in the block cooperatively loads a slice of As and Bs from GMEM into SMEM, `__syncthreads()`, then each thread (or group) does its share of the BM×BN × BK partial dot products using only SMEM reads. Slide along K, repeat K/BK times, accumulate into C.

## Why it wins

Without tiling, computing one element of C reads 2K floats from GMEM. With a BM×BN×BK tile, the same C-element still needs 2K floats *logically* — but each float is loaded from GMEM once per block and reused inside the block:

- each loaded element of A is used BN times (once by every column in the output tile)
- each loaded element of B is used BM times

**GMEM loads per result drop by a factor of BM (for B) or BN (for A).** For square BM=BN=32, that's a 32× reduction in DRAM traffic per result. Arithmetic intensity climbs from ~0.25 FLOPs/byte to the tens.

## Cost

- **A synchronization** — `__syncthreads()` between the cooperative load and the compute means every thread in the block stalls until the slowest loader finishes.
- **SMEM capacity** — a block can stage at most what fits in the SM's SMEM partition. More SMEM per block → fewer resident blocks → possibly lower occupancy (see `occupancy.md`).
- **Code complexity** — the kernel now has three nested levels (block outer over K-chunks; cooperative load; per-thread compute) plus a barrier.

## What it still doesn't fix

Every FMA in the compute loop still reads its two operands from SMEM. The compute instructions become dominated by `LDS` (load-from-SMEM) — the SMEM pipeline saturates before the ALUs do. The next lever pulls repeated operands out of SMEM and into registers: see `register-tiling.md`.

## Context

**Source:** siboehm.com/articles/22/CUDA-MMM (Kernel 3)
**Question that led here:**
