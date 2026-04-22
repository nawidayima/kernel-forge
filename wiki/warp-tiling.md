# Warp Tiling

Blocks map to SMs. Threads map to CUDA cores. What maps to the warp? In naive kernels, nothing — warps are an emergent consequence of how threadIdx linearizes. But several hardware facts *only* make sense at warp granularity, so explicit warp-level tiling is the bridge between block-level and thread-level work.

## Why warps need their own tier

Three hardware facts, none of which the block/thread decomposition exposes:

1. **Warp schedulers are the real dispatch unit.** Each SM has a small number (typically 4) of warp schedulers. Every clock, each scheduler picks one eligible warp and issues one instruction. A block with 8 warps rotates those 4 schedulers — warps on the same scheduler run *concurrently*, warps on different schedulers run *in parallel*. Tile layout should let each scheduler own a contiguous piece of work.
2. **Shared-memory bank conflicts are per-warp.** The 32 banks are checked once per warp cycle; threads in *different* warps never conflict on the same cycle. Only the 32 lanes *within* the same warp can collide, so layout decisions that avoid conflicts must be made at warp granularity.
3. **The register cache is warp-local.** Recent GPUs cache the last few register reads per warp. Tight register-tile locality *within a warp* hits this cache; scattering the same work across warps does not.

Plus the forward-looking fact: **tensor-core / MMA instructions are inherently warp-wide** — a single `mma` instruction consumes fragments contributed by all 32 lanes. Any tile hierarchy that wants to lower to tensor cores must already have a warp-sized unit.

## The nested hierarchy

```
Grid
 └── Block tile (BM × BN)              ← assigned to one SM
      └── Warp tile (WM × WN)          ← assigned to one warp scheduler
           └── Thread tile (TM × TN)   ← per-thread register rectangle
                └── Register tile      ← regM[TM], regN[TN], acc[TM*TN]
```

Example: a block tile of 128×128 split into 2×4 warp tiles of 64×32; each warp tile carries the 32 threads of its warp, each thread owning TM×TN register entries. The geometry is chosen so each warp's 32 lanes cover a contiguous sub-region — not a scattered set of C elements pulled from across the block.

```
Block tile 128×128 (numbers = warp id)

  ┌────────┬────────┬────────┬────────┐
  │   0    │   1    │   2    │   3    │  WM = 64
  │        │        │        │        │
  ├────────┼────────┼────────┼────────┤
  │   4    │   5    │   6    │   7    │  WM = 64
  │        │        │        │        │
  └────────┴────────┴────────┴────────┘
     WN=32    WN=32    WN=32    WN=32

  Each 64×32 warp tile is owned by one warp.
  Inside: 32 threads each compute a TM × TN thread tile.
```

## Why this changes performance

Without an explicit warp tile, consecutive threads in a warp may touch SMEM addresses that land across different banks or different cache lines depending on how threadIdx linearizes into the tile. With a warp tile, the 32 lanes of a warp are guaranteed to access a coherent footprint: one contiguous SMEM row per cycle, one 128B cache line, one bank per lane. The SMEM pipeline stops stalling on conflicts, the register cache starts hitting, and the compiler can emit the widest `LDS.128` loads.

Quantitatively this is usually a single-digit percent gain over a well-tuned 2D blocktile — small absolute, but at 90%+ of a production BLAS's throughput every percent is hard-fought. It's also *the* structural change required to later swap the outer-product inner loop for tensor-core `mma` instructions.

## Context

**Source:** siboehm.com/articles/22/CUDA-MMM (Kernel 10)
**Question that led here:**
