# Warps and Hardware Constraints

## What is a warp

32 threads that execute in lockstep. This is silicon, not software — a warp scheduler issues one instruction and 32 ALUs execute it simultaneously on different data. This is SIMT (Single Instruction, Multiple Threads).

A block of 256 threads = 8 warps. The SM schedules warps, not individual threads.

## Warp divergence

If threads in a warp hit an `if/else`, the warp executes BOTH branches. Threads that shouldn't run a branch get masked off — they do nothing but still waste cycles. This is why the study guide flags control divergence as a key Phase 2 concept.

## Block size constraints

**Hard limits (launch fails):**
- Maximum 1024 threads per block
- Must be a positive integer

**Soft constraints (runs but hurts performance):**
- **Not a multiple of 32** — `threads=100` gives 3 full warps + one partial warp of 4. 28 idle lanes wasted.
- **Too few threads** — SM can't hide memory latency. While one warp waits for memory, the SM switches to another ready warp. More warps = more latency hiding ("occupancy").
- **Too many threads** — each thread uses registers. More threads = fewer registers each, risking "register spilling" to slow local memory.

## Why 256

- 256 / 32 = 8 warps (clean division)
- Fits within 1024 with room for multiple blocks per SM
- Enough warps to hide latency without crushing register pressure
- 128, 256, 512 are all common; the right number depends on register/shared memory usage (profiler territory)

## Context

**Exercise:** grayscale_v2 (Phase 1)
**Question that led here:** "What is a warp? What stops me from assigning an arbitrary number of threads per block?"
