# Blocks and Threads: Why the Two-Level Hierarchy

## The concept

A flat list of threads can't scale. The GPU needs a grouping unit that maps to physical hardware — that's the block.

A block is a group of threads assigned to one SM (streaming multiprocessor). All threads in a block:
- Share fast on-chip memory (`__shared__`)
- Can synchronize with each other (`__syncthreads()`)
- Share a pool of registers

Threads in different blocks **cannot** cooperate. They may run on different SMs.

## Why this enables scaling

An L4 has 58 SMs. An H100 has 132. Same kernel, same code — the hardware just distributes blocks across more SMs. You never control which SM gets which block. The study guide calls this "transparent scalability."

## The initialization (CPU side)

```cpp
const int threads = 256;                          // threads per block (multiple of 32, because warps)
const int blocks = (N + threads - 1) / threads;   // ceiling division
```

N=1000: 4 blocks x 256 threads = 1024 total. 24 threads exit early via `if (idx < N)`.

Each block is split into warps of 32 threads — the actual unit the hardware schedules.

## When blocks matter

For vectoradd/grayscale, threads don't cooperate — blocks are just an organizational detail. Blocks become essential in tiled matmul (Phase 2) where threads in a block load shared tiles together.

## Context

**Exercise:** grayscale_v2 (Phase 1)
**Question that led here:** "Why does the concept of a block exist? Why can't we just use raw indices?"
