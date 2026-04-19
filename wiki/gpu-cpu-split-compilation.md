# Split Compilation: One File, Two Processors

## The concept

A `.cu` file contains code for both the CPU and GPU. The CUDA compiler (`nvcc`) does two passes — it extracts `__global__` and `__device__` functions and compiles them for the GPU instruction set, then sends everything else to the host C++ compiler (clang/gcc) for the CPU.

The `<<<blocks, threads>>>` launch syntax is the bridge. `nvcc` replaces it with runtime API calls that queue work on the GPU. It looks like a function call but it's really "send this work to a different processor and continue."

## The qualifiers

- `__global__` — callable from CPU, runs on GPU (your kernel)
- `__device__` — callable from GPU, runs on GPU (helper functions)
- `__host__` — callable from CPU, runs on CPU (default, can be omitted)

## Other C++ syntax in kernels

- `void` — kernel returns nothing. Threads write results to memory via pointers instead.
- `const` — promise to the compiler that you won't modify data behind this pointer. Enables optimization.
- `__restrict__` — stronger promise that pointers don't alias (point to overlapping memory).

## Context

**Exercise:** grayscale_v2 (Phase 1)
**Question that led here:** "Is the .cu file CPU code or GPU code? It seems to be a confusing mix of the two."
**Key insight:** The confusion is correct — it IS both. The compiler splits it. Understanding this removes the magic.
