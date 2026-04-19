# GPU-CPU Async: The Command Queue Model

## The concept

The GPU has a command queue (CUDA calls it a "stream"). The CPU pushes work onto it and moves on immediately. The GPU pops and executes in order.

```
CPU:  [push kernel A] [push kernel B] [push memcpy] [do other work...]
GPU:                   [kernel A.....] [kernel B.....] [memcpy...]
```

The CPU never blocks unless you explicitly force it.

## When does the CPU wait?

- `cudaDeviceSynchronize()` — block until GPU finishes everything
- `cudaMemcpy()` (default variant) — implicitly waits because CPU needs the data
- PyTorch's `.cpu()` or `print(tensor)` — triggers sync because CPU needs to read GPU memory

## What cudaGetLastError() actually checks

It does NOT wait for the kernel to finish. It only checks if the launch command itself was valid (e.g., did you request too many threads per block). If the kernel crashes or produces wrong output, you won't know at that point.

## Why async matters

Blocking wastes hardware:
```
Blocking:  CPU: [launch]...[wait]...[launch]...[wait]
           GPU: ...[kernel]...........[kernel]........

Async:     CPU: [launch][launch][launch][other work]
           GPU: ........[kernel 1][kernel 2][kernel 3]
```

The GPU stays fed. Becomes critical when chaining kernels or overlapping compute with memory transfers (Phase 5+).

## Context

**Exercise:** grayscale_v2 (Phase 1)
**Question that led here:** "How is the async between GPU and CPU managed?"
