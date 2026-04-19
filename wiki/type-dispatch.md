# Type Dispatch: Framework ↔ Kernel Boundary

## The concept

A CUDA kernel operates on raw memory — `float*`, `__half*`, `double*`. It has no idea what a PyTorch tensor is. The C++ wrapper function is the boundary where PyTorch's type-flexible tensors get converted into typed pointers via `data_ptr<T>()`.

If you hardcode `float*`, you can only handle float32. The test harness (or any real workload) may send float16, bfloat16, or double.

## The mechanism

`AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor.scalar_type(), "name", ([&] { ... }))` — a PyTorch macro that:

1. Inspects the tensor's dtype at runtime
2. Instantiates a C++ template with the matching `scalar_t` type
3. Calls your templated kernel with correctly-typed pointers

Your kernel becomes `template <typename scalar_t>` and uses `scalar_t*` instead of `float*`.

## Transferable principle

The boundary between a high-level framework and low-level code always requires explicit type negotiation. This isn't unique to CUDA — it's the same issue as C FFI, numpy ctypes, WebAssembly memory views. The GPU just makes it fatal (runtime crash) instead of silently wrong.

## Context

**Exercise:** vectoradd_v2 (Phase 1)
**Error:** `RuntimeError: expected scalar type Float but found Half`
**Root cause:** Kernel used `const float*` pointers; test harness sent float16 tensors.
