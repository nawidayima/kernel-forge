# GPU Performance Engineering — Wiki

Concepts learned through hands-on exercises. One page per transferable principle.

| Page | One-liner | Learned from |
|------|-----------|--------------|
| [type-dispatch](type-dispatch.md) | CUDA kernels are typed; PyTorch tensors are polymorphic — the boundary needs explicit negotiation | vectoradd_v2 |
| [gpu-cpu-split-compilation](gpu-cpu-split-compilation.md) | A .cu file is both CPU and GPU code — nvcc splits it into two binaries | grayscale_v2 |
| [blocks-and-threads](blocks-and-threads.md) | Blocks exist because threads need a cooperating group that maps to physical SMs | grayscale_v2 |
| [gpu-cpu-async](gpu-cpu-async.md) | CPU pushes to a command queue and moves on; GPU executes in order; sync only when forced | grayscale_v2 |
| [warps-and-hardware-constraints](warps-and-hardware-constraints.md) | A warp is 32 lockstep threads — the real scheduling unit; block size must respect hardware limits | grayscale_v2 |
