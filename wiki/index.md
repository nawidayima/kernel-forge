# GPU Performance Engineering — Wiki

Concepts learned through hands-on exercises. One page per transferable principle.

| Page | One-liner | Learned from |
|------|-----------|--------------|
| [type-dispatch](type-dispatch.md) | CUDA kernels are typed; PyTorch tensors are polymorphic — the boundary needs explicit negotiation | vectoradd_v2 |
| [gpu-cpu-split-compilation](gpu-cpu-split-compilation.md) | A .cu file is both CPU and GPU code — nvcc splits it into two binaries | grayscale_v2 |
| [blocks-and-threads](blocks-and-threads.md) | Blocks exist because threads need a cooperating group that maps to physical SMs | grayscale_v2 |
| [gpu-cpu-async](gpu-cpu-async.md) | CPU pushes to a command queue and moves on; GPU executes in order; sync only when forced | grayscale_v2 |
| [warps-and-hardware-constraints](warps-and-hardware-constraints.md) | A warp is 32 lockstep threads — the real scheduling unit; block size must respect hardware limits | grayscale_v2 |
| [memory-coalescing](memory-coalescing.md) | Warps form along threadIdx.x; which matrix index that drives decides whether a load is 1 transaction or 32 | matmul/1_naive |
| [arithmetic-intensity](arithmetic-intensity.md) | FLOPs per byte is the master variable — every optimization after coalescing climbs this ratio | _pending_ |
| [shared-memory-tiling](shared-memory-tiling.md) | Stage a BM×BK×BN tile into SMEM once per block; each loaded element reused across every thread in the block | _pending_ |
| [register-tiling](register-tiling.md) | Each thread owns a TM×TN rectangle; outer-product inner loop turns few SMEM loads into many FMAs | _pending_ |
| [warp-tiling](warp-tiling.md) | The middle tier — warps are what the hardware schedules, so the tile hierarchy needs a warp-shaped layer | _pending_ |
| [occupancy](occupancy.md) | Three resources cap resident warps per SM; but more occupancy isn't always better | _pending_ |
| [vectorized-loads](vectorized-loads.md) | `float4` + `reinterpret_cast` compress the instruction stream without changing the memory pattern | _pending_ |
