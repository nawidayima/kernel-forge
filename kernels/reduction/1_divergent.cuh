#pragma once

#include <cuda_runtime.h>

// Divergent tree reduction (PMPP ch. 10, first version).
// Each block reduces BLOCK_SIZE elements; grid reduces N = blocks * BLOCK_SIZE
// into `partial_sums`. A second launch reduces `partial_sums` into a scalar.
//
// Launch:
//   int block_size = 256;
//   int num_blocks = (N + block_size - 1) / block_size;
//   reduce_divergent<<<num_blocks, block_size, block_size*sizeof(float)>>>(...)
//
// Principle to internalize: the classic `stride *= 2` pattern with
// `if (tid % (2*stride) == 0)` causes warp divergence — active lanes are
// spread across the warp, so SIMT execution runs all lanes and masks off
// most of them. Measure occupancy/IPC before optimizing.
__global__ void reduce_divergent(const float *input, float *partial_sums, int N) {
    // TODO: Your implementation here.
    // 1. extern __shared__ float s[];
    // 2. Load one element per thread into s[tid].
    // 3. for (stride = 1; stride < blockDim.x; stride *= 2):
    //      __syncthreads();
    //      if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
    // 4. if (tid == 0) partial_sums[blockIdx.x] = s[0];
}
