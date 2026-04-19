#pragma once

#include <cuda_runtime.h>

// Coalesced / warp-primitive tree reduction.
// Same launch shape as reduce_divergent.
//
// Principle to internalize: once the active region fits in a single warp
// (stride <= 32), use __shfl_down_sync to avoid shared-memory traffic and
// __syncthreads entirely. Active lanes should be contiguous so the warp
// does useful work on every cycle.
__global__ void reduce_coalesced(const float *input, float *partial_sums, int N) {
    // TODO: Your implementation here.
    // Approach:
    //   1. Each thread sums two elements on load (grid-stride or paired load).
    //   2. Shared-memory tree with `if (tid < stride) s[tid] += s[tid + stride]`
    //      — contiguous active lanes, no divergence.
    //   3. Final warp reduction with __shfl_down_sync(0xffffffff, v, offset).
    //   4. Lane 0 writes to partial_sums[blockIdx.x].
}
