#pragma once

#include <cuda_runtime.h>

// Tiled SGEMM using shared memory.
//   C (M x N) = A (M x K) * B (K x N)
//
// Launch with BK=BM=BN=32 thread blocks:
//   dim3 block(32, 32);
//   dim3 grid((N + 31) / 32, (M + 31) / 32);
//
// Principle to internalize: global-memory traffic drops by a factor of BM (or BN)
// once each tile of A and B is cooperatively loaded into shared memory and reused
// by every thread in the block.
template <int BM = 32, int BN = 32, int BK = 32>
__global__ void sgemm_tiled(int M, int N, int K,
                            const float *A, const float *B, float *C) {
    // TODO: Your implementation here.
    // 1. Declare __shared__ float As[BM][BK], Bs[BK][BN];
    // 2. Walk k in chunks of BK:
    //      - Cooperatively load A tile and B tile into shared memory.
    //      - __syncthreads();
    //      - Accumulate partial dot product for this thread's output element.
    //      - __syncthreads();
    // 3. Write acc to C[row, col].
}
