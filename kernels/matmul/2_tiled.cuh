#pragma once

#include <cuda_runtime.h>
#include <cassert>

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
    assert (BK == BM);
    assert (BK == BN);
    assert (BK == blockDim.x);
    assert (BK == blockDim.y);
    int by = blockIdx.y;                            
    int bx = blockIdx.x;                            
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int col = bx*blockDim.x + tx;
    int row = by*blockDim.y + ty;


    __shared__ float Atile[BK*BK];
    __shared__ float Btile[BK*BK];
    float sum = 0.0;
    for (int phase = 0; phase < (K + BK - 1)/BK; ++phase){ 

        int A_load_col = phase*BK + tx;
        int B_load_row = phase*BK + ty;

        if (row < M && A_load_col < K){
            Atile[ty*BK+tx] = A[row*K + A_load_col];
        }
        else {
            Atile[ty*BK+tx] = 0.0;
        }
        
        if (col < N && B_load_row < K){
            Btile[ty*BK+tx] = B[B_load_row*N + col];
        }
        else {
            Btile[ty*BK+tx] = 0.0;
        }
        __syncthreads();

        if (row < M && col < N){
            for (int k = 0; k < BK; ++k){
                sum += Atile[ty*BK + k] * Btile[k*BK + tx];
            }
        }
        __syncthreads();
    }
    if (row < M && col < N){
        C[row*N + col] = sum;
    }
}
