#pragma once

#include <cuda_runtime.h>

// Naive SGEMM: each thread computes one element of C.
//   C (M x N) = A (M x K) * B (K x N)
// All arrays row-major.
//
// Launch:
//   dim3 block(32, 32);
//   dim3 grid((N + 31) / 32, (M + 31) / 32);
//
// Principle to internalize: this kernel is memory-bound. Count how many
// times each A[row, k] is loaded across all threads that need it.
__global__ void sgemm_naive(int M, int N, int K,
                            const float *A, const float *B, float *C) {
    // TODO: Your implementation here.
    // 1. row = blockIdx.y * blockDim.y + threadIdx.y
    //    col = blockIdx.x * blockDim.x + threadIdx.x
    // 2. Bounds check: if (row < M && col < N) ...
    // 3. Accumulate sum_k A[row * K + k] * B[k * N + col], write to C[row * N + col].
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[N * k + col];
        }
        C[row * N + col] = sum;
    }
}
