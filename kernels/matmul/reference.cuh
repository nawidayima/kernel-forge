#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "common/utils.cuh"

// C (M x N) = alpha * A (M x K) * B (K x N) + beta * C
// Arrays are row-major. cuBLAS is column-major, so we compute
// C^T = B^T * A^T, swapping shapes accordingly.
inline void cublas_sgemm(const float *A, const float *B, float *C,
                         int M, int N, int K,
                         float alpha = 1.0f, float beta = 0.0f) {
    static cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N);
}
