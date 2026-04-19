#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdint>

// CPU reference for INT4 weight-only GEMM.
//   A:      (M, K) float activations
//   W_q:    (K/2, N) packed int4 weights (two 4-bit nibbles per byte,
//           low nibble = W[2*k0, n], high nibble = W[2*k0+1, n])
//   scales: (N,)    per-column FP32 scale
//   B:      (M, N)  output (activations * dequant(W))
//
// Dequant: w_fp = ((int8_t)nibble - 8) * scale  (symmetric, zero-point = 8)
inline void cpu_int4_gemm_reference(const float *d_A, const uint8_t *d_Wq,
                                    const float *d_scales, float *d_B,
                                    int M, int N, int K) {
    float *hA = (float *)std::malloc((size_t)M * K * sizeof(float));
    uint8_t *hW = (uint8_t *)std::malloc((size_t)(K / 2) * N * sizeof(uint8_t));
    float *hS = (float *)std::malloc((size_t)N * sizeof(float));
    float *hB = (float *)std::malloc((size_t)M * N * sizeof(float));
    cudaMemcpy(hA, d_A, (size_t)M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hW, d_Wq, (size_t)(K / 2) * N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(hS, d_scales, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; k += 2) {
                uint8_t packed = hW[(k / 2) * N + n];
                int lo = (int)(packed & 0xF) - 8;
                int hi = (int)((packed >> 4) & 0xF) - 8;
                float w0 = (float)lo * hS[n];
                float w1 = (float)hi * hS[n];
                acc += (double)hA[m * K + k] * w0;
                acc += (double)hA[m * K + k + 1] * w1;
            }
            hB[m * N + n] = (float)acc;
        }
    }

    cudaMemcpy(d_B, hB, (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice);
    std::free(hA); std::free(hW); std::free(hS); std::free(hB);
}
