#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

// CPU reference attention forward: O = softmax(Q @ K^T / sqrt(d)) @ V.
//   Q, K, V: (B, H, N, d)   row-major, no mask, no dropout
//   O:       (B, H, N, d)
// Copies tensors back to host, computes softly, copies result to d_O.
inline void cpu_attention_reference(const float *d_Q, const float *d_K,
                                    const float *d_V, float *d_O,
                                    int B, int H, int N, int d) {
    size_t total = (size_t)B * H * N * d;
    float *hQ = (float *)std::malloc(total * sizeof(float));
    float *hK = (float *)std::malloc(total * sizeof(float));
    float *hV = (float *)std::malloc(total * sizeof(float));
    float *hO = (float *)std::malloc(total * sizeof(float));
    cudaMemcpy(hQ, d_Q, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hK, d_K, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hV, d_V, total * sizeof(float), cudaMemcpyDeviceToHost);

    float scale = 1.0f / std::sqrt((float)d);
    float *scores = (float *)std::malloc((size_t)N * sizeof(float));

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float *Qh = hQ + ((size_t)b * H + h) * N * d;
            const float *Kh = hK + ((size_t)b * H + h) * N * d;
            const float *Vh = hV + ((size_t)b * H + h) * N * d;
            float *Oh = hO + ((size_t)b * H + h) * N * d;
            for (int i = 0; i < N; ++i) {
                float max_s = -INFINITY;
                for (int j = 0; j < N; ++j) {
                    float s = 0.0f;
                    for (int k = 0; k < d; ++k) s += Qh[i * d + k] * Kh[j * d + k];
                    s *= scale;
                    scores[j] = s;
                    if (s > max_s) max_s = s;
                }
                double denom = 0.0;
                for (int j = 0; j < N; ++j) {
                    scores[j] = (float)std::exp((double)(scores[j] - max_s));
                    denom += scores[j];
                }
                for (int k = 0; k < d; ++k) {
                    double acc = 0.0;
                    for (int j = 0; j < N; ++j) acc += scores[j] * Vh[j * d + k];
                    Oh[i * d + k] = (float)(acc / denom);
                }
            }
        }
    }

    cudaMemcpy(d_O, hO, total * sizeof(float), cudaMemcpyHostToDevice);
    std::free(hQ); std::free(hK); std::free(hV); std::free(hO); std::free(scores);
}
