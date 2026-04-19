#pragma once

#include <cuda_runtime.h>
#include <cstdlib>

// CPU reference for MoE dispatch + expert GEMM + combine, top-k=1.
//   X:         (T, d)  token activations
//   expert_id: (T,)    int, in [0, E)
//   W:         (E, d, d_out)  per-expert weight matrices, row-major per expert
//   Y:         (T, d_out)     output: Y[t] = X[t] @ W[expert_id[t]]
inline void cpu_moe_reference(const float *d_X, const int *d_expert_id,
                              const float *d_W, float *d_Y,
                              int T, int d, int d_out, int E) {
    float *hX = (float *)std::malloc((size_t)T * d * sizeof(float));
    int *hE = (int *)std::malloc((size_t)T * sizeof(int));
    float *hW = (float *)std::malloc((size_t)E * d * d_out * sizeof(float));
    float *hY = (float *)std::malloc((size_t)T * d_out * sizeof(float));
    cudaMemcpy(hX, d_X, (size_t)T * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hE, d_expert_id, (size_t)T * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hW, d_W, (size_t)E * d * d_out * sizeof(float), cudaMemcpyDeviceToHost);

    for (int t = 0; t < T; ++t) {
        int e = hE[t];
        const float *We = hW + (size_t)e * d * d_out;
        for (int o = 0; o < d_out; ++o) {
            double acc = 0.0;
            for (int k = 0; k < d; ++k) acc += (double)hX[t * d + k] * We[k * d_out + o];
            hY[t * d_out + o] = (float)acc;
        }
    }

    cudaMemcpy(d_Y, hY, (size_t)T * d_out * sizeof(float), cudaMemcpyHostToDevice);
    std::free(hX); std::free(hE); std::free(hW); std::free(hY);
}
