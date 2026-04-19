#pragma once

#include <cuda_runtime.h>

// CPU reference sum used to verify reduction kernels.
// Copies `N` floats from device, sums on host, returns the result.
inline float cpu_sum_reference(const float *d_input, int N) {
    float *h = (float *)std::malloc(N * sizeof(float));
    cudaMemcpy(h, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);
    double acc = 0.0;
    for (int i = 0; i < N; ++i) acc += h[i];
    std::free(h);
    return (float)acc;
}
