#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

inline void init_random(float *host_data, int N, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) host_data[i] = dist(rng);
}

inline float *alloc_and_init(int N, unsigned seed = 42) {
    float *d_ptr = nullptr;
    float *h_ptr = (float *)std::malloc(N * sizeof(float));
    init_random(h_ptr, N, seed);
    CHECK_CUDA(cudaMalloc(&d_ptr, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, N * sizeof(float), cudaMemcpyHostToDevice));
    std::free(h_ptr);
    return d_ptr;
}

inline float *alloc_zero(int N) {
    float *d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_ptr, 0, N * sizeof(float)));
    return d_ptr;
}
