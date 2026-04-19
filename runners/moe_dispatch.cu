#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "common/runner.cuh"
#include "common/utils.cuh"
#include "kernels/moe_dispatch/reference.cuh"
#include "kernels/moe_dispatch/1_gather_scatter.cuh"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <kernel_num> [T d d_out E]\n", argv[0]);
        printf("  1 = staged gather -> per-expert GEMM -> scatter\n");
        return 1;
    }
    int kernel = std::atoi(argv[1]);
    int T = argc > 2 ? std::atoi(argv[2]) : 4096;
    int d = argc > 3 ? std::atoi(argv[3]) : 1024;
    int d_out = argc > 4 ? std::atoi(argv[4]) : 1024;
    int E = argc > 5 ? std::atoi(argv[5]) : 8;

    float *X = alloc_and_init((size_t)T * d, /*seed=*/13);
    float *W = alloc_and_init((size_t)E * d * d_out, /*seed=*/17);
    float *Y = alloc_zero((size_t)T * d_out);
    float *Y_ref = alloc_zero((size_t)T * d_out);

    // Random top-1 routing
    int *h_expert = (int *)std::malloc(T * sizeof(int));
    std::mt19937 rng(21);
    std::uniform_int_distribution<int> edist(0, E - 1);
    for (int t = 0; t < T; ++t) h_expert[t] = edist(rng);
    int *expert_id = nullptr;
    CHECK_CUDA(cudaMalloc(&expert_id, T * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(expert_id, h_expert, T * sizeof(int), cudaMemcpyHostToDevice));
    std::free(h_expert);

    cpu_moe_reference(X, expert_id, W, Y_ref, T, d, d_out, E);

    int *counts = nullptr, *offsets = nullptr, *cursor = nullptr, *perm = nullptr;
    float *Y_packed = nullptr;
    CHECK_CUDA(cudaMalloc(&counts, E * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&offsets, E * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&cursor, E * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&perm, T * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&Y_packed, (size_t)T * d_out * sizeof(float)));

    auto launch = [&]() {
        switch (kernel) {
            case 1: {
                CHECK_CUDA(cudaMemsetAsync(counts, 0, E * sizeof(int)));
                CHECK_CUDA(cudaMemsetAsync(cursor, 0, E * sizeof(int)));
                moe_count_experts<<<(T + 255) / 256, 256>>>(expert_id, counts, T, E);
                // Exclusive prefix-sum counts -> offsets, on host for simplicity.
                std::vector<int> h_counts(E);
                std::vector<int> h_offsets(E);
                cudaMemcpy(h_counts.data(), counts, E * sizeof(int), cudaMemcpyDeviceToHost);
                int acc = 0;
                for (int e = 0; e < E; ++e) { h_offsets[e] = acc; acc += h_counts[e]; }
                cudaMemcpy(offsets, h_offsets.data(), E * sizeof(int), cudaMemcpyHostToDevice);

                moe_build_perm<<<(T + 255) / 256, 256>>>(expert_id, offsets, perm, cursor, T);

                for (int e = 0; e < E; ++e) {
                    int start = h_offsets[e];
                    int count = h_counts[e];
                    if (count == 0) continue;
                    const float *We = W + (size_t)e * d * d_out;
                    dim3 block(128);
                    dim3 grid(count);
                    moe_expert_gemm<<<grid, block>>>(X, perm, We, Y_packed, start, count, d, d_out);
                }

                {
                    dim3 block(128);
                    dim3 grid(T, (d_out + 127) / 128);
                    moe_unpermute<<<grid, block>>>(Y_packed, perm, Y, T, d_out);
                }
                break;
            }
            default: printf("Unknown kernel %d\n", kernel); std::exit(1);
        }
    };
    float ms = run_kernel(launch);

    verify(Y, Y_ref, (size_t)T * d_out, /*tol=*/1e-2f);
    double flops = 2.0 * (double)T * d * d_out;  // one d x d_out matmul per token
    double gflops = flops / (ms * 1.0e-3) / 1.0e9;
    printf("perf: %.3f ms/iter  %.1f GFLOP/s  (T=%d d=%d d_out=%d E=%d)\n",
           ms, gflops, T, d, d_out, E);
    printf("RESULT exercise=moe_dispatch kernel=%d T=%d d=%d d_out=%d E=%d ms=%.6f gflops=%.3f\n",
           kernel, T, d, d_out, E, ms, gflops);

    cudaFree(X); cudaFree(W); cudaFree(Y); cudaFree(Y_ref);
    cudaFree(expert_id); cudaFree(counts); cudaFree(offsets); cudaFree(cursor);
    cudaFree(perm); cudaFree(Y_packed);
    return 0;
}
