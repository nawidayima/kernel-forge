#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>

#include "common/runner.cuh"
#include "common/utils.cuh"
#include "kernels/quantized_gemm/reference.cuh"
#include "kernels/quantized_gemm/1_int4_dequant.cuh"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <kernel_num> [M K N]\n", argv[0]);
        printf("  1 = int4 dequant fused with GEMM\n");
        return 1;
    }
    int kernel = std::atoi(argv[1]);
    int M = argc > 2 ? std::atoi(argv[2]) : 1024;
    int K = argc > 3 ? std::atoi(argv[3]) : 4096;
    int N = argc > 4 ? std::atoi(argv[4]) : 4096;
    if (K % 2 != 0) { printf("K must be even (K/2 packed nibbles per column)\n"); return 1; }

    float *A = alloc_and_init((size_t)M * K, /*seed=*/5);

    // Random packed int4 weights
    size_t w_bytes = (size_t)(K / 2) * N;
    uint8_t *h_W = (uint8_t *)std::malloc(w_bytes);
    std::mt19937 rng(9);
    for (size_t i = 0; i < w_bytes; ++i) h_W[i] = (uint8_t)(rng() & 0xFF);
    uint8_t *W_q = nullptr;
    CHECK_CUDA(cudaMalloc(&W_q, w_bytes));
    CHECK_CUDA(cudaMemcpy(W_q, h_W, w_bytes, cudaMemcpyHostToDevice));
    std::free(h_W);

    // Per-column scales
    float *h_S = (float *)std::malloc(N * sizeof(float));
    std::uniform_real_distribution<float> sdist(0.005f, 0.05f);
    for (int i = 0; i < N; ++i) h_S[i] = sdist(rng);
    float *scales = nullptr;
    CHECK_CUDA(cudaMalloc(&scales, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(scales, h_S, N * sizeof(float), cudaMemcpyHostToDevice));
    std::free(h_S);

    float *B = alloc_zero((size_t)M * N);
    float *B_ref = alloc_zero((size_t)M * N);
    cpu_int4_gemm_reference(A, W_q, scales, B_ref, M, N, K);

    auto launch = [&]() {
        switch (kernel) {
            case 1: {
                constexpr int BM = 64, BN = 64, BK = 32;
                dim3 block(BN, BM);
                dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
                int4_gemm_dequant_fused<BM, BN, BK><<<grid, block>>>(A, W_q, scales, B, M, N, K);
                break;
            }
            default: printf("Unknown kernel %d\n", kernel); std::exit(1);
        }
    };
    float ms = run_kernel(launch);

    verify(B, B_ref, (size_t)M * N, /*tol=*/1e-2f);
    report_performance(M, K, N, ms);

    // Weight bandwidth: int4 weights move 0.5 B per element.
    size_t wbytes = (size_t)M * N * K / 2;  // per iter, assuming no reuse
    printf("  (weight bytes moved assuming no reuse: %.2f GB)\n", wbytes / 1.0e9);

    double gflops = (2.0 * (double)M * K * N / (ms * 1e-3)) / 1e9;
    printf("RESULT exercise=quantized_gemm kernel=%d M=%d K=%d N=%d ms=%.6f gflops=%.3f\n",
           kernel, M, K, N, ms, gflops);

    cudaFree(A); cudaFree(W_q); cudaFree(scales); cudaFree(B); cudaFree(B_ref);
    return 0;
}
