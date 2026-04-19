#include <cstdio>
#include <cstdlib>

#include "common/runner.cuh"
#include "common/utils.cuh"
#include "kernels/matmul/reference.cuh"
#include "kernels/matmul/1_naive.cuh"
#include "kernels/matmul/2_tiled.cuh"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <kernel_num> [M K N]\n", argv[0]);
        printf("  0 = cuBLAS (timed; sets performance target)\n");
        printf("  1 = naive SGEMM\n");
        printf("  2 = tiled SGEMM\n");
        return 1;
    }
    int kernel = std::atoi(argv[1]);
    int M = argc > 2 ? std::atoi(argv[2]) : 4096;
    int K = argc > 3 ? std::atoi(argv[3]) : 4096;
    int N = argc > 4 ? std::atoi(argv[4]) : 4096;

    float *A = alloc_and_init((size_t)M * K, /*seed=*/1);
    float *B = alloc_and_init((size_t)K * N, /*seed=*/2);
    float *C = alloc_zero((size_t)M * N);
    float *C_ref = alloc_zero((size_t)M * N);

    cublas_sgemm(A, B, C_ref, M, N, K);

    float ms = 0.0f;
    auto launch = [&]() {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        switch (kernel) {
            case 0: cublas_sgemm(A, B, C, M, N, K); break;
            case 1: sgemm_naive<<<grid, block>>>(M, N, K, A, B, C); break;
            case 2: sgemm_tiled<32, 32, 32><<<grid, block>>>(M, N, K, A, B, C); break;
            default: printf("Unknown kernel %d\n", kernel); std::exit(1);
        }
    };
    ms = run_kernel(launch);

    verify(C, C_ref, (size_t)M * N, /*tol=*/1e-2f);
    report_performance(M, K, N, ms);

    double gflops = (2.0 * (double)M * K * N / (ms * 1e-3)) / 1e9;
    printf("RESULT exercise=matmul kernel=%d M=%d K=%d N=%d ms=%.6f gflops=%.3f\n",
           kernel, M, K, N, ms, gflops);

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(C_ref);
    return 0;
}
