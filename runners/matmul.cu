#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/runner.cuh"
#include "common/utils.cuh"
#include "kernels/matmul/reference.cuh"
#include "kernels/matmul/1_naive.cuh"
#include "kernels/matmul/2_tiled.cuh"

struct MatmulShape {
    int M;
    int K;
    int N;
};

inline void launch_matmul_kernel(int kernel, int M, int K, int N,
                                 const float *A, const float *B, float *C) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    switch (kernel) {
        case 1: sgemm_naive<<<grid, block>>>(M, N, K, A, B, C); break;
        case 2: sgemm_tiled<32, 32, 32><<<grid, block>>>(M, N, K, A, B, C); break;
        default: printf("Unknown kernel %d\n", kernel); std::exit(1);
    }
}

bool run_matmul_test_case(int kernel, MatmulShape shape) {
    int M = shape.M;
    int K = shape.K;
    int N = shape.N;

    float *A = alloc_and_init((size_t)M * K, /*seed=*/1);
    float *B = alloc_and_init((size_t)K * N, /*seed=*/2);
    float *C = alloc_zero((size_t)M * N);
    float *C_ref = alloc_zero((size_t)M * N);

    printf("test kernel=%d M=%d K=%d N=%d\n", kernel, M, K, N);
    fflush(stdout);

    cublas_sgemm(A, B, C_ref, M, N, K);
    launch_matmul_kernel(kernel, M, K, N, A, B, C);
    CHECK_CUDA(cudaDeviceSynchronize());

    bool ok = verify(C, C_ref, (int)((size_t)M * N), /*tol=*/1e-2f);

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(C_ref);
    return ok;
}

int run_matmul_tests(int kernel) {
    MatmulShape cases[] = {
        {32, 32, 32},
        {64, 64, 64},
        {33, 32, 32},
        {32, 32, 33},
        {32, 33, 32},
        {33, 33, 33},
        {31, 31, 31},
        {127, 65, 97},
        {257, 128, 128},
        {128, 257, 128},
        {128, 128, 257},
        {257, 129, 193},
    };

    int passed = 0;
    int total = (int)(sizeof(cases) / sizeof(cases[0]));
    for (int i = 0; i < total; ++i) {
        if (run_matmul_test_case(kernel, cases[i])) ++passed;
    }

    printf("summary %d/%d PASS\n", passed, total);
    return passed == total ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <kernel_num> [M K N]\n", argv[0]);
        printf("       %s test <kernel_num>\n", argv[0]);
        printf("  0 = cuBLAS (timed; sets performance target)\n");
        printf("  1 = naive SGEMM\n");
        printf("  2 = tiled SGEMM\n");
        return 1;
    }

    if (std::strcmp(argv[1], "test") == 0) {
        if (argc < 3) {
            printf("Usage: %s test <kernel_num>\n", argv[0]);
            return 1;
        }
        int kernel = std::atoi(argv[2]);
        if (kernel == 0) {
            printf("test mode expects a custom kernel: 1 = naive, 2 = tiled\n");
            return 1;
        }
        return run_matmul_tests(kernel);
    }

    int kernel = std::atoi(argv[1]);
    int M = argc > 2 ? std::atoi(argv[2]) : 4096;
    int K = argc > 3 ? std::atoi(argv[3]) : 4096;
    int N = argc > 4 ? std::atoi(argv[4]) : 4096;

    float *A = alloc_and_init((size_t)M * K, /*seed=*/1);
    float *B = alloc_and_init((size_t)K * N, /*seed=*/2);
    float *C = alloc_zero((size_t)M * N);
    float *C_ref = alloc_zero((size_t)M * N);

    // cuBLAS runs unconditionally — it's both the reference result for
    // verification and the performance target for comparison.
    auto cublas_launch = [&]() { cublas_sgemm(A, B, C_ref, M, N, K); };
    float cublas_ms = run_kernel(cublas_launch);

    print_banner("matmul", kernel);
    printf("  %-10s  M=%d  K=%d  N=%d\n", "shape", M, K, N);

    float ms = cublas_ms;
    if (kernel != 0) {
        auto launch = [&]() {
            launch_matmul_kernel(kernel, M, K, N, A, B, C);
        };
        ms = run_kernel(launch);
        verify(C, C_ref, (int)((size_t)M * N), /*tol=*/1e-2f);
    }

    report_performance(M, K, N, ms, "perf");
    if (kernel != 0) {
        report_performance(M, K, N, cublas_ms, "cuBLAS");
        double ratio = cublas_ms / ms;  // fraction of cuBLAS throughput (1.0 = parity)
        printf("  %-10s  %.3f  (%.1f%% of cuBLAS)\n", "ratio", ratio, ratio * 100.0);
    }
    print_footer();

    double flops = 2.0 * (double)M * (double)K * (double)N;
    double gflops = flops / (ms * 1e-3) / 1e9;
    double cublas_gflops = flops / (cublas_ms * 1e-3) / 1e9;
    printf("RESULT exercise=matmul kernel=%d M=%d K=%d N=%d ms=%.6f gflops=%.3f cublas_ms=%.6f cublas_gflops=%.3f\n",
           kernel, M, K, N, ms, gflops, cublas_ms, cublas_gflops);

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(C_ref);
    return 0;
}
