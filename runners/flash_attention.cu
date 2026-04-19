#include <cstdio>
#include <cstdlib>

#include "common/runner.cuh"
#include "common/utils.cuh"
#include "kernels/flash_attention/reference.cuh"
#include "kernels/flash_attention/1_naive_attention.cuh"
#include "kernels/flash_attention/2_flash_forward.cuh"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <kernel_num> [B H N d]\n", argv[0]);
        printf("  1 = naive (materialized scores), 2 = FlashAttention-2 forward\n");
        return 1;
    }
    int kernel = std::atoi(argv[1]);
    int B = argc > 2 ? std::atoi(argv[2]) : 2;
    int H = argc > 3 ? std::atoi(argv[3]) : 8;
    int N = argc > 4 ? std::atoi(argv[4]) : 1024;
    int d = argc > 5 ? std::atoi(argv[5]) : 64;

    size_t tensor_elems = (size_t)B * H * N * d;
    float *Q = alloc_and_init(tensor_elems, /*seed=*/11);
    float *K = alloc_and_init(tensor_elems, /*seed=*/22);
    float *V = alloc_and_init(tensor_elems, /*seed=*/33);
    float *O = alloc_zero(tensor_elems);
    float *O_ref = alloc_zero(tensor_elems);
    float *scratch = alloc_zero((size_t)B * H * N * N);

    cpu_attention_reference(Q, K, V, O_ref, B, H, N, d);

    auto launch = [&]() {
        switch (kernel) {
            case 1: {
                dim3 grid(N, H * B);
                dim3 block(128);
                attention_naive<<<grid, block>>>(Q, K, V, O, scratch, B, H, N, d);
                break;
            }
            case 2: {
                constexpr int BR = 64, BC = 64;
                dim3 grid((N + BR - 1) / BR, H * B);
                dim3 block(BR);
                flash_attention_forward<BR, BC><<<grid, block>>>(Q, K, V, O, B, H, N, d);
                break;
            }
            default: printf("Unknown kernel %d\n", kernel); std::exit(1);
        }
    };
    float ms = run_kernel(launch);

    verify(O, O_ref, tensor_elems, /*tol=*/1e-2f);

    // FLOPs: 2 * B * H * (2 * N * N * d) = 4 B H N^2 d  (Q@K^T + P@V)
    double flops = 4.0 * B * H * (double)N * (double)N * d;
    double gflops = flops / (ms * 1.0e-3) / 1.0e9;
    printf("perf: %.3f ms/iter  %.1f GFLOP/s  (B=%d H=%d N=%d d=%d)\n",
           ms, gflops, B, H, N, d);
    printf("RESULT exercise=flash_attention kernel=%d B=%d H=%d N=%d d=%d ms=%.6f gflops=%.3f\n",
           kernel, B, H, N, d, ms, gflops);

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); cudaFree(O_ref); cudaFree(scratch);
    return 0;
}
