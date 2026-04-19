#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "common/runner.cuh"
#include "common/utils.cuh"
#include "kernels/reduction/reference.cuh"
#include "kernels/reduction/1_divergent.cuh"
#include "kernels/reduction/2_coalesced.cuh"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <kernel_num> [N]\n", argv[0]);
        printf("  1 = divergent tree, 2 = coalesced + warp shuffle\n");
        return 1;
    }
    int kernel = std::atoi(argv[1]);
    int N = argc > 2 ? std::atoi(argv[2]) : (1 << 24);  // 16M floats

    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;

    float *input = alloc_and_init((size_t)N);
    float *partials = alloc_zero((size_t)num_blocks);
    float *d_final = alloc_zero(1);

    float ref = cpu_sum_reference(input, N);

    auto launch = [&]() {
        size_t shmem = block_size * sizeof(float);
        switch (kernel) {
            case 1:
                reduce_divergent<<<num_blocks, block_size, shmem>>>(input, partials, N);
                reduce_divergent<<<1, block_size, shmem>>>(partials, d_final, num_blocks);
                break;
            case 2:
                reduce_coalesced<<<num_blocks, block_size, shmem>>>(input, partials, N);
                reduce_coalesced<<<1, block_size, shmem>>>(partials, d_final, num_blocks);
                break;
            default: printf("Unknown kernel %d\n", kernel); std::exit(1);
        }
    };
    float ms = run_kernel(launch);

    float h_final = 0.0f;
    CHECK_CUDA(cudaMemcpy(&h_final, d_final, sizeof(float), cudaMemcpyDeviceToHost));
    float abs_err = std::fabs(h_final - ref);
    float rel_err = abs_err / (std::fabs(ref) + 1e-8f);
    printf("verify: %s  result=%.6f ref=%.6f  rel_err=%.3e\n",
           rel_err < 1e-3f ? "PASS" : "FAIL", h_final, ref, rel_err);

    size_t bytes = (size_t)N * sizeof(float);
    report_bandwidth(bytes, ms, "read");

    double gbps = (double)bytes / (ms * 1e-3) / 1e9;
    printf("RESULT exercise=reduction kernel=%d N=%d ms=%.6f gbps=%.3f\n",
           kernel, N, ms, gbps);

    cudaFree(input); cudaFree(partials); cudaFree(d_final);
    return 0;
}
