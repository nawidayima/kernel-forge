#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

#include "common/runner.cuh"
#include "common/utils.cuh"
#include "kernels/cross_entropy/reference.cuh"
#include "kernels/cross_entropy/1_naive.cuh"
#include "kernels/cross_entropy/2_fused.cuh"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <kernel_num> [B V]\n", argv[0]);
        printf("  1 = naive (max + sum + gather), 2 = fused online softmax\n");
        return 1;
    }
    int kernel = std::atoi(argv[1]);
    int B = argc > 2 ? std::atoi(argv[2]) : 4096;
    int V = argc > 3 ? std::atoi(argv[3]) : 32000;

    float *logits = alloc_and_init((size_t)B * V);

    // Random int targets in [0, V)
    int *h_targets = (int *)std::malloc(B * sizeof(int));
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dist(0, V - 1);
    for (int b = 0; b < B; ++b) h_targets[b] = dist(rng);
    int *d_targets = nullptr;
    CHECK_CUDA(cudaMalloc(&d_targets, B * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_targets, h_targets, B * sizeof(int), cudaMemcpyHostToDevice));

    float *per_row = alloc_zero((size_t)B);
    float ref_loss = cpu_ce_reference(logits, d_targets, B, V);

    int block_size = 256;
    size_t shmem = 2 * block_size * sizeof(float);
    auto launch = [&]() {
        switch (kernel) {
            case 1: cross_entropy_naive<<<B, block_size, shmem>>>(logits, d_targets, per_row, B, V); break;
            case 2: cross_entropy_fused<<<B, block_size, shmem>>>(logits, d_targets, per_row, B, V); break;
            default: printf("Unknown kernel %d\n", kernel); std::exit(1);
        }
    };
    float ms = run_kernel(launch);

    // Mean loss on host for verification.
    float *h_per_row = (float *)std::malloc(B * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_per_row, per_row, B * sizeof(float), cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int b = 0; b < B; ++b) sum += h_per_row[b];
    float mean = (float)(sum / B);
    float rel_err = std::fabs(mean - ref_loss) / (std::fabs(ref_loss) + 1e-8f);
    printf("verify: %s  mean_loss=%.6f ref=%.6f  rel_err=%.3e\n",
           rel_err < 1e-3f ? "PASS" : "FAIL", mean, ref_loss, rel_err);

    size_t bytes = (size_t)B * V * sizeof(float);
    report_bandwidth(bytes, ms, "logits_read");

    double gbps = (double)bytes / (ms * 1e-3) / 1e9;
    printf("RESULT exercise=cross_entropy kernel=%d B=%d V=%d ms=%.6f gbps=%.3f\n",
           kernel, B, V, ms, gbps);

    std::free(h_targets); std::free(h_per_row);
    cudaFree(logits); cudaFree(d_targets); cudaFree(per_row);
    return 0;
}
