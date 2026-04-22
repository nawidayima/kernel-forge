#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <unistd.h>

#include "common/utils.cuh"

constexpr int WARMUP_ITERS = 5;
constexpr int TIMED_ITERS = 20;

// ANSI color helpers — suppressed when stdout is not a tty (e.g. piped to a log).
inline const char *c_pass() { return isatty(1) ? "\033[1;32m" : ""; }
inline const char *c_fail() { return isatty(1) ? "\033[1;31m" : ""; }
inline const char *c_head() { return isatty(1) ? "\033[1;36m" : ""; }
inline const char *c_dim()  { return isatty(1) ? "\033[2m"    : ""; }
inline const char *c_off()  { return isatty(1) ? "\033[0m"    : ""; }

inline void print_banner(const char *exercise, int kernel) {
    printf("\n%s━━━ %s kernel=%d ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%s\n",
           c_head(), exercise, kernel, c_off());
}

inline void print_footer() {
    printf("%s━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%s\n\n",
           c_head(), c_off());
}

// Runs `launch` once to warm, then TIMED_ITERS times under cudaEvent timing.
// `launch` is any callable of signature `void()`. Returns average ms/iter.
template <typename Launch>
inline float run_kernel(Launch &&launch) {
    for (int i = 0; i < WARMUP_ITERS; ++i) launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < TIMED_ITERS; ++i) launch();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / TIMED_ITERS;
}

inline bool verify(const float *d_result, const float *d_reference, int N,
                   float tolerance = 1e-3f) {
    float *h_result = (float *)std::malloc(N * sizeof(float));
    float *h_reference = (float *)std::malloc(N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_reference, d_reference, N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int first_bad = -1;
    for (int i = 0; i < N; ++i) {
        float diff = std::fabs(h_result[i] - h_reference[i]);
        float denom = std::fabs(h_reference[i]) + 1e-8f;
        float rel = diff / denom;
        if (diff > max_abs_err) max_abs_err = diff;
        if (rel > max_rel_err) max_rel_err = rel;
        if (first_bad < 0 && rel > tolerance && diff > tolerance) first_bad = i;
    }

    bool ok = (first_bad < 0);
    printf("  %-10s  %s%s%s  (max_abs=%.2e  max_rel=%.2e  tol=%.0e)\n",
           "verify",
           ok ? c_pass() : c_fail(),
           ok ? "PASS" : "FAIL",
           c_off(),
           max_abs_err, max_rel_err, tolerance);
    if (!ok) {
        printf("  %sfirst mismatch%s @ i=%d: result=%.6f reference=%.6f\n",
               c_fail(), c_off(), first_bad, h_result[first_bad], h_reference[first_bad]);
    }

    std::free(h_result);
    std::free(h_reference);
    return ok;
}

// Reports performance for dense GEMM (M x K) * (K x N) = (M x N).
inline void report_performance(int M, int K, int N, float ms, const char *label = "perf") {
    double flops = 2.0 * (double)M * (double)K * (double)N;
    double gflops = flops / (ms * 1.0e-3) / 1.0e9;
    printf("  %-10s  %10.3f ms   %10.1f GFLOP/s\n", label, ms, gflops);
}

// Generic bandwidth report for memory-bound kernels.
inline void report_bandwidth(size_t bytes_moved, float ms, const char *label = "perf") {
    double gbps = (double)bytes_moved / (ms * 1.0e-3) / 1.0e9;
    printf("  %-10s  %10.3f ms   %10.1f GB/s\n", label, ms, gbps);
}
