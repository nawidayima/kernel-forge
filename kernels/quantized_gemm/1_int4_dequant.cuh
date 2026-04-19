#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// INT4 weight-only GEMM with fused dequantization.
//   A:      (M, K) float activations
//   W_q:    (K/2, N) packed int4 weights (low nibble = row 2k, high = 2k+1),
//           symmetric with zero-point = 8
//   scales: (N,) per-column FP32 scale
//   B:      (M, N) output
//
// Launch: standard tiled-GEMM grid (dim3 grid((N+BN-1)/BN, (M+BM-1)/BM)).
//
// Principle to internalize: weight-only quantization is bandwidth-driven.
// The 4x reduction in weight bytes moves the bottleneck from HBM to either
// dequant arithmetic or L2 reuse. Fuse the dequant into the GEMM inner
// loop — never materialize a dequantized weight tile. Compare against
// Marlin (arXiv:2408.11743) once this works.
template <int BM = 64, int BN = 64, int BK = 32>
__global__ void int4_gemm_dequant_fused(const float *A, const uint8_t *W_q,
                                        const float *scales, float *B,
                                        int M, int N, int K) {
    // TODO: Your implementation here.
    // 1. Declare shared tiles: A_s[BM][BK], W_s[BK][BN] (decoded to float).
    // 2. For each K-chunk:
    //      - Load A tile (coalesced).
    //      - Load and unpack W tile: each thread reads a byte, splits into
    //        two nibbles, subtracts 8, multiplies by scales[col].
    //      - __syncthreads(); accumulate; __syncthreads();
    // 3. Write output.
}
