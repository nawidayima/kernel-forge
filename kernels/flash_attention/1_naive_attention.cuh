#pragma once

#include <cuda_runtime.h>

// Naive attention forward on device: materializes the full (N x N) scores
// matrix per (batch, head).
//   Q, K, V: (B, H, N, d) row-major, d <= 128
//   O:       (B, H, N, d)
//   scratch: (B, H, N, N)   // caller allocates
//
// Launch strategy is up to you; a common starting point is:
//   dim3 grid(N, H * B);     // one block per (query row, batch*head)
//   dim3 block(128);
//
// Principle to internalize: this is the "before" for FlashAttention. After
// profiling, look at HBM traffic — the N*N scratch write and re-read is the
// cost that the tiled version eliminates.
__global__ void attention_naive(const float *Q, const float *K, const float *V,
                                float *O, float *scratch,
                                int B, int H, int N, int d) {
    // TODO: Your implementation here.
    // 1. Compute scores[i, j] = dot(Q[i], K[j]) * (1/sqrt(d)) for one query row.
    // 2. Write scores to the (N x N) scratch slab.
    // 3. Row-softmax scratch (block-reduce max, then block-reduce exp-sum).
    // 4. O[i, :] = sum_j scratch[i, j] * V[j, :]
}
