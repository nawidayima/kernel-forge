#pragma once

#include <cuda_runtime.h>

// FlashAttention-2 style forward pass (arXiv:2307.08691).
//   Q, K, V: (B, H, N, d) row-major, d <= 128
//   O:       (B, H, N, d)
// No scratch: the online softmax recurrence keeps per-row running state in
// registers and streams K, V tiles through shared memory.
//
// Recommended starting shape: Br = 64 (query rows per block), Bc = 64 (KV
// rows per tile). Head dim d fits in shared memory alongside one tile.
//
// Launch:
//   dim3 grid(ceil_div(N, Br), H * B);
//   dim3 block(Br);              // or (Br * Br) if you parallelize within
//
// Principle to internalize: this is an online algorithm. The per-row state
// is (m_i, l_i, O_i) — running max, running denominator, running output.
// For each KV tile, update all three with the two-line recurrence from the
// paper. The entire attention is computed without ever materializing the
// (N x N) scores matrix.
template <int BR = 64, int BC = 64>
__global__ void flash_attention_forward(const float *Q, const float *K,
                                        const float *V, float *O,
                                        int B, int H, int N, int d) {
    // TODO: Your implementation here.
    // Outer loop: walk K/V tiles of size (BC, d).
    // Inner: block-parallel over Br query rows.
    // Per query row, maintain (m_i, l_i) and output accumulator O_i of size d.
    // Update rule per tile j:
    //   S_ij = Q_i @ K_j^T * (1/sqrt(d))           // (Br, Bc)
    //   m_ij = rowmax(S_ij)
    //   P_ij = exp(S_ij - m_ij)
    //   l_ij = rowsum(P_ij)
    //   m_new = max(m_i, m_ij)
    //   alpha = exp(m_i - m_new); beta = exp(m_ij - m_new)
    //   l_new = alpha * l_i + beta * l_ij
    //   O_new = alpha * O_i + beta * (P_ij @ V_j)
    //   (m_i, l_i, O_i) = (m_new, l_new, O_new)
    // Final: O_i /= l_i and write out.
}
