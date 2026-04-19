#pragma once

#include <cuda_runtime.h>

// Fused cross-entropy: single pass per row using the online softmax trick.
//   logits: (B, V)
//   targets: (B,)
//   per_row_loss: (B,) — output
//
// Principle to internalize: replacing the naive max + sum passes with one
// online softmax pass cuts global-memory traffic roughly in half. This is
// the same running-max, running-denom recurrence used in FlashAttention.
// Reach for Liger-Kernel's FLCE (arXiv:2410.10989) after this works — it
// extends the same idea to the full logit projection.
__global__ void cross_entropy_fused(const float *logits, const int *targets,
                                    float *per_row_loss, int B, int V) {
    // TODO: Your implementation here.
    // Online softmax recurrence per thread, then block reduction:
    //   m_new = max(m_old, x)
    //   d_new = d_old * exp(m_old - m_new) + exp(x - m_new)
    // Combine across threads with the same recurrence (pairwise).
    // loss[b] = log(d) + m - logit[target[b]]
}
