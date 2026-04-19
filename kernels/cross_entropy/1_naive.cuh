#pragma once

#include <cuda_runtime.h>

// Naive cross-entropy: three passes over logits per row, with a materialized
// softmax intermediate.
//   logits: (B, V)
//   targets: (B,)
//   per_row_loss: (B,) — output
//
// Launch one block per row (B blocks), BLOCK_SIZE threads, 2*BLOCK_SIZE floats shmem.
//
// Principle to internalize: this is the "before" for kernel fusion. After
// writing and profiling it, compare memory traffic against 2_fused.cuh.
__global__ void cross_entropy_naive(const float *logits, const int *targets,
                                    float *per_row_loss, int B, int V) {
    // TODO: Your implementation here.
    // For each row b:
    //   1. Find max (parallel reduction).
    //   2. Compute sum of exp(logit - max) (parallel reduction).
    //   3. loss[b] = log(sum) + max - logit[target[b]]
}
