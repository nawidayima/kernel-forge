#pragma once

#include <cuda_runtime.h>

// Gather-compute-scatter MoE dispatch, top-k=1, single device.
//   X:         (T, d)  token activations
//   expert_id: (T,)    in [0, E)
//   W:         (E, d, d_out)  per-expert weights
//   Y:         (T, d_out)     output
//
// A common three-phase structure:
//   (1) Permutation build:
//       - Count tokens per expert.
//       - Prefix-sum to get expert offsets in the permuted buffer.
//       - Scatter token indices into `perm` (T,) grouped by expert.
//   (2) Per-expert GEMM:
//       - For each expert e, run a dense GEMM of the gathered token tile
//         ((count_e, d) @ (d, d_out)).
//   (3) Unpermute / scatter outputs back to their original token rows.
//
// This stub launches (3) phases via separate kernels. You may also fuse the
// outer loop into a single grouped-GEMM launch (see MegaBlocks /
// arXiv:2211.15841). Keep it staged for the first pass.
//
// Principle to internalize: the data-dependent routing breaks the
// regular-parallelism assumption. The cost is gather overhead, load
// imbalance across experts, and per-expert kernel-launch latency. Fused
// dispatch kernels (FusedXpert, AdaFuse) fix these.

// Phase 1a — count tokens per expert.
__global__ void moe_count_experts(const int *expert_id, int *counts,
                                  int T, int E) {
    // TODO: atomicAdd one per token into counts[expert_id[t]].
}

// Phase 1b — build the permutation given prefix-summed offsets.
// `offsets`:  (E,)   start index into `perm` for each expert (exclusive PSUM of counts)
// `perm`:     (T,)   output: perm[pos] = original token index t
// `cursor`:   (E,)   scratch, initialized to 0 on host or in a prior kernel
__global__ void moe_build_perm(const int *expert_id, const int *offsets,
                               int *perm, int *cursor, int T) {
    // TODO: for each t, pos = offsets[expert_id[t]] + atomicAdd(&cursor[expert_id[t]], 1)
    //       perm[pos] = t
}

// Phase 2 — per-expert GEMM over a packed token slice.
// Called E times from host (or launched once with one block per expert).
// X_packed: (count_e, d) view built by gathering rows X[perm[offset:offset+count_e]]
// You can either materialize X_packed or gather on-the-fly via perm inside the kernel.
__global__ void moe_expert_gemm(const float *X, const int *perm,
                                const float *W_e, float *Y_packed,
                                int start, int count, int d, int d_out) {
    // TODO: one block per output row of this expert's slice.
    //       token_t = perm[start + local_row]
    //       Y_packed[local_row * d_out + o] = sum_k X[token_t * d + k] * W_e[k * d_out + o]
}

// Phase 3 — scatter packed output back to original token rows of Y.
__global__ void moe_unpermute(const float *Y_packed, const int *perm,
                              float *Y, int T, int d_out) {
    // TODO: Y[perm[pos] * d_out + o] = Y_packed[pos * d_out + o]
}
