#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>

// CPU reference for cross-entropy loss: for each row of `logits` (B x V),
// compute -log softmax(logits[b])[target[b]], then average over the batch.
// Returns the scalar mean loss.
inline float cpu_ce_reference(const float *d_logits, const int *d_targets,
                              int B, int V) {
    float *h_logits = (float *)std::malloc(B * V * sizeof(float));
    int *h_targets = (int *)std::malloc(B * sizeof(int));
    cudaMemcpy(h_logits, d_logits, B * V * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_targets, d_targets, B * sizeof(int), cudaMemcpyDeviceToHost);

    double loss = 0.0;
    for (int b = 0; b < B; ++b) {
        const float *row = h_logits + b * V;
        float max_v = row[0];
        for (int v = 1; v < V; ++v) if (row[v] > max_v) max_v = row[v];
        double denom = 0.0;
        for (int v = 0; v < V; ++v) denom += std::exp((double)(row[v] - max_v));
        double log_denom = std::log(denom) + max_v;
        loss += log_denom - (double)row[h_targets[b]];
    }
    loss /= B;

    std::free(h_logits);
    std::free(h_targets);
    return (float)loss;
}
