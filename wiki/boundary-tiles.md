# Boundary Tiles

Tiled kernels are easiest to reason about when `M`, `N`, and `K` are exact multiples of the tile size. Real kernels cannot assume that. The last block in each dimension is often only partially valid, and the last K phase may contain fewer than `BK` dot-product terms.

The key distinction is:

```
loader validity != output validity != barrier participation
```

Each thread in a block has three jobs in a tiled SGEMM phase:

1. Load one element of the A tile.
2. Load one element of the B tile.
3. Accumulate one output element of C.

Those jobs have different predicates:

| Job | Valid when |
|---|---|
| Load A | `row < M && A_load_col < K` |
| Load B | `B_load_row < K && col < N` |
| Accumulate/store C | `row < M && col < N` |

A thread with an invalid output column may still be responsible for loading a valid A element. A thread with an invalid output row may still be responsible for loading a valid B element. If the whole phase is guarded by `if (row < M && col < N)`, the edge block stops being a cooperative loader.

## Zero-fill partial tiles

Out-of-bounds tile entries should be written as zero in shared memory:

```cuda
if (row < M && A_load_col < K) {
    Atile[ty*BK + tx] = A[row*K + A_load_col];
} else {
    Atile[ty*BK + tx] = 0.0f;
}
```

The same applies to B. This lets the inner loop stay fixed-width (`k = 0..BK-1`) even when the final K phase is partial. Invalid lanes contribute `0 * x` or `x * 0`, which is algebraically neutral.

## Barriers are not bounds-checked

Every thread in the block must reach each `__syncthreads()`. Bounds checks protect global memory accesses and final stores; they do not decide whether a thread participates in the barrier.

The phase count also needs ceiling division:

```cuda
num_phases = (K + BK - 1) / BK
```

Using `K / BK` silently drops the final partial K tile. For `K < BK`, it runs zero phases and leaves C at zero.

## Edge-store aliasing

The final C store must be guarded too:

```cuda
if (row < M && col < N) {
    C[row*N + col] = sum;
}
```

For `N=33`, an invalid logical coordinate like `(row=0, col=33)` maps to the same linear address as `(row=1, col=0)`. An unguarded edge-block store can therefore corrupt a valid element, not merely write past the end of the row.

## Context

**Exercise:** matmul/2_tiled (Phase 2)
**Question that led here:** "What do ragged matmul test failures reveal about bounds checks, cooperative loading, and final stores?"
