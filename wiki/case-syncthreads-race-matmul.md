# Case Study: Reproducing a `__syncthreads` Race in Tiled SGEMM

A reproducible non-deterministic correctness failure in a tiled SGEMM kernel that's missing a second `__syncthreads()` at the end of each phase. The signature — same binary, same input, different bounded errors across runs, with one occasional pass — is the textbook fingerprint of an inter-warp race on shared memory.

**Status:** bug present. Race empirically confirmed. Fix and post-fix verification pending.

## The kernel under test

Phase loop with **one** barrier per iteration:

```cuda
for (int phase = 0; phase < K/BK; ++phase) {
    Atile[ty*BK + tx] = A[...];
    Btile[ty*BK + tx] = B[...];
    __syncthreads();                  // (1) load → compute fence — present
    for (int k = 0; k < BK; ++k) {
        sum += Atile[ty*BK + k] * Btile[k*BK + tx];
    }
    // ── (2) compute → next-load fence — MISSING ──
}
```

A fast warp finishing its k-loop can start writing the **next** phase's tiles while a slower warp in the same block is still reading the **current** phase's tiles. Specifically: every warp reads every row of `Btile` (loop over `k`), but only writes its own `ty` row at load time — so warp 0 writing `Btile` row 0 in phase n+1 collides with warp 5 reading `Btile[k=0, tx]` in phase n.

For the load/sync/compute pattern with both barriers, see `shared-memory-tiling.md`.

## Hardware

| Item | Value |
|---|---|
| GPU | NVIDIA A40 (Ampere GA102) |
| Compute capability | 8.6 |
| CUDA | 12.4 |
| Build | `CUDAARCHS=86 cmake -DCMAKE_BUILD_TYPE=Release` |

## Setup

- M = N = K = 4096 (square)
- Block: 32 × 32 = 1024 threads
- Grid: 128 × 128 = 16,384 blocks
- Tile: BM = BN = BK = 32
- Tolerance for `verify`: `1e-2` absolute and relative

## Empirical data — 8 consecutive runs of the same binary

| Run | Verify | max_abs | max_rel | First mismatch index |
|---|---|---|---|---|
| 1 | FAIL | 6.79e-01 | 1.58e-01 | 12,721,665 |
| 2 | **PASS** | 0.00 | 0.00 | — |
| 3 | FAIL | 1.11e+00 | 8.32e+00 | 10,884,480 |
| 4 | FAIL | 1.57e+00 | 1.95e+01 | 2,498,848 |
| 5 | FAIL | 1.44e+00 | 6.74e-01 | 4,997,856 |
| 6 | FAIL | 1.31e+00 | 3.46e+00 | 6,816 |
| 7 | FAIL | 1.62e+00 | 5.67e+01 | 10,490,465 |
| 8 | FAIL | 1.34e+00 | 4.06e+01 | 6,160,928 |

7 of 8 runs fail. Wall-clock is consistent (~70.88 ms, ~1939 GFLOP/s, ~8.7% of cuBLAS) — the race corrupts values, not timing.

## What this signature tells you

Three failure patterns are diagnostically distinct:

| Symptom | Diagnosis |
|---|---|
| Same indices fail every run with same magnitude | Deterministic bug — wrong indexing, wrong loop bound, off-by-one |
| **Different indices each run, magnitudes vary but bounded** | **Race** — non-deterministic warp scheduling produces different overlap patterns each launch |
| Verify passes, throughput far below ceiling | Performance bug — coalescing, banks, occupancy |

Run 2's pass is not a contradiction. The race *opportunity* exists every iteration; whether the scheduler happens to interleave a write-then-read in a way that produces a wrong sum depends on timing below the model's visibility (cache state, other tenants on the SM, IL parallelism inside the warp). One pass in eight is consistent with an event of small per-element probability multiplied across millions of dependent reads.

The bounded `max_abs` (1–7, not orders of magnitude off, no NaNs) is also informative: when the race triggers, it swaps in *adjacent-phase* tile values. Those have the same statistical distribution as the correct values, so individual perturbations are O(1), not O(K).

## Performance footnote

Even if correct, ~8.7% of cuBLAS is what the bare-bones tiled kernel delivers — `shared-memory-tiling.md` calls this out under "What it still doesn't fix." The next jump comes from register tiling (`register-tiling.md`), where each thread accumulates a TM × TN rectangle of outputs.

## Open

- Post-fix throughput (after adding the second `__syncthreads()`): TBD — paste here for before/after.
- A separate latent issue — `__syncthreads()` placed *inside* `if (row < M && col < N)` — would deadlock for non-multiple-of-BK problem sizes. Deferred while the divisibility assertions hold.

## Context

**Source:** Empirical observation, kernel-forge phase 2 (`kernels/matmul/2_tiled.cuh`).
**Question that led here:** "Can we reproduce a `__syncthreads`-omission race condition empirically, and what does the failure pattern look like?"
