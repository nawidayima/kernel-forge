# 2026-05-04 — matmul/2_tiled — Output Guard + Perf Prediction

## Status

- **Exercise:** Phase 2 (tiled SGEMM). Correctness now passing on all ragged sizes.
- **Change:** Final `C[row*N + col] = sum` is hoisted out of the phase loop and guarded by `if (row < M && col < N)`. Previously it lived inside the phase loop inside the compute bounds-guard, so each output cell received P = ceil(K/BK) stores instead of 1.
- **GPU:** NVIDIA L4 (RunPod, pod 9270hxgvsrb2ra). Same GPU as the Apr 21 baseline in `benchmark_results/matmul_L4_2026-04-21.csv`.

## Results

Ragged correctness suite: **12/12 PASS** (was 3/12 last session).

```
test kernel=2 M=32 K=32 N=32      PASS
test kernel=2 M=64 K=64 N=64      PASS
test kernel=2 M=33 K=32 N=32      PASS
test kernel=2 M=32 K=32 N=33      PASS
test kernel=2 M=32 K=33 N=32      PASS
test kernel=2 M=33 K=33 N=33      PASS
test kernel=2 M=31 K=31 N=31      PASS
test kernel=2 M=127 K=65 N=97     PASS
test kernel=2 M=257 K=128 N=128   PASS
test kernel=2 M=128 K=257 N=128   PASS
test kernel=2 M=128 K=128 N=257   PASS
test kernel=2 M=257 K=129 N=193   PASS
```

4096³ throughput, before vs after the store hoist:

| Variant                                     | GFLOP/s | % cuBLAS |
| ------------------------------------------- | ------- | -------- |
| In-loop guarded store (P=128 stores / cell) | ~1220   | 9.8%     |
| Post-loop guarded store (1 store / cell)    | ~1245   | 10.0%    |

cuBLAS on this pod runs ~12,400 GFLOP/s, so this is a different (slower) GPU than the previous A6000-class pod, where the unfixed kernel hit 1907 GFLOP/s with cuBLAS at ~22,000 GFLOP/s. Cross-pod absolute numbers are not comparable. Same-pod relative deltas are.

## Perf prediction vs observation

Prediction before running: 2-5% speedup. Observed: ~2%.

Reasoning: 127 redundant stores per output cell means about 8.5 GB of extra store traffic on 4096³. Every redundant store goes to the same address as the previous store, so L2 absorbs them and DRAM sees roughly 1 write per cell either way. The cost is L2 write-port pressure plus an extra STG instruction issue per phase, both small relative to the inner k-loop's work.

The speedup matching the lower end of the prediction is the actual lesson here: the kernel is not store-bound. The next ~10x of throughput up to cuBLAS has to come from upstream of the C store. Likely candidates: register tiling (each thread computes a TMxTN sub-tile), vectorized loads (`float4`), bank-conflict-free shared layout, larger BM/BK.

## Aliasing intuition (why the unguarded store fails N-edge but not M-edge)

For N-edge cases like `M=32 K=32 N=33`:
- Block `bx=1` covers logical columns 32..63. Only `tx=0` (col=32) is a valid output column.
- Invalid thread `ty=0, tx=1` computes `row*N + col = 0*33 + 33 = 33`.
- Index 33 in row-major C with N=33 columns is `(row=1, col=0)`, a valid cell that another thread wrote correctly. The invalid thread's store overwrites it.

For M-edge cases like `M=33 K=32 N=32`:
- Block `by=1` covers rows 32..63. Only `ty=0` (row=32) is a valid output row.
- Invalid thread `ty=1, tx=0` computes `33*32 + 0 = 1056`.
- C is allocated as M*N = 1056 floats, valid indices 0..1055. Index 1056 lands past the end of the allocation. The store goes into padding the verifier never reads.

So the M-edge case before the guard wasn't actually correct, just lucky: it scribbled past the allocation rather than into a live cell. Row-major aliasing only bites when the *last* (innermost) dimension overshoots, because that is the dimension that wraps modulo the row stride. Earlier-dimension overshoots fall off the end.

Column-major flips the asymmetry.

## Why the test suite cannot rank E vs B vs C

The runner verifies post-kernel `C` against cuBLAS using `max_abs` and `max_rel`. It observes one snapshot, after the kernel returns. It does not see how many writes happened, what intermediate values were stored, or how much L2/DRAM bandwidth was burned. Throughput is reported but not asserted.

So any kernel where the *last* write to each `C[i]` is the correct dot product passes. Pre-fix in-loop store (E) is functionally equivalent to canonical post-loop store (B) at the verifier interface. Distinguishing them requires either timing comparison, an Nsight Compute store-count metric, or manual inspection.

## Open questions for next session

1. Where is the kernel actually bottlenecked? `ncu --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis ./build/matmul 2 1024 1024 1024` to find dominant stall reason.
2. The compute body inside the phase loop is still wrapped in `if (row < M && col < N)`. Canonical SGEMM computes unconditionally because zero-padded shared tiles make the wasted compute harmless. Worth removing if it doesn't change correctness.
3. Register tiling (`wiki/register-tiling.md` is stub-only): each thread computes a TMxTN sub-tile of C instead of a single element. Arithmetic intensity rises from BM to TM\*TN with no extra shared-memory pressure.
4. Vectorized loads: replace per-float global loads of A and B with `float4`, cutting load instruction count by 4x. Wiki page `vectorized-loads.md` already exists.

## Cross-references

- Updated: `kernels/matmul/2_tiled.cuh` (output guard hoisted, dead commented-out block removed).
- Previous session: `docs/notes/2026-04-30-matmul-ragged-boundary-tests.md` — open question about row-major aliasing is answered above.
- Wiki: `wiki/boundary-tiles.md` — output guard now empirically validated.
