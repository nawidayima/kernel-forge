# 2026-04-30 — matmul/2_tiled — Ragged Boundary Tests

## Status

- **Exercise:** Phase 2 (tiled SGEMM).
- **Conceptual checkpoint:** the original shared-memory race was fixed by adding the second `__syncthreads()` at the end of the phase loop. Post-fix runs at `M=N=K=4096` passed consistently at ~1907 GFLOP/s, roughly 8.3-8.4% of cuBLAS.
- **New lesson:** once the divisible case is correct, ragged dimensions expose a different class of mistakes: loader predicates, output predicates, partial K phases, and final stores must be reasoned about separately.

## Where we are in the Socratic thread

Confirmed:

- `__syncthreads()` is a block-wide barrier and shared-memory ordering point. The first barrier protects load -> compute; the second protects compute -> next phase overwrite.
- The second barrier made the 4096 divisible case correct. Throughput stayed in the same band, with a small plausible slowdown from the added barrier.
- Non-multiple dimensions expose different bugs than the original race. The observed cases gave separate signals for M edges, N edges, and K remainders.
- The cleaner phase structure now being debugged has these conceptual pieces:
  - ceiling phase count: `(K + BK - 1) / BK`
  - separate predicates for A tile load and B tile load
  - zero-fill for invalid shared-memory tile entries
  - unconditional barriers
  - output-guarded compute

Current latest observed test output:

```text
M=32 K=32 N=32      PASS
M=64 K=64 N=64      PASS
M=33 K=32 N=32      PASS
M=32 K=32 N=33      FAIL
M=32 K=33 N=32      FAIL
M=33 K=33 N=33      FAIL
M=31 K=31 N=31      FAIL
M=127 K=65 N=97     FAIL
M=257 K=128 N=128   FAIL
M=128 K=257 N=128   FAIL
M=128 K=128 N=257   FAIL
M=257 K=129 N=193   FAIL
summary 3/12 PASS
```

Important interpretation:

- `M=33 K=32 N=32 PASS` says the M edge is no longer the main problem.
- `M=32 K=32 N=33 FAIL` isolates an N-edge problem.
- Inspection showed the remaining definite bug: final `C[row*N + col] = sum` is still unconditional. For `N=33`, invalid logical column `col=33` aliases valid linear address `row=1, col=0`, so invalid edge-block threads can overwrite valid C entries.
- Earlier `K=33` and `K=31` failures diagnose the phase-count issue. With ceiling division now present, remaining K failures should be rechecked after guarding the final C store.

## Open Socratic Questions

1. Add the final output guard: why does `(row=0, col=33)` alias `(row=1, col=0)` in row-major storage when `N=33`?
2. After guarding the final store, which tests still fail? If K-ragged cases still fail, inspect the phase count and zero-fill predicates again.
3. Why can invalid-output threads still be valid cooperative loaders? Explain using `M=32, K=32, N=33`, where only `tx=0` has a valid output in the second column block.
4. After all ragged correctness cases pass, compare the final divisible-case throughput to the post-race-fix baseline (~1907 GFLOP/s). Did making boundary behavior correct change the fast path?

## Next Session Plan

1. Resume from the remaining definite bug: final C store needs the output-valid predicate.
2. Use the N-edge case (`M=32, K=32, N=33`) to explain row-major aliasing from an invalid logical column into a valid later-row element.
3. Re-run ragged correctness cases; expect the N-edge failures to improve first.
4. If K-ragged cases still fail, inspect the phase count and zero-fill logic.
5. Once correctness passes, compare final divisible-case throughput to the post-race-fix baseline.

## Cross-references

- **New wiki page:** `wiki/boundary-tiles.md` — separate predicates for A loads, B loads, C compute/store, and unconditional barrier participation.
- **Existing case study:** `wiki/case-syncthreads-race-matmul.md` — original race reproduction, now updated with post-fix performance and linked forward to boundary-tile handling.
