# 2026-04-27 — matmul/2_tiled — Race Discovery

## Status

- **Exercise:** Phase 2 (tiled SGEMM).
- **Kernel:** `kernels/matmul/2_tiled.cuh`. Race condition empirically confirmed — 7/8 runs fail verify against cuBLAS at M=N=K=4096. Cause: missing second `__syncthreads()` at the end of the phase loop.
- **Latent (not yet fixed):** the existing `__syncthreads()` is *inside* `if (row < M && col < N)` — would deadlock at non-divisible problem sizes. Currently safe because of the divisibility asserts.
- **Pod:** A40 (sm_8.6, GA102). Built with `CUDAARCHS=86`, binary at `build/matmul`. Throughput ~1939 GFLOP/s = 8.7% of cuBLAS (race corrupts values, not timing).

## Where we are in the Socratic thread

Confirmed correct over recent iterations:
- Coalescing pattern α: `Atile[ty*BW+tx] = A[row*K + phase*BW+tx]`. Within a warp (fixed `ty`, varying `tx`), GMEM addresses are contiguous → 1 transaction per warp per load.
- Register accumulator (`float sum = 0`) lives in registers across the K-sweep; one `C[row*N+col] = sum` write at the end.
- 1:1 thread-to-tile-element mapping for the `BM = BN = BK = blockDim.x = blockDim.y` square case. User understands production kernels relax this with a load loop per thread.
- Template signature restored to `<int BM=32, int BN=32, int BK=32>` to match runner's `sgemm_tiled<32,32,32>`.

Empirical milestone:
- Reproduced the race signature: 7/8 fails, scattered mismatch indices, bounded `max_abs` of 1–7. Captured in `wiki/case-syncthreads-race-matmul.md`.

## Open Socratic questions (not yet answered)

These were left hanging when the user pivoted to running the kernel and observing the race empirically:

1. **Where does the missing barrier go?** What barrier prevents the inter-warp race, and where in the phase loop does it have to be (which line, exactly)?
2. **Why isn't the existing `__syncthreads` enough?** The first barrier (after the load) does *not* solve this. Why not?

After the user fixes this:

3. **Did the fix change throughput?** Pre- and post-fix should both be ~1939 GFLOP/s — race corrupts values, not timing. If post-fix throughput is different, that's a separate signal worth investigating.

Deferred (latent issue, not yet probed):

4. **The `__syncthreads` inside the if-block.** What happens to threads outside `(row < M && col < N)` at non-multiple sizes? What guarantee does `__syncthreads` make about which threads must reach it?

## Next session plan

1. Resume tutor mode. Don't re-explain the race — user has now seen it empirically; they need to write the fix.
2. User adds the second `__syncthreads()` at the end of the phase loop, re-syncs to pod, rebuilds, runs the 8x loop. Expect 8/8 PASS.
3. Update `wiki/case-syncthreads-race-matmul.md` "Open" section with post-fix data and a one-line conclusion.
4. Probe Q4 (the `__syncthreads`-in-if issue) before moving on, ideally by running with a non-multiple problem size to surface it empirically.
5. Then assess: register tiling next (`register-tiling.md` is stub-only), or stay on tiled with rectangular tile shapes (BM ≠ BN ≠ BK)?

## Cross-references

- **Case study:** `wiki/case-syncthreads-race-matmul.md` — empirical record (8 runs, signature analysis, performance footnote, "Open" section to fill in post-fix).
- **Wiki updates this session:**
  - `wiki/shared-memory-banks.md` (new — 32 banks, terminology for bank/port/lane/cycle, three access patterns + ASCII visualization, `+1` padding mitigation, detection).
  - `wiki/shared-memory-tiling.md` (cross-link to banks page in "What it still doesn't fix"; question footer filled).
  - `wiki/register-tiling.md`, `wiki/occupancy.md` (question footers filled).
  - `wiki/index.md` (two new entries).
- **Kernel:** `kernels/matmul/2_tiled.cuh` — race present, fix pending.
- **Build/run on pod:**
  - `CUDAARCHS=86 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
  - `./build/matmul 2 4096 4096 4096`
  - `for i in 1 2 3 4 5 6 7 8; do ./build/matmul 2 4096 4096 4096; done` for race signature.
- **Memory rules in play:** `feedback_wiki_first.md` (read wiki before answering conceptual questions; lazy-fill question footers), `feedback_pedagogical_simplicity.md`, `user_role.md` (Amiya writes the kernels — Socratic only, scaffold/stub only).
