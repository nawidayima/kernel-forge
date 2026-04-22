# Register Tiling

Give each thread a rectangular block of outputs instead of a single scalar. The per-thread accumulator lives in registers, and every SMEM load now feeds multiple FMAs. This is the largest single jump on the SGEMM optimization ladder, and the lesson generalizes: **computing more per thread raises arithmetic intensity quadratically.**

## Step 1: 1D — one thread owns a column of TM outputs

Instead of thread (i,j) computing `C[i,j]`, let it compute `C[i..i+TM, j]` — a column of TM outputs. The inner loop:

```
for k in 0..BK:
    b = Bs[k, j]                     // one SMEM load of B
    for m in 0..TM:
        acc[m] += As[i+m, k] * b     // TM FMAs, one SMEM load of A each
```

One SMEM load of B is reused across TM FMAs. For TM=8, that's one B-load per 8 multiply-adds instead of one per one. GMEM loads per result also drop — fewer threads per tile means each loaded tile element serves fewer redundant copies of itself.

## Step 2: 2D — one thread owns a TM×TN rectangle

A column gives *linear* reuse (factor TM). A rectangle of size TM×TN gives **quadratic reuse** via outer-product structure:

```
for k in 0..BK:
    regM[0..TM] = As[i..i+TM, k]              // TM SMEM loads
    regN[0..TN] = Bs[k, j..j+TN]              // TN SMEM loads
    for m in 0..TM:
        for n in 0..TN:
            acc[m,n] += regM[m] * regN[n]     // TM*TN FMAs, zero SMEM loads
```

Per `k` iteration: **TM·TN FMAs for (TM+TN) SMEM loads**. Reuse factor `TM·TN / (TM+TN)`.

| Shape | Results/thread | SMEM loads per FMA |
|---|---|---|
| scalar | 1 | 2 |
| column (1D, TM=8) | 8 | 9/8 ≈ 1.13 |
| square (2D, TM=TN=8) | 64 | 16/64 = 0.25 |

A square isn't just "more results" — it's a qualitatively better reuse pattern. Doubling both TM and TN from 4 to 8 quadruples results per thread *and* doubles the reuse factor.

## The outer-product inner loop, visualized

```
                 regN (TN values loaded from Bs)
                   0   1   2   3   4   5   6   7
                ┌───┬───┬───┬───┬───┬───┬───┬───┐
regM[0] ───►    │ + │ + │ + │ + │ + │ + │ + │ + │   acc[0,*] += regM[0] * regN[*]
                ├───┼───┼───┼───┼───┼───┼───┼───┤
regM[1] ───►    │ + │ + │ + │ + │ + │ + │ + │ + │
                ├───┼───┼───┼───┼───┼───┼───┼───┤
regM[2] ───►    │ + │ + │ + │ + │ + │ + │ + │ + │
                ├───┼───┼───┼───┼───┼───┼───┼───┤
   ...          │              ...                │
                ├───┼───┼───┼───┼───┼───┼───┼───┤
regM[7] ───►    │ + │ + │ + │ + │ + │ + │ + │ + │
                └───┴───┴───┴───┴───┴───┴───┴───┘

TM+TN = 16 SMEM loads  produce  TM·TN = 64 FMAs
```

Every `acc[m,n]` is a register that stays live for the whole K-sweep; only at the end of the K-loop is it written back to C.

## Cost: register pressure

Each thread holds `TM·TN + TM + TN` floats in registers (accumulator + operand caches), plus indexing and loop overhead. For TM=TN=8 that's ~80 registers per thread before bookkeeping. This pushes against the SM's register-file budget and can drop occupancy (see `occupancy.md`).

The trade is almost always worth it: the register file is typically several times larger than SMEM per SM, and quadratic reuse beats the latency-hiding occupancy buys. But there's a ceiling — past some TM·TN, register spills to local memory (which lives in GMEM) erase all the gains.

## What it still doesn't fix

Even with big register tiles, threads within a warp may end up loading SMEM chunks whose layout the compiler can't see as warp-aligned. The next lever is laying the per-thread tiles out to match warp boundaries — see `warp-tiling.md`.

## Context

**Source:** siboehm.com/articles/22/CUDA-MMM (Kernels 4, 5)
**Question that led here:**
