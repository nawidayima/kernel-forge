# Arithmetic Intensity

FLOPs executed per byte transferred between global memory and the chip. This single number decides whether a kernel is *memory-bound* (DRAM pipe runs dry waiting to feed the ALUs) or *compute-bound* (ALUs are the bottleneck). Every optimization after naive coalescing is, at its core, one long climb of this ratio.

## The roofline

Every GPU has two ceilings:

- **Compute peak** — how fast the ALUs retire fused-multiply-adds (FLOPs/s).
- **Memory peak** — how fast DRAM can feed the chip (bytes/s).

Their ratio defines the **ridge**:

```
ridge = compute_peak / memory_bandwidth   [FLOPs per byte]
```

A kernel with AI below the ridge is memory-bound; its maximum GFLOP/s is `AI × memory_bandwidth` no matter how fast the ALUs are. A kernel above the ridge is compute-bound; adding bandwidth wouldn't help.

For modern data-center GPUs the ridge is typically in the range of tens to low hundreds of FLOPs/byte. A naive kernel that does "one FMA per loaded value" sits at AI ≈ 0.25 FLOPs/byte — two orders of magnitude below the ridge.

## SGEMM as a case study

A dense matmul C = A·B of size N has:

- Useful work: `2N³` FLOPs (one FMA per inner-loop iteration, N³ iterations)
- Minimum memory traffic: `3N² × 4B` (read A, read B, write C — each element touched once)
- **Minimum AI: 2N³ / 12N² = N/6 FLOPs per byte**

At N=4096 the minimum AI is ~680 FLOPs/byte — well above any GPU's ridge. SGEMM *should* be compute-bound.

The catch: a naive kernel re-reads A and B from DRAM on every inner-loop iteration, loading ~N³ bytes instead of ~3N². Its *effective* AI is ~0.25 FLOPs/byte, which is why it runs at a few percent of peak.

**The optimization ladder is the process of pushing effective AI from ~0.25 back up toward N/6.**

## The ladder in AI terms

Each tier cuts GMEM loads per output element by a constant factor. With 2K FLOPs per C result:

| Tier | GMEM floats per result | Effective AI |
|---|---|---|
| Naive (re-read everything) | ~2K per FMA iteration | ~0.25 FLOPs/B |
| SMEM tile, BM=BN=B | `~2K / B` | `~B / 2` FLOPs/B |
| SMEM + 1D register tile, TM | `~2K / (B·TM)` halved more | `~B·TM / 2` FLOPs/B |
| SMEM + 2D register tile, TM=TN=T | further ÷ T/2 | `~B·T² / (2(T+T))` = `~B·T/4` FLOPs/B |

Concrete walk for B=32, T=8: 8 → 16 → 32 FLOPs/byte across the tiling tiers. A well-tuned production SGEMM approaches the ridge — it has extracted essentially all the reuse the math permits.

## When to use this concept

Before reaching for a profiler, compute the AI ceiling the kernel's memory pattern allows:

```
ceiling_gflops = AI × memory_bandwidth
```

If measured GFLOP/s is close to this ceiling, the kernel is memory-bound and needs higher AI (more reuse, more per-thread work) — adding occupancy or tweaking launch parameters won't help. If measured GFLOP/s is far below the ceiling, the problem is latency or instruction throughput, not bandwidth, and the levers are different.

## Context

**Source:** siboehm.com/articles/22/CUDA-MMM
**Question that led here:**
