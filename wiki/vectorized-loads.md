# Vectorized Loads

A warp issuing 32 × 32-bit loads does one 128-byte transaction (coalescing). A single thread issuing one 128-bit load does the same single transaction — but using **one instruction instead of four**. Vectorized loads compress the instruction stream without changing the memory pattern.

## What the hardware offers

Three widths are available on every modern CUDA arch:

| Instruction | Width | C expression |
|---|---|---|
| `LD(G/S).E`     | 32-bit  | `float x = *p;`             |
| `LD(G/S).E.64`  | 64-bit  | `float2 x = *(float2*)p;`   |
| `LD(G/S).E.128` | 128-bit | `float4 x = *(float4*)p;`   |

`LDG` is from GMEM, `LDS` from SMEM. Fewer instructions fetched, decoded, and queued into the memory pipeline (the "MIO" pipe) directly relieves the bottleneck that tiled-kernel inner loops almost always hit.

## GMEM side: `reinterpret_cast` as an alignment promise

To get `LDG.E.128` on a user-supplied pointer, you write:

```cpp
float4 a_frag = *reinterpret_cast<float4*>(&A[row*K + col]);
```

The compiler emits the 128-bit load **iff** it can prove the address is 16-byte aligned. For a kernel-argument `float* A`, the compiler cannot prove this — so without the cast it falls back to four separate `LDG.E` even when the indices are contiguous. The `reinterpret_cast` is a programmer assertion:

> *"I have arranged the outer dimensions so that this pointer is 16-byte aligned. Trust me and emit the wide load."*

The tile dimensions must actually satisfy the alignment — tile widths must be multiples of 4 floats, and the base pointer must be 16B-aligned (which `cudaMalloc` guarantees).

## SMEM side: layout controls the opcode

SMEM is allocated by the compiler, so it *can* prove alignment and will auto-emit `LDS.128` — *if* the access pattern asks for 4 contiguous floats per thread. Whether it does depends on the tile layout:

- If threads within a warp walk along the **contiguous** axis of the SMEM tile (stride 1), the compiler packs 4 × 4B into one `LDS.128` per thread.
- If threads walk along the **strided** axis, each thread asks for one scalar at its own stride — compiler must emit narrow loads.

The typical fix: **store A transposed in SMEM** so the per-thread inner loop reads stride-1 along A as well as along B. That single change lets both `As` and `Bs` loads become `LDS.128`.

```
A loaded as-is:          A stored transposed in SMEM:
                         (threads step along rows → stride 1)

row → stride K                col → stride 1
 T0  T1  T2  T3                T0  T1  T2  T3
  │   │   │   │                │   │   │   │
  ▼   ▼   ▼   ▼                ▼   ▼   ▼   ▼
[ . . . . . . . . ]          [ . . . . . . . . ]
  narrow LDS per thread        one LDS.128 per thread
```

## Why this is the last few percent, not a step change

Vectorization doesn't change *bytes* moved or *FLOPs* computed — only *instructions* dispatched per warp. Expect low single digits of % improvement, not a step change. Its place on the ladder is specifically when the kernel is already arithmetic-intensity-saturated and stalling on MIO-pipe congestion: removing 3 of every 4 SMEM load instructions directly lifts throughput. It's the cheapest few percent available once the tile hierarchy is right.

## Context

**Source:** siboehm.com/articles/22/CUDA-MMM (Kernel 6)
**Question that led here:**
