# Shared-Memory Banks and Conflicts

Shared memory is fast because it's on-chip SRAM, but the bandwidth into it is partitioned. To serve a warp's 32 lanes in one cycle, the SRAM is split into **32 parallel banks**. When two lanes hit the same bank at different addresses, the bank serializes them — a *bank conflict* — and the warp's load takes as many cycles as the worst-conflicted bank.

## Terminology

- **Bank** — an independently-addressable SRAM module with its own port. Shared memory is built as 32 banks running in parallel rather than as one giant SRAM, because that's the only practical way to give a warp 32-access-per-cycle aggregate bandwidth.
- **Port** — the physical access channel into a memory module. A single-ported SRAM serves exactly one read *or* one write per cycle. Each bank has one port, so two lanes hitting the same bank at different addresses queue at that single port and serialize. (NVIDIA documents the *behavior* — one access per cycle per bank — but not the port-count implementation. The single-port framing is general computer-architecture practice; multi-ported SRAM at this size is impractical because port count costs roughly quadratic transistor area.)
- **Lane** — one thread's slot within a warp. A warp has 32 lanes, indexed 0..31. When a warp issues a single load from shared memory, each lane carries its own address; the bank-conflict question is whether those 32 addresses spread across banks or pile onto one.
- **Cycle** — one tick of the SM's clock (sub-nanosecond at typical GPU clock rates, ~1–2 GHz). Hardware throughput is reported per cycle: "one access per cycle" means one access per clock tick. Latency to *use* a loaded value in a downstream FMA is longer (many cycles of pipeline), but throughput is what governs bank-conflict behavior.

## Bank layout

Successive 4-byte words are assigned to successive banks, round-robin:

```
byte addr     0    4    8   12  ...  124  128  132  ...
word index    0    1    2    3  ...   31   32   33  ...
bank          0    1    2    3  ...   31    0    1  ...
```

So `bank(addr) = (addr / 4) % 32`. A warp reading 32 contiguous words hits 32 different banks → 1 cycle.

## Three patterns a warp can produce

| Pattern | What the 32 lanes do | Bank load | Cycles |
|---|---|---|---|
| **Conflict-free** | 32 different addresses, one per bank | even | 1 |
| **Broadcast** | All lanes read the *same* address | 1 bank, fanned to all | 1 |
| **K-way conflict** | K lanes hit one bank at K different addresses | that bank serializes | K |

The broadcast exception matters in matmul: in the inner compute loop, a whole warp often reads `As[ty][k]` with `ty` constant within the warp — every lane wants the same address → broadcast, no penalty.

Same three patterns drawn at 8 lanes × 8 banks for legibility (real hardware is 32 × 32):

```
─── Conflict-free (1 cycle) ─────────────────────────
 Lanes:  L0  L1  L2  L3  L4  L5  L6  L7
          │   │   │   │   │   │   │   │
          ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
 Banks:  B0  B1  B2  B3  B4  B5  B6  B7
         every bank serves one access — all 8 in parallel

─── Broadcast (1 cycle) ─────────────────────────────
 Lanes:  L0  L1  L2  L3  L4  L5  L6  L7
          └───┴───┴───┴───┴───┴───┴───┘
                       │  (same address)
                       ▼
 Bank:   B0  ··  ··  ··  ··  ··  ··  ··
         one read, hardware fans the value to all 8 lanes

─── 8-way conflict (8 cycles) ───────────────────────
 Lanes:  L0  L1  L2  L3  L4  L5  L6  L7
          │   │   │   │   │   │   │   │
          └───┴───┴───┴───┴───┴───┴───┘
                       │  (8 different addresses,
                       ▼   all map to B0)
 Bank:   B0  ··  ··  ··  ··  ··  ··  ··
         B0's single port serializes 8 reads — other 7 banks idle
```

Conflict-free uses all 8 banks once; conflict uses 1 bank 8 times. The wasted bandwidth in the conflict case isn't just slower — it's *wasted*: B1–B7 sit idle while B0 catches up.

## Where conflicts appear

The canonical bite happens when a warp accesses a 2D shared tile by *column*: lane `i` reads `tile[i][col]` with `col` fixed and `i` varying across the warp. Addresses are then `stride` words apart. If `stride` is a multiple of 32, every lane maps to the same bank:

```
bank((i * 32) % 32) = 0 for all i   →   32-way conflict
```

This shows up in optimized SGEMM kernels that load A transposed into shared memory so the inner loop can broadcast across K. It also shows up any time you have a `[N][32]` shmem tile and read down a column.

## The +1 padding mitigation

Declaring the inner dimension one larger than the access stride breaks the conflict:

```cuda
__shared__ float As[BM][BK + 1];   // pad to BK+1 instead of BK
```

Now the row stride is `BK+1` words. If `BK = 32`, then `bank((i * 33) % 32) = i % 32` — every lane lands on a different bank. Cost: one wasted column per row. Trade: bytes for bandwidth.

This is a derived consequence of the bank rule, not a separate NVIDIA prescription. It's standard in the SGEMM optimization literature (siboehm.com/articles/22/CUDA-MMM, CUTLASS).

## Detecting it

`ncu` reports `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` (and the matching `_st` counter) per kernel. When `ncu` is blocked by the host (Lambda, RunPod), you predict instead: any time a warp reads a 2D shared tile down a column whose stride is a multiple of 32, expect an N-way conflict and add `+1` padding.

## Context

**Source:** CUDA C++ Best Practices Guide §10.2.3.1 ("Shared Memory and Memory Banks") for the bank rule, serialization behavior, and broadcast exception. The padding trick is derived; common references are Boehm's SGEMM tutorial and CUTLASS.
**Question that led here:**
