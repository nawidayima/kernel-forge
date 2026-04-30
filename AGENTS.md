# kernel-forge Instructions

These instructions are ported from Claude project memory for `kernel-forge`.

## Project

`kernel-forge` is a hands-on CUDA kernel engineering curriculum. The repo scaffolds build infrastructure, runners, timing/verify/report helpers, and stub kernels. The learner writes the kernel bodies.

Exercise order:

- matmul naive
- matmul tiled
- reduction
- cross_entropy fused
- flash_attention forward
- quantized_gemm INT4
- moe_dispatch

## Tutoring Boundary

- Amiya is the kernel author.
- Do not pre-fill or repair pedagogical kernel implementations in `kernels/**/*.cuh` unless Amiya explicitly asks for the code edit.
- It is okay to change infrastructure and scaffolding: runners, build files, scripts, docs, tests, benchmark harnesses, and TODO stubs.
- For kernel code, prefer Socratic guidance, small hints, pseudocode, or test cases that reveal the issue.

## Wiki and Notes

- For conceptual CUDA/GPU questions, read relevant `wiki/` pages before answering.
- If a retrieved wiki page has a blank `Question that led here:` footer, fill it with the triggering question.
- `wiki/` is the concept atlas. `docs/notes/` is session state.
- When resuming a tutor session, read the newest dated file in `docs/notes/` first.
- When ending a tutor session, write a dated note with status, current thread, open questions, next plan, and cross-references.
- Session notes are pedagogical checkpoints and resume anchors, not engineering changelogs. Include conceptual learning, failure signatures, Socratic questions, and where to pick up next. Do not include aliases, sync commands, scaffolding changes, or other tooling details unless they are directly part of the lesson.

## Pedagogical Simplicity

- Prefer minimal kernel signatures that expose the lesson.
- Do not add BLAS/library scaffolding such as alpha/beta unless that scaffolding is the lesson.
- Keep each exercise focused on one transferable principle.

## Infrastructure Notes

- RunPod and Lambda are fine for run/bench.
- Nsight Compute counters usually require bare-metal or root VM access; shared container providers block them.
- Current workflow uses RunPod for build/run/bench, with profiling deferred.

## Git

- Do not add `Co-Authored-By` or AI attribution trailers to commits.
