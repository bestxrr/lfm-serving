# Draft-Prefix Production Crash Fix

Date: 2026-07-30

Evidence label: LOCAL-CLEAN-MECHANISM. No shared-H100 latency claim is made.

## Failure

The production candidate could terminate EngineCore in mixed, prefix-cached
traffic when draft metadata combined an active-request row-count vector with a
capacity-sized block table. `compute_new_slot_mapping()` then received
different request dimensions for `input` and `repeats`.

## Fix

- Slice rebuilt draft block tables to the active request count.
- Derive query lengths once and permanently assert that their request dimension
  matches the active block-table rows before `repeat_interleave`.
- Preserve zero-row requests as zero repeats.
- Guard the proposer path. A pre-mutation failure skips speculation for that
  step; a post-mutation failure also disables speculation for the remainder of
  the engine run so a dirty draft state cannot be reused.
- Gate readiness on startup warmup completion. Warmup covers greedy and top-p
  sampling, short and 4,400-token prompts, and a four-request burst containing
  two full-prefix hits.

Runtime patch:
`submission/patches/vllm-0.25.1-lfm2-draft-prefix-v11.patch`

## Validation

- ShortConv canonical property test: 128/128 randomized cases bitwise exact.
- Fault injection before draft mutation: engine survived and completed 6/6
  requests; speculation resumed on later steps.
- Corrected BF16-draft REP30 run: 30/30 requests completed without EngineCore
  failure (`results/v0251_fp8_rep30_draft350m_bf16_20260730-092917`).
- Production entrypoint smoke: server and CUDA graphs initialized; all warmup
  requests completed; the full-hit four-request burst completed; readiness
  sentinel was written; observed prefix hit rate was 46.5%; shutdown was clean.
- Peak project VRAM in the fault-injection run: 8,634 MiB.

The local smoke used `VLLM_USE_FLASHINFER_SAMPLER=0` because of the local
environment. The Docker image does not set this variable and therefore retains
the production image's default FlashInfer sampler.

## Verdict

The observed engine-fatal metadata mismatch is fixed and guarded against future
proposer exceptions. Candidate `prefix-v3` is production-crash-ready for image
construction. Its local REP30 performance did not beat the paired FP8 baseline,
so the submission remains an official-environment mechanism test rather than a
local latency winner.
