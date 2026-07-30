# Generic LFM2 Draft Prefix/Graph Image Gate

Date: 2026-07-30

Evidence label: LOCAL-CLEAN-MECHANISM for correctness/counters; LOCAL-CONTENDED
for all wall-clock latency. No local latency result is used as H200 evidence.

## Runtime and candidate

- vLLM: 0.25.1, upstream tag commit
  `752a3a504485790a2e8491cacbb35c137339ad34`
- Target: `LiquidAI/LFM2.5-1.2B-Instruct`, online
  `fp8_per_tensor`
- Draft: `LiquidAI/LFM2.5-350M` BF16, revision
  `b9d6e4e2d75f440b12a2b4d731c808004ecbbd89`
- Speculation: generic draft model, k=1
- Scheduler: `max_model_len=5120`, `max_num_batched_tokens=512`
- Prefix caching: enabled, xxhash
- CUDA graphs: enabled
- Production compose: `max_num_seqs=8`

## Root cause and fix

The final prefix-caching failure was not a hardware or capacity issue. The
proposer rebuilt per-group attention metadata correctly, but
`ForwardContext.layer_slot_mapping` still supplied one shared slot mapping to
all draft layers. Under prefix caching, the draft attention groups have
different block tables, so all ShortConv states and Q/K/V inputs matched the
cache-off oracle while the first attention output diverged.

The production patch:

1. Separates target and draft hybrid cache ownership.
2. Builds compact real-token draft rows for MID, FINAL, accept, and reject.
3. Keeps canonical committed-prefix ShortConv state and uses the specialized
   qlen 1/2 Triton replay kernel.
4. Rebuilds metadata and slot mappings for every draft cache group.
5. Supplies per-layer slot mappings to `set_forward_context`.
6. Preserves independent prefix-cache restore/copy buffers for target and
   draft groups.

## Correctness and mechanism gates

| Gate | Result |
|---|---|
| Randomized ShortConv replay property test | 128/128 bitwise equal; accepted counts 1 and 2 have zero output/state error |
| Six draft attention layers, cache-off vs prefix, 432-token chunk | Bitwise equal after slot fix |
| CUDA graph A/A, SER1-64 | Identical output and proposal sequence; 18/45 accepted in both runs |
| Repeated identical prompt | Identical output and proposals; 19/33 accepted in both; second prompt effectively fully cached |
| Mixed two-request overlap | 0 errors; acceptance 40.0% and 56.8% |
| Growing SER6-64 | 0 errors; 120/260 accepted; 1.462 tokens/step |
| Prefix + graph SER8 greedy | 0 errors; 900/1517 accepted; 1.593 tokens/step |
| Prefix hit rate, SER8 | 77.9% final aggregate |
| Peak project VRAM | 7,774 MiB, below 10,000 MiB gate |
| Patch application | Dry-run PASS on clean v0.25.1 |
| Python syntax | `py_compile` PASS for all seven patched files |

The graph SER8 candidate diverges from the non-spec greedy text trajectory.
Graph A/A is exact, and prior layer/state audits localized this class to BF16
kernel-order changes in qlen-2 target verification rather than cache/state
corruption. This is recorded as a numerical trajectory caveat, not hidden as
byte identity.

## Decision

**Correctness-ready but performance-rejected after the REP30 campaign.**

The mechanism gate is above the pre-registered 1.35 tokens/step threshold,
prefix reuse is active, CUDA graph replay is deterministic, mixed requests
complete, and local VRAM remains below 10 GiB. A later anchor-interleaved
REP30 campaign nevertheless measured a roughly 4.5x TPOT regression for both
BF16 and FP8 drafts. Do not submit this image as a performance candidate.

## Submission package

- Patch:
  `submission/patches/vllm-0.25.1-lfm2-draft-prefix-v10.patch`
  - SHA256:
    `64df29959bce5f446ae26383fa81fcce836049b02305886b8217586bf6eb46de`
- Dockerfile:
  `submission/Dockerfile.fp8-v0251-draft350m-k1-prefix-v2`
- Compose:
  `submission/docker-compose.fp8-v0251-draft350m-k1-prefix-v2.yml`
- Image tag:
  `siconhoccode/lfm-serving:fp8-v0251-draft350m-k1-prefix-v2`

The local checkout has no target checkpoint under `submission/model`, so the
image must be built in the existing GCP checkout where that checkpoint is
already present. The 350M draft requires no rsync; the Dockerfile downloads
and pins it during build.

## Raw artifacts

- Graph A:
  `results/v0251_fp8_350m_draft_k1_prefix_prodmem_graph_20260730-045852`
- Graph B:
  `results/v0251_fp8_350m_draft_k1_prefix_prodmem_graph_20260730-050027`
- Graph SER8:
  `results/v0251_fp8_350m_draft_k1_prefix_prodmem_graph_20260730-050144`
- Graph SER8 trace:
  `results/draft_prefix_graph_ser8_greedy_trace_20260730.jsonl`
- Repeated prompt:
  `results/draft_prefix_repeat_trace_20260730.jsonl`
- Mixed overlap:
  `results/draft_prefix_slotfix_overlap2_trace_20260730.jsonl`
- Growing turns:
  `results/draft_prefix_slotfix_ser6_64_trace_20260730.jsonl`
- Non-spec prefix control:
  `results/v0251_fp8_control_prefix_prodmem_20260730-045650`
