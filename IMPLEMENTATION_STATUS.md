# HTIS-Context implementation status

Updated 2026-10-13. Working branch: `implementation/htis-context-phase-0-2`.
Starting research tip: `d918894e0803691c5acfef7a92060b6e1b4c2281`.
The starting working tree was clean. `main` and the original research/SignalRank
artifacts remain preserved. Changes are local and uncommitted; no push or merge.

## Current implementation

Phase 0 and Phase 1 are implemented and validated on a genuinely pretrained
SmolLM2-135M checkpoint. Phase 2 reference scope is implemented and its final
recorded pretrained example passed. See the evidence ledger for exact scope.

- `htis_context.adapter`: independent Llama eager forward using pretrained
  weights; FP32 only, original-position RoPE, causal slot order, binary masks,
  GQA without repeated KV groups, immutable input-state append semantics.
- `provenance`: separate immutable source/token/position/mask recipes,
  execution identities, and semantic-route types. External embeddings do not
  enter KV APIs. Actual embedding-model routing is not implemented.
- `state`: request-owned tensors, isolated snapshots, safe-tensor persistence
  and scope/provenance/integrity checks before restore. Original positions
  survive restart/reload. Approximate lineage cannot enter exact snapshots.
- `pages`: insertion-cached key extrema/value features; full, protected recent,
  sinks+recent, prompt-protected random, past-attention and current-query
  summary heuristics. Selection precedes actual attention matmul.
- `prefix_cache`: explicit exact-prefix lookup, independent complete entries,
  CPU copy accounting, bounded admission and deterministic LRU/LFU eviction.
  Similar chunks, modified ancestry, masks, positions, model/adapters,
  precision, scope or implementation lineage produce misses.
- `experiment`: real pretrained smoke factorial: reuse on/off x six policies,
  three authored sources, two timing repeats, eight teacher-forced steps and
  recorded greedy behavior. Dense reference logits are offline measurement
  targets, never runtime policy features.
- CLI: `download`, `inspect`, `dense`, `equivalence`, `phase2`. Model/checkpoint,
  workload, numerical tolerances and dependencies are pinned. Command records
  include actual output, exit codes, timestamps and source hashes.

## Evidence and verification

- Original research suite: **55 passed** before extension.
- Final combined suite with `HTIS_RUN_PRETRAINED=1`: **115 passed**, including
  actual pretrained equivalence and persisted snapshot reload.
- Standalone dense equivalence: **97 checks passed**, including cached versus
  uncached/upstream logits, multi-token append, multi-step teacher forcing,
  eight greedy steps, padding/batch isolation, shifted/boundary RoPE positions,
  physical GQA layout and incompatible identity rejection. FP32 tolerances:
  `atol=5e-5`, `rtol=1e-5`.
- Final Phase 2 run: **105 checks passed**, 72 trials, 576 recorded
  teacher-forced steps. Raw trials are losslessly compressed JSONL; the
  normalized JSON summary names and hashes the raw artifact.
- Maximum absolute dense-equivalence logit error was **4.9591064453125e-5**;
  all comparisons passed the predeclared absolute/relative tolerances.
- Final smoke means: full-cache active KV **4,638,720 bytes** versus
  **1,320,960 bytes** for bounded policies. Query-summary next-token KL
  **0.229736** and NLL increase **0.169907 nats/token**; its CPU forward time
  **158.45 ms/step** versus full **156.58 ms/step**. These are tiny-sample
  reference diagnostics, not task correctness or an efficiency claim. Independent
  CPU prefix snapshots and metadata remain additional storage.
- Paired reuse-on/off quality diagnostics were identical across all 36 pairs;
  maximum mean next-token KL difference **0.0**. Twenty-two evidence artifact
  hashes and all numerical-run model/test/workload source hashes were verified.
- Original simulator/held-out ablation commands ran successfully and retain
  their synthetic-only labeling. SignalRank: **4 tests passed** and Vite build
  passed. New Linux CI job is configured; remote CI has not been executed.

Evidence lives under `transformer-efficiency/results/phase0`, `phase1`, and
`phase2`. [RESEARCH_PLAN.md](RESEARCH_PLAN.md) and
[ADR 0001](transformer-efficiency/decisions/0001-reference-adapter.md) map gates.
The complete implementation file list is [CHANGED_FILES.json](transformer-efficiency/CHANGED_FILES.json);
executed commands and failures are in [COMMANDS_EXECUTED.md](transformer-efficiency/COMMANDS_EXECUTED.md).

## Failures, repairs and remaining boundaries

The initial command recorder could not print Node's Unicode test output under
Windows cp1252; UTF-8 output fixed it. Node child-process spawn and pytest
temporary directories were blocked by the filesystem/process sandbox; the
same tests passed with normal local access. These were execution-environment
failures, not upstream numerical failures.

A new random MHA fixture constructed weights inside `inference_mode`, where
PyTorch omits mutation counters. The fixture now uses `no_grad`; the adapter
explicitly rejects models lacking version counters rather than silently losing
the stale-weight guard. The final suite passes. A CLI invocation before editable
installation completed failed to import the package; the source path was set
explicitly and the command rerun. Failed command records are retained.
Editable packaging succeeded, but Windows denied launching the generated
`htis-context.exe` even with normal process access. The supported local command
is `.venv/Scripts/python.exe -m htis_context.cli`; this module path executes
all recorded pretrained runs. No ACLs or security settings were changed.

The first pretrained smoke run found nonzero next-token KL for every bounded
policy. Query-summary selection was slower than the full eager reference in
that run, despite using fewer physical KV vectors. This is retained as a
negative result, not converted into a speedup claim. The implementation was
tightened to fold query groups into attention rows, avoiding an implicit KV
broadcast/materialization dimension. Memory checks now run before attention
score allocation; they are tensor estimates, not certified allocator peaks.

Supported runtime scope: CPU FP32 reference; dense batched/multi-token append;
sparse batch=1 and one new token; one selected page set shared across KV groups
within each layer. Cold prefill remains dense. Exact cache entries are complete
prefixes and require explicit lookup, not an automatically deduplicated radix
trie. Prefix admission/restore hashes and copies have real costs. Stored CPU
bytes and active KV bytes are separate; Python/allocator and model/MLP/logit
activation overhead is excluded from the cache-tensor estimate.

No learned linear/MLP controller, HTIS attention/locality/content training,
counterfactual dataset pipeline, per-group adaptive budget, recoverable
in-request archive, recomputation fallback, runtime omission trigger, official
long-context benchmark, second-family pretrained validation, optimized dense
kernel comparison, GPU measurements, paper, or research UI exists yet.
The NumPy omission bound remains unchanged: a fixed-state local reference,
not final-answer correctness or repair for earlier approximate states.

The host has 16 GB RAM and an AMD integrated GPU; PyTorch CUDA is unavailable.
Small CPU experiments are feasible. GPU efficiency experiments are unmeasured,
not replaced with synthetic results. No paid compute was used. `npm ci`
reported five vulnerabilities in the preserved locked demo dependencies;
dependency upgrades were outside this transformer implementation scope.

## Exact next phase and continuation prompt

**Next: Phase 3, offline counterfactual labels and learned baseline importance.**

> Read IMPLEMENTATION_STATUS.md, CLAIM_EVIDENCE.json, RESEARCH_PLAN.md and
> transformer-efficiency/CODEX_BUILD_PROMPT.md completely. Preserve the working
> tree. Rerun the 115-test suite and pinned SmolLM2-135M equivalence report before
> modifying attention. Implement Phase 3 on the working Phase 2 adapter: offline
> single-page and multi-page cache interventions, local output damage and
> teacher-forced next-token KL labels, separate label/runtime-feature types,
> public-source checksums and model/state provenance, document/session splits
> plus a task-family holdout, and no-lookahead tests. Train and save linear and
> small MLP baselines before ablatable attention/locality/content fusion. Keep
> common protection and physical byte budgets; include prompt-protected random,
> recent, history and query-summary baselines. Record real training costs,
> held-out results, uncertainty and negative findings. Do not call the existing
> heuristics trained importance. Preserve exact prefix eligibility, original
> positions, GQA storage, snapshot isolation, and approximate lineage separation.
> Tiered archive recovery and runtime bounds remain Phase 4. Do not spend on
> paid compute, merge, deploy or claim speedups without measured evidence.
