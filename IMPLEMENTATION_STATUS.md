# HTIS-Context implementation status

Updated 2026-10-13. Working branch: `implementation/htis-context-phase-0-2`.
Starting research tip: `d918894e0803691c5acfef7a92060b6e1b4c2281`.
The Phase 0–2 implementation is committed at `0e134c2`. The continuation
started from a clean working tree. Phase 3 changes are local and uncommitted;
`main`, SignalRank, the original HTIS artifact, and existing results are preserved.
No push, merge, deployment or history rewriting was performed in this continuation.

## Current implementation

### Continuation checkpoint (2026-10-13)

The prior implementation is committed at `0e134c2`; the current working tree
was clean at continuation start. No research ref was reimported. All 22 prior
artifact hashes, numerical-run source/configuration hashes and installed
dependency versions matched. Existing Phase 0–2 evidence is reused unchanged.

| Phase | Status | Supporting implementation/evidence |
|---|---|---|
| 0 | VERIFIED COMPLETE | Existing audit/configuration/ADR; `results/phase0/environment.json` and checkpoint manifest |
| 1 | VERIFIED COMPLETE | Existing adapter/provenance/state; `results/phase1/equivalence-final.json`, 97 checks |
| 2 | VERIFIED COMPLETE | Existing pages/prefix-cache/policies; `results/phase2/pretrained-final.json`, 105 checks and 72 trials; combined 115-test record |
| 3 | VERIFIED COMPLETE | `public_corpus`, `runtime_features`, `offline_labels`, `counterfactual_data`, `learned`, `training`, `phase3_evaluation`; `results/phase3` trace/training/evaluation records and 61 targeted tests |
| 4 | NOT STARTED | Recoverable in-request archive and runtime bound remain proposed |
| 5 | NOT STARTED | Controlled benchmark/hardware study remains proposed |
| 6 | NOT STARTED | Paper/explorer remains proposed |

Completion statuses use the staged brief's exit gates and the explicitly
documented CPU reference scope. Phase 3 extends existing policy/cache
interfaces; the dense adapter and cache engine are not being replaced.

### Phase 3 implementation and evidence

The selected phase is **VERIFIED COMPLETE** as a small CPU pretrained-model
research implementation, not a demonstration of general policy superiority.
No later phase was implemented or started.

- Public corpus: twelve checksummed book-body extracts, 40,345 UTF-8 bytes,
  six training documents, three validation documents and three test documents.
  Acquisition URLs, upstream response hashes, extract hashes and preparation
  rules are in `transformer-efficiency/data/phase3-public-v1/corpus.json`.
- Counterfactual data: **31 real pretrained traces**, **1,302 runtime-feature
  rows**, **558 interventions** (465 single-page, 93 two-page), sampled layers
  0/14/29, local output perturbation, next-token KL and two-step mean KL.
  Runtime features and offline labels are separate compressed artifacts with
  source, token, original-position, state, model and computation provenance.
- Splits: **180 train / 90 validation / 195 test single-page labels**, grouped
  by source document/session. Compare tasks are test-only. Delayed relevance,
  contradictory updates and task switches are authored stress instructions
  appended to public passages. The value-outlier case scales one real KV value
  by 100 and is explicitly an artificial supplied-state perturbation.
- Trained outputs: **16 saved safe-tensor checkpoints**, linear and MLP first,
  then HTIS attention/locality/content fusion, plus no-attention, no-locality,
  no-content, no-query and no-value ablations, seeds 17/29. Normalization,
  checkpoint selection and empirical residual calibration use train/validation
  only. Trainable parameters range from 40 to 985; model artifacts record
  parameter counts, source and weight hashes, curves, configurations and costs.
- Runtime validation: **528 held-out real-model trials/checks**, two physical
  page budgets, equal protection, four heuristic baselines, all trained seeds
  and HTIS exploration arms. Raw trials include KL, scoring/forward time,
  active KV/metadata bytes, controller/feature tensor storage and available
  versus missing original positions. All budget and lineage checks passed.
- Survivor-bias diagnostic: budgeted exploration replaces one available free
  page; training-document recollection uses previous selected attention only.
  Separate recollection feature/label files are checksummed and never used to
  tune this run's models. No removed KV is recovered; Phase 4 remains untouched.
- Final targeted suite: **61 passed**, including real pretrained capture and
  intervention, existing page-cache regressions, future-independence and original
  adjacency tests, all model gradient updates, actual checkpoint reload and
  validation-MSE reproduction, scope/revision mismatch and corruption rejection.
  Phase 0–2's expensive experiments were reused by hash rather than rerun.

Trace generation took **191.77 seconds**, model training **12.07 seconds**,
and held-out runtime evaluation **203.55 seconds** on CPU. These are actual
session costs, not throughput or GPU performance claims.

### Phase 3 negative findings and scope

At four pages, protected recency had mean next-token KL **0.260575** versus
**0.319004** for HTIS seed 17 and **0.312487** for seed 29. Seed 17's mean
forward/scoring times were **190.21 / 61.87 ms**, versus recency's
**126.17 / 6.00 ms**. The explored seed-17 arm worsened mean KL to **0.398958**.
These tiny-sample CPU comparisons do not establish superiority or a speedup.

Scoring results were seed-sensitive: HTIS seen-task mean group Spearman was
**0.455556 / 0.072222** for seeds 17/29. The maximum absolute two-page KL
nonadditivity residual was **0.00231511**; summing learned page risks is not
exact joint-risk prediction. Validation residual intervals are empirical
diagnostics, not probability or correctness guarantees.

The locality branch is a genuine learned **page-level** convolution over
original adjacent page IDs, with gap masks. Token-level convolution is not
implemented. Labels sample three layers; all-layer policy evaluation explicitly
extrapolates elsewhere. The label continuations are teacher-forcing inputs,
not validated answers. Feature storage and controller costs are charged, but
full allocator/kernel-scratch/RAM/GPU peaks remain unmeasured. No correctness
failure occurred in the selected phase tests; quality/cost regressions are
retained rather than hidden or repaired through test-set tuning.

Reused unchanged: dense adapter, cache objects, page summaries, exact-prefix
provenance rules, baseline policies, checkpoint loader, prior model/configuration,
existing CI, all prior artifacts, SignalRank and original HTIS. New code extends
the existing policy signature; no earlier-phase engine was rebuilt.

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

No per-group adaptive budget, recoverable
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

**Next unfinished phase: Phase 4, recoverable memory and cost-aware routing.**

> Continue the existing implementation from `0e134c2` plus the local Phase 3
> checkpoint. Read IMPLEMENTATION_STATUS.md, RESEARCH_PLAN.md, CLAIM_EVIDENCE.json
> and CODEX_BUILD_PROMPT.md; preserve all working files and trained artifacts.
> Verify relevant hashes and select Phase 4 only. Extend the existing RequestState,
> LayerKV, page policies and provenance rules with bounded same-lineage CPU archive
> offload/reload, transfer/copy counters and measured fetch costs. Keep immutable
> dense prefixes separate from approximate trajectories. Integrate the local
> omission reference only with complete omitted-page coverage; label fallback
> as exact-current-snapshot behavior, never historical dense recovery. Add tests
> for stale lineage, unavailable pages, task switches, thrashing, budget ceilings
> and fallback. Use the existing saved controller weights; retain the Phase 3
> finding that protected recency was better and cheaper in this small study.
> Implement cost-aware refresh/cheap-policy routing and a recoverable pretrained
> example, record all costs and failures, update the same status/evidence files,
> and stop at the Phase 4 checkpoint. Do not start Phase 5, redeploy, merge, push,
> rewrite history or incur paid compute without authorization.
