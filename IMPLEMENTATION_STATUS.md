# HTIS-Context implementation status

Updated 2026-10-13. Working branch: `implementation/htis-context-phase-0-2`.
Starting research tip: `d918894e0803691c5acfef7a92060b6e1b4c2281`.
The Phase 0–2 implementation is committed at `0e134c2`. The continuation
started from a clean working tree. Phase 3 was subsequently committed at `28a959d`.
The Phase 4 continuation was committed at `727c91c`. Phase 5 started clean at that checkpoint;
`main`, SignalRank, the original HTIS artifact, and existing results are preserved.
No push, merge, deployment or history rewriting was performed in this continuation.

## Current implementation

### Phase 5 continuation (2026-10-13)

The Phase 4 checkpoint's 93 recorded artifact hashes were reverified before
continuing. The Ryzen 7 PRO 250 host has 16 GB RAM and integrated Radeon graphics;
Torch 2.9.1 is CPU-only. CUDA benchmarking is blocked on local hardware/build.
The pinned SmolLM2-135M FP32 weights and all Phase 0–4 numerical sources,
checkpoints, evidence, SignalRank and the original HTIS artifact are preserved.

Existing-path nested spans, cProfile and a separate native Torch profile are
recorded under `transformer-efficiency/results/phase5/profile-v1`. Uninstrumented
decode samples remain distinct from profiling overhead. Controller bookkeeping,
feature extraction, copying/checksums and bound/fallback work dominate much of
the extra cost. Optional current-query/interval and benefit-cost wrappers extend
existing policies; they do not retrain weights or compute dense attention to score.

The frozen `phase5-cpu-v2.json` manifest specifies four validation and eight
held-out short cases: complete, untruncated SQuAD v1.1 paragraphs and explicitly
authored memory probes. This is a tiny CPU pilot, not an official long-context
benchmark. Validation chooses controller settings and the strongest simple
baseline before test execution. The first manifest/run is retained, including
its accounting/design limitations.

Phase 5 is **VERIFIED COMPLETE only for the short FP32 CPU pilot**. The recorded
checkpoint audit verified **161 artifact hashes**, including all 93 prior hashes.
**GPU study BLOCKED / NOT EXECUTED; official
long-context benchmarks and cross-family generalization NOT EXECUTED. Phase 6
NOT STARTED.** See [Phase 5 findings and continuation](transformer-efficiency/results/phase5/FINDINGS.md),
[final analysis](transformer-efficiency/results/phase5/analysis-v2/summary.json),
[comparison table](transformer-efficiency/results/phase5/analysis-v2/comparison.md)
and [reproduction runbook](transformer-efficiency/results/phase5/RUNBOOK.md).

Executed **48 final validation trials/checks**, **196 held-out quality trials**,
**32 actual cold/warm requests**, **204 runtime checks**, **32 upstream SDPA
common-token comparisons** (max absolute error 3.62396e-5), and **186 passing
tests**. Held-out execution took 567.36 seconds, with runtime sources unchanged
and no concurrent test/profiling jobs. Synthetic fixtures remain unit tests;
these experiment outputs use genuinely pretrained weights and unchanged learned
checkpoints. Final analysis pairs stochastic/checkpoint seeds, averages source
clusters for bootstrap and exports inspected PNG/SVG figures. Initial analysis
and its labeling/pairing defects are preserved rather than rewriting results.

| Core policy | Free F1 | Teacher KL | CPU decode ms |
|---|---:|---:|---:|
| Full reference | 0.2542 | 0 | 131.22 |
| Upstream SDPA, full KV | 0.2542 | separate equivalence checks | 99.04 |
| Protected recency | 0.0500 | 0.7380 | 134.16 |
| Query-summary | 0.1333 | 0.5614 | 139.64 |
| HTIS | 0.1104 | 0.6226 | 148.74 |
| HTIS recovery | 0.1104 | 0.6247 | 523.19 |
| Recency/query recovery | 0.2218 | 0.4779 | 373.59 |
| Bound/fallback | 0.2542 | near numerical zero | 942.90 |

All core whole-output EM scores were zero under the frozen six-token protocol;
do not relabel low divergence as answer accuracy. Recency recovery improved F1
on 3/8 cases; HTIS recovery on 0/8. HTIS improves over recency here but is dominated
by simple query-summary on F1/latency/total storage. Its lower target NLL remains
a separate nondominated tradeoff; there is no universal quality conclusion.
Conditional/cost gates failed to improve the overall measured frontier. Retained
feature caching saved 33.87 ms/step in the matched two-case diagnostic. Larger
fixed-budget HTIS performed better on that tiny subset, without sufficiently
strong eight-page controls to establish a general result.

Exact reuse preserved every repeated output and reduced actual mean simple
request end-to-end latency from 1618.11 to 1004.71 ms; warm HTIS was 1202.50 ms,
warm HTIS recovery 3787.53 ms. Prefix admission and up to 11,525,602 extra stored
bytes are charged separately. Core active KV is at most 1,474,560 bytes;
HTIS recovery total cache/index/scorer storage reaches 17,168,707 versus full
reference 12,804,960. These are CPU tensors plus stated estimates, not allocator/
RSS peaks or GPU memory claims. Base model weights are separate 538,060,032 bytes.

There were 16,470 measured fixed-state per-head bound comparisons with zero
observed violations. Fallback does not certify final answers or repair earlier
approximation. Six source/session bootstrap groups, two timing repeats and two
ablation cases give exploratory uncertainty only. The decision is **B**, with
**C** on the primary F1 frontier and **D** for broader systems claims.
Continue a separately preregistered **Phase 5 extension**—stronger composed
query/random/HTIS budget controls, longer supported contexts and compatible
local GPU acceptance—before a Phase 6 manuscript or explorer. No paid compute,
commit, push, merge, deployment or history rewriting occurred.

### Phase 4 continuation (2026-10-13)

Phase 4 is **VERIFIED COMPLETE within the small CPU reference scope**.
The final audit verified all 70 prior artifact hashes, 93 total evidence hashes,
unchanged benchmark runtime sources, matching incremental/final raw trials,
and preservation of earlier results, trained weights, SignalRank and original HTIS.
The repaired complete suite passed **165 tests**, including the unchanged
upstream-equivalence path, saved Phase 3 weights/validation diagnostics and
29 new recovery tests. See `results/phase4/tests-final-current.json`.

`recovery.py` extends the existing RequestState/LayerKV and adapter. It owns
bounded, immutable CPU pages indexed by layer and original page ID, with
checksummed tensors, execution/access identity, offload ancestry and a chained
computation digest. Functional request forks share private immutable archive
records; every reload copies tensors. Admission/eviction is deterministic FIFO.
Active, archived, discarded and unavailable-before-archive states are explicit.
Missing tensors cannot be recomputed by this implementation.

The RecoveryPolicy wraps existing policies. Selection physically assembles only
chosen pages before the adapter attention matmul. Current task/query, query-bound
rank changes, estimated omission risk, validation residual uncertainty and
intervals trigger refresh. Measured scorer/copy cost estimates can select a cheap
recency route; cooldown limits replacements. The existing HTIS weights, training
normalizer and validation calibration are frozen. No runtime future targets or
dense teacher weights are used for selection.

Optional adaptive retrieval invokes the unchanged NumPy fixed-state bound with
float64 exact-state summaries covering every visible omitted page. Fetch priority
includes both possible mass and value distance. Archived pages can be swapped
within the active budget; insufficient resources leave a recorded unmet
tolerance. Missing summaries invalidate coverage. Discarded tensors can retain
summaries but prevent exact fallback. Blockwise online softmax reads one archived
page at a time with explicit memory/time admission checks; it reproduces attention
on the **current supplied KV snapshot**, never an earlier dense generation.
Approximate lineage stays approximate even after fallback.

The four-page active capacity, 8 MiB archive, 4 MiB index and 64 MiB combined known
working estimate are configured independently. Physical tensor bytes, index
estimates, copies, scorer/controller/integrity/bound/fallback time and divergence
are recorded. These are CPU residency and cost measurements; allocator/RSS peaks,
GPU transfers/residency savings, optimized-kernel speedups and answer correctness
are unmeasured. In-request storage estimates exclude base weights, offline teacher
fixtures and independent exact-prefix-cache entries (charged in their own events).
Latency limits are admission/check-boundary controls, not hard real-time guarantees.

Failures remain recorded: initial fixture assumptions, temporary-directory access,
inference-mode model construction, and the ninth-workload inference-tensor mutation.
The latter exposed missing incremental experiment persistence; the runner now
writes completed trials and workload/progress checkpoints before final aggregation.
It does not invent the lost intermediate metrics. Further repairs discard routing
metadata from dense prefix snapshots, preserve original scorer source contracts
outside pinned extension blocks, cache immutable byte totals/summaries, and apply
cooldown to value-sensitive adaptive fetching.

#### Measured Phase 4 outcome

The frozen study completed **198 trials / 1,804 checks**, nine workloads,
eleven arms, two repeats, maximum ten teacher-forced tokens per trial.
Every active/archive-budget and measured local-bound coverage check passed.
Actual comparison cost was **974.63 seconds** on CPU. The final artifacts are
`results/phase4/evaluation-v3/report.json` and its lossless compressed raw trials;
completed-trial checkpoints are also retained. No final held-out tuning occurred.

| Policy | Mean KL to dense | Mean CPU forward ms |
|---|---:|---:|
| Protected recency, no recovery | 0.235721 | 156.11 |
| Protected recency + query-summary recovery | 0.313246 | 433.60 |
| Attention history + recovery available | 0.274597 | 434.40 |
| Frozen HTIS seed 17, no recovery | 0.312681 | 201.77 |
| Frozen HTIS seed 17 + recovery | 0.417444 | 600.36 |
| Deterministic cycling recovery | 0.502561 | 409.42 |
| Query recovery + adaptive local bound | 0.386856 | 762.60 |
| Query recovery + bound/blockwise fallback | 0.000000548 | 1103.86 |
| Expensive-access prior, cheap recency route | 0.235721 | 400.66 |
| Unavailable archive, bound enabled | 0.303614 | 620.84 |
| HTIS always refresh, no cooldown ablation | 0.407528 | 597.01 |

**Recovery did not improve the overall tradeoff.** Simple query recovery improved
two of nine workloads and worsened seven. The controlled value-outlier mean KL
improved from 0.800582 to 0.727474; the references-to-inactive-context case
worsened from 0.038891 to 0.473202. HTIS recovery improved only one workload
relative to HTIS without recovery, and was better than simple query recovery in
only two workloads. Its recent-update case worsened from 0.245142 to 1.027633.
The existing Phase 3 negative finding remains unchanged and is reinforced here.

An additional offline first-query check validates the outlier description:
page 1 at layer 14 had mean attention mass **1.925%**, maximum **6.245%** across
nine query heads, versus uniform page fraction **8.081%**. Scaling its real
pretrained values by 100 increased mean value norm from **5.379 to 537.936** and
local omission damage from **0.086582 to 8.898569**. This artificial supplied-state
stress is separately labeled and is not natural model quality. The diagnostic
never feeds runtime selection: `diagnostics/outlier-first-query.json`.

The attention-history arm performed **zero reloads**: frozen archive attention
history did not displace active survivors. HTIS conditional refresh executed
4,823 of 4,920 layer decisions versus 4,920 for always refresh; it did not establish
a useful latency/quality advantage. Scoring archived candidates cost HTIS
**155.90 ms/step**; archive/controller work **334.13 ms/step**, with integrity work
**98.24 ms/step** separately. Simple recovery had 1,434 reloads and 7,374 offloads;
HTIS had 1,349 / 7,289. Actual CPU copy latencies remain separate from the declared
expensive-access prior. That prior prevented all fetches/scoring and reproduced
recency's KL exactly, but archive maintenance still increased latency substantially.

All arms had the same active ceiling **1,474,560 KV bytes**. Recovery archive
storage peaked at **5,421,600 bytes**; index estimates at about **2.32 MB**;
in-request cache/index/scorer storage at **9,377,856 bytes**. The combined known
working estimate remained below **14,361,887 bytes**, versus a 64 MiB ceiling.
The independent exact-prefix entry stored another 4,930,560 KV bytes plus metadata,
with a measured **18.03 ms** restore and identical-state hash. Archival did not
eliminate memory and these CPU measurements establish no GPU savings.

Bound diagnostics contain **44,271 query-head comparisons**, with **zero observed
bound violations** under complete supplied-state coverage. Median bound/error
ratio was **45.25**, with extremely large ratios when actual error was tiny.
The adaptive-only arm failed its 0.1 local tolerance in all 4,920 layer calls;
its bound cost was **291.46 ms/step** and it added 2,924 reloads without aggregate
quality benefit. Fallback executed 4,919 times; one call stopped on its latency
budget (task-family-shift, repeat 1, step 3, layer 25). It retained an explicit
unmet tolerance and left the trajectory approximate. With no archive, 4,920
fallback attempts were unavailable despite complete retained summaries.

Artifacts include 131,388 recovery event rows, 44,280 controller decisions,
bound-tightness records, paired comparisons and negative examples. The copy
counts include fallback reads. Archive lookup hits count only available indexed
pages; unavailable fallback is tracked separately, not hidden as a semantic hit.

Remaining prototype costs include flat metadata scans, Python/tensor bookkeeping,
whole-archive integrity hashes and float64 omission summaries retained even in
no-bound recovery arms. These observations support profiling and stronger simple
baselines, not adding a more complex learned controller or claiming a speedup.
Disk persistence of approximate archives and valid-prefix recomputation are not
implemented. Phase 5/6 have not begun.

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
| 4 | VERIFIED COMPLETE | Bounded CPU archive, controller, adaptive local bound and blockwise fallback; 165 tests, 198 trials / 1,804 checks and 93 evidence hashes |
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

No per-group adaptive budget, provenance-correct prefix recomputation, official
long-context benchmark, second-family pretrained validation, optimized dense
kernel comparison, GPU measurements, paper, or research UI exists yet.
Phase 4 adds an in-request archive and an optional runtime trigger using the
unchanged NumPy omission reference. Its scope remains fixed-state local output
error, not final-answer correctness or repair for earlier approximate states.

The host has 16 GB RAM and an AMD integrated GPU; PyTorch CUDA is unavailable.
Small CPU experiments are feasible. GPU efficiency experiments are unmeasured,
not replaced with synthetic results. No paid compute was used. `npm ci`
reported five vulnerabilities in the preserved locked demo dependencies;
dependency upgrades were outside this transformer implementation scope.

## Exact next phase and continuation prompt

**Next unfinished phase: Phase 5, controlled model-quality and resource experiments.**
Recommendation: profile archive/index/integrity/scorer/bound overhead and compare
stronger simple compositions before increasing controller complexity. Preserve
Phase 4's finding that protected recency was better and cheaper overall. Use
new preregistered evaluation data for later tuning; do not tune on these outcomes.
This session stops at Phase 4 and has not executed Phase 5 or Phase 6.

> Continue HTIS-Context with Phase 5 only, from `28a959d` plus the preserved local
> Phase 4 checkpoint (or its subsequent user commit). Read IMPLEMENTATION_STATUS.md,
> RESEARCH_PLAN.md, CLAIM_EVIDENCE.json, CODEX_BUILD_PROMPT.md and the Phase 4
> report/checkpoint. Inspect AGENTS.md and the working tree; preserve all edits,
> trained weights and earlier evidence. Verify the 93 recorded artifact hashes
> and relevant runtime source/configuration hashes. Retain the negative results:
> recency KL 0.235721 versus HTIS recovery 0.417444, CPU forward 156.11 versus
> 600.36 ms; loose costly bounds and one latency-limited fallback. Do not retune
> on the existing held-out runs. Profile the current implementation and compose
> strong exact-prefix/recency/query-recovery baselines before more learned
> complexity. Preregister quality tolerances and new model/task/dataset revisions.
> Run the brief's reuse x retention factorial with no-recovery/no-bound ablations,
> matched optimized dense kernels, available local hardware, repeated synchronized
> timings where applicable, and document-level uncertainty. Keep CPU, archive,
> index, work buffers and GPU residency separate; no GPU claim without measurement.
> A second validated pretrained family is required for generalization claims.
> Record actual outcomes/failures and update the same status/evidence files.
> Do not spend on paid compute without approval, reset, merge, push, deploy,
> backdate, or begin Phase 6 automatically. Stop at the Phase 5 checkpoint or
> document the concrete resource/correctness blocker.
