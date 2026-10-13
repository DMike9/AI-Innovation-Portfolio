# HTIS-Context research plan

The implementation contract is [CODEX_BUILD_PROMPT.md](transformer-efficiency/CODEX_BUILD_PROMPT.md).
The architectural starting point is [HTIS_CONTEXT_ARCHITECTURE.md](transformer-efficiency/HTIS_CONTEXT_ARCHITECTURE.md).
Read [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) and
[CLAIM_EVIDENCE.json](CLAIM_EVIDENCE.json) before making claims or continuing.

## Acceptance gates

1. **Phase 0:** clean baseline or documented failures; preserve user changes;
   inspect hardware and dependencies; pin a pretrained model after reviewing
   its official card/license/config. Record real commands and artifacts.
2. **Phase 1:** independent exact adapter with approximation disabled; compare
   cached/uncached logits, teacher forcing and greedy generation to unchanged
   upstream. Validate original positions, RoPE, padding/masks, GQA, append,
   snapshots, isolation, model revision, and safe restart/reload.
3. **Phase 2:** physically selected attention inputs, bounded page allocations,
   exact prefix lookup/invalidation and deterministic LRU/LFU; record costs
   and quality changes on real weights. Full capacity must remain equivalent.
4. **Phase 3:** offline interventions for single/multiple pages; local output
   damage and next-token KL labels separate from runtime features. Hold out
   documents/sessions and a task family. Train linear and MLP baselines, then
   ablatable attention/locality/content branches. No future-token features.
5. **Phase 4:** bounded CPU archive with same-lineage page recovery, transfer
   counters and measured latency; missing/stale-page failures. Integrate the
   local omission reference with complete omitted-page coverage and correct
   current-snapshot fallback semantics. Never declare dense-history recovery
   after approximate steps without valid-prefix recomputation.
6. **Phase 5:** controlled quality/resource experiments, prefix reuse x policy
   factorial, composed strong baselines, benchmark revisions, multiple seeds,
   paired uncertainty and optimized dense kernels. Use available local hardware
   or request approval before paid resources. A second model family is needed
   for generalization claims.
7. **Phase 6:** build the paper and evidence explorer from measured records.
   Refresh primary literature before novelty assertions; retain negative results.

## Reproducibility and scientific boundaries

- Preserve the legacy `htis` simulator, original `src/filename.py`, historical
  artifacts, and separate SignalRank applied demo.
- Keep source recipes, semantic routing vectors, reusable KV, and active
  attention tensors separately typed. Semantic similarity cannot authorize KV.
- Exact reuse includes model/checkpoint, tokenizer/template, ordered ancestry,
  positions, masks, RoPE, adapters, precision/format, scope and implementation
  lineage. Never insert approximate states into the exact namespace.
- Declare tolerances before model validation. Report unsupported cases and
  numerical failures; do not substitute random fixtures for pretrained runs.
- Command records retain output, exit codes, timestamps, configurations and
  source hashes. Weights stay in the ignored local cache, never Git.
- CPU tensor byte estimates and eager-reference timings do not establish GPU
  memory savings, throughput gains or long-context task quality.

The phase/module/acceptance mapping and model choice rationale are in
[ADR 0001](transformer-efficiency/decisions/0001-reference-adapter.md).

## Selected continuation: Phase 3 only

Reuse the verified Phase 0–2 CPU implementation committed at `0e134c2`.
The selected experiment is pinned in `transformer-efficiency/configs/phase3-small.json`.
Use twelve checksummed public book extracts with six train, three validation,
and three test documents. Keep all sessions from each source in the same split;
reserve the compare task family for test only. Stress cases are explicitly
authored instructions or a controlled value perturbation on real pretrained KV.

Generate labels through the existing adapter/policy interface: single-page and
two-page omissions at three sampled layers, local attention-output damage,
next-token KL, and two-step teacher-forced KL. Write runtime features and
offline targets to physically separate artifacts. Keep source/position/state
hashes and intervention costs. Future continuations belong only to the offline
label path; they are not validated task answers.

Fit linear and MLP baselines first, then attention/locality/content fusion and
branch/query/value ablations with two seeds. Select checkpoints and residual
calibration only on validation documents. The first locality adaptation is
a learned convolution over original adjacent **pages**, with missing-neighbor
masks, not a claim of a token-level implementation. Evaluate held-out scoring
separately from model-output KL at equal protection and physical KV budgets.
Charge feature and learned-weight storage/scoring overhead separately.

Add budgeted exploration among still available pages and recollect diagnostic
labels under approximate survivor states on training documents. Do not recover
evicted pages or add archive/fetch behavior in this run; that is Phase 4.
Keep negative outcomes and small-sample/layer-extrapolation limitations.
Stop after validating this phase and updating the existing status/evidence ledger.

Phase 3 checkpoint outcome: VERIFIED COMPLETE within the small pretrained CPU
scope, with 16 saved controllers, 558 intervention labels and 528 held-out
runtime checks. The candidate did not outperform protected recency in the
reported four-page comparison. Retain that negative result. The next unfinished
phase is Phase 4; no archive, adaptive fetch or cost-aware trigger was added here.

## Selected continuation: Phase 4 only

Continue the committed Phase 3 checkpoint `28a959d`; preserve its weights,
splits, labels and negative findings. The frozen CPU study is specified in
`transformer-efficiency/configs/phase4-small.json`: nine held-out public-source
or explicitly authored/perturbed workloads, eleven arms, two repeats and the
same four-page active budget and protection rules. No new training or threshold
selection uses these held-out outcomes. This is an architecture validation,
not Phase 5's official benchmark or optimized hardware study.

Extend the existing RequestState/LayerKV and pretrained adapter with a separate
bounded CPU archive. Preserve original source slots and already-rotated KV;
bind archive restoration to the execution identity and complete current state,
including ordered ancestry, positions, masks and approximate lineage. Admission
and eviction use deterministic FIFO. Retain discarded-page summaries within a
separate index budget, but declare their tensors unrecoverable. Exact-prefix
snapshots remain independent from request routing metadata.

Use causal current-query summaries, previous selected attention and the frozen
Phase 3 estimator. Refresh on task/query/rank/risk/uncertainty/interval triggers;
use measured score/copy costs and conservative priors to select a cheap route.
Apply cooldown to retention changes. Compare protected recency with/without
recovery, attention history with recovery, HTIS with/without recovery, a causal
deterministic baseline, bound/fallback ablations, and costly/unavailable archive
arms. Costly access is a declared decision-model prior; actual CPU copy times
remain measured separately and are not fabricated GPU transfers.

The optional value-sensitive bound covers every visible omitted page, including
summarized pages whose tensors were discarded. Measure local tightness where
exact-current-state streaming fallback executes. A missing summary invalidates
the stated bound; missing KV prevents exact fallback. Bound tolerance is local
per attention head, never final-answer correctness. Streaming over original
archive pages has explicit memory/time budgets and cannot repair prior drift.

Run deterministic unit tests and genuine pretrained integration, retain actual
failure records and repairs, emit raw decisions/transfers/quality measurements,
verify artifact hashes and update the existing status and claim/evidence ledger.
Stop at Phase 4. Recommend Phase 5 only after this checkpoint; do not run it.

The measured Phase 4 outcome is negative overall: mean KL was 0.235721 for
protected recency without recovery, 0.313246 for simple query recovery and
0.417444 for frozen HTIS recovery. The architecture works, but conditional
recovery did not rescue this learned ranking or establish efficiency. Bound
fallback nearly reproduced the dense trajectory in this run at high CPU cost;
one latency-limited call explicitly left tolerance unmet. Preserve raw events,
44,271 fixed-state bound comparisons, all negative examples and prior findings.
The next scientific phase is Phase 5, with preregistered stronger composed
baselines and cost profiling before more controller complexity. Do not start
Phase 5 automatically or spend money on compute.

## Selected continuation: Phase 5 only

The user explicitly authorized Phase 5 after committing Phase 4 at `727c91c`.
Preserve its 93 hashes, trained Phase 3 weights and negative results. Stop before
Phase 6. No paid compute, remote provisioning, push, merge or deployment.

Profile the existing pretrained path before controller changes. Keep actual
uninstrumented wall times separate from nested spans/cProfile/native operator
diagnostics. Compare unchanged simple/learned/recovery policies and upstream
SDPA. CPU reference overhead is not inherent algorithmic overhead or GPU evidence.

Freeze a compatible short pilot before final evaluation: four validation cases,
eight held-out cases, complete SQuAD v1.1 paragraphs selected by stable hashes,
and labeled authored probes. Generate six greedy tokens and retain every raw
output. Measure normalized answer EM/F1 and independently teacher-forced KL/NLL.
No silent benchmark truncation, answer mining or final-test threshold selection.
The model supports 8,192 positions; the pilot uses at most 256 total tokens for
local CPU scope. Official long-context evaluation and a second family remain
unexecuted and cannot support generalization claims.

Compare matched four-page physical budgets/protection, explicitly separating
pure window, unbounded full attention, and extra archive/index/prefix storage.
Choose the strongest simple retention candidate on validation before crossing
exact reuse × learned HTIS × recovery in actual cold/warm request timings.
Run existing trained branch/query/value ablations, fixed-budget sensitivity,
cooldown/refresh/trigger/bound-frequency and insertion-feature-cache diagnostics.
No adaptive-budget component exists to ablate. Never claim a heuristic is a
faithful H2O/SnapKV/Quest reproduction.

Use document/session paired bootstrap, grouping the positional variants of the
same authored fact together. Seeds are averaged within a source group; tokens
are not independent statistical observations. Report descriptive p50/p95,
per-workload/seed outcomes, dominated policies and actual artifact-based figures.
Select an evidence-backed B/C/D conclusion if coordination does not establish
a superior quality/latency/memory frontier. CUDA execution is blocked locally;
test runner gating and synchronization, clearly separating unexecuted GPU paths.

Phase 5 checkpoint outcome: the narrow CPU pilot completed 196 quality trials,
32 actual factorial requests, 204 checks, 32 SDPA equivalence comparisons and
186 tests. The frozen controller selected recency on validation. HTIS improved
over recency on the new pilot but simple query-summary dominated it on the
F1/latency/storage frontier; NLL gives a separate tradeoff. Recency recovery
improved F1 on 3/8 cases at high cost, HTIS recovery on 0/8. Exact prefix reuse
provided measured CPU request savings and preserved output tokens; coordination
did not establish benefit beyond independent composition. Existing Phase 3/4
negative evidence remains unchanged.

Decision B with C on primary F1 and D for broader claims. Keep Phase 6 NOT STARTED.
Next: a separately versioned Phase 5 extension with stronger query/random controls
at 4/8/16 pages, larger paired sessions, supported longer contexts and compatible
local GPU correctness gates before timing. Revisit first-compaction scorer and
archive-maintenance costs before adding another learned controller. Do not bypass
checkpoint identity to move trained weights onto a different pretrained model.
The exact continuation prompt and discriminating experiments are in
`transformer-efficiency/results/phase5/FINDINGS.md`.
