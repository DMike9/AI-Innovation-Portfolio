# HTIS-Context: Codex implementation brief

Use this brief in the local `DMike9/AI-Innovation-Portfolio` repository. The research upgrade is on `research/htis-kv-importance-benchmarks` unless it has since been merged. This is a staged implementation request, not a request for a nicer README alone.

## Master instruction

Act as a research engineer working on transformer inference, model-integrated KV memory, and reproducible systems experiments. Implement HTIS-Context: Task-Conditioned Importance Control for Reusable Transformer Memory.

The user's original concept is hierarchical importance scoring plus caching prompts/context to avoid repeated or unnecessary transformer work. Develop that concept into a real, testable inference-memory architecture. Preserve the original HTIS artifact and existing HTIS-KV simulator. Keep SignalRank as a separate legacy applied demo, not the flagship and not evidence about neural attention.

First read all applicable AGENTS.md instructions, the root README, transformer-efficiency/RESEARCH.md, transformer-efficiency/HTIS_CONTEXT_ARCHITECTURE.md, the current htis package and tests, and the existing workflows. Treat the architecture document as the starting specification, not as proof that its proposed features exist. Inspect the actual working tree, branch, dependency versions, hardware, available checkpoints and test results. Do not overwrite uncommitted user changes, reset the branch, force push, or backdate commits. If the research branch is absent locally, inspect remote refs and integrate it non-destructively rather than inventing its contents.

The target is an original, technically defensible implementation and a credible research portfolio. Do not claim global novelty merely because we name a method. The specific contribution must survive comparisons with close prior art and composed baselines.

## Architecture to implement

Separate four objects: source text/token IDs, semantic routing embeddings, reusable model KV states, and the GPU active KV working set. Embeddings never substitute for the original content or model states. Exact prefix reuse is distinct from approximate non-prefix fusion and from output caching.

Implement two timescales:

1. Across requests: reuse compatible exact prefix states and make dependency-aware cache admission/eviction decisions based on measured reuse value and recomputation/fetch cost.
2. Within a request: score and allocate active KV pages by current task/query, predicted output degradation, uncertainty, and actual memory/transfer/compute cost.

The hierarchy is prefix/session -> segment -> page -> layer/KV-head group. Support shared K/V in grouped-query attention without materializing one KV copy per query head. Preserve original token positions, masks and RoPE semantics through compaction/offload/reload.

Develop the original HTIS idea into three ablatable branches: observed-attention statistics; a lightweight locality convolution on original adjacent positions; and a content MLP using projected K/V, value-sensitive features and metadata. Use task/query- and layer/group-conditioned fusion. Start with a trained linear model and MLP baseline before adding complexity. Cache features computed at prefill/insertion. Do not compute full attention first just to decide what not to attend to.

Importance means estimated conditional damage or lost utility from an action, not a claim that attention weights explain the final answer. Train using offline counterfactual page-removal perturbations, teacher-forced logit divergence, and measured action costs. Future utility may be an offline label but never a runtime feature. Score interactions and distribution shift must be evaluated rather than assumed away.

Use a bounded GPU working set, a separately bounded CPU archive, and a source/recomputation store. Retain cache provenance: model/revision, tokenizer/template, prefix ancestry, positions, mask semantics, RoPE configuration, adapters, dtype/format, access scope and computation lineage. Similarity alone cannot establish compatible KV reuse. Different preceding context usually changes a chunk's states. Approximate-trajectory pages must not contaminate dense-equivalent prefix caches.

The existing `htis.bounded_omission` module provides a fixed-state local output-error bound using key extrema and value centroid/radius summaries. Validate it, then investigate an adaptive fetch trigger. All omitted pages must remain represented in the bound. It is not a bound on the final answer, not an interval-arithmetic certificate, and not a repair for earlier trajectory drift. CPU paging/fallback costs and metadata must be charged. If all original pages are unavailable, do not pretend an exact fallback exists.

## Scientific and engineering requirements

Use primary papers and official implementation/documentation sources. Read the architecture's references, especially H2O, StreamingLLM, SnapKV, Ada-KV, Quest, SparQ, RetrievalAttention, SGLang, CacheBlend, A3, KVP, VaSE, Random Attention, and local error-certificate work. Refresh the literature before novelty claims. Record what is reproduced, adapted, independently implemented, proposed, or not tested. Do not copy incompatible licensed code.

Use real code paths, not random scores behind a research-looking UI. Keep synthetic/reference, pretrained-model, and measured-hardware evidence separate. Never fabricate experiments, training, citations, charts, publication acceptance, or deployment history. Keep negative findings. Record real commit dates; the original prototype already has a historical 2025-08-06 commit.

Maintain RESEARCH_PLAN.md, IMPLEMENTATION_STATUS.md and a machine-readable claim/evidence ledger. Every measured claim must point to a run artifact and configuration. Track tests and checkpoints after each phase. Use local instruction files where appropriate without replacing existing project guidance. Do not commit secrets, private VA data, model weights or large unapproved datasets.

Implement a clean optional research package, for example `htis_context`, alongside the dependency-light existing `htis` package. Use typed modules, validation, deterministic tests, configuration files and CLI entry points. Pin the tested dependency environment and model/dataset revisions. Preserve the original simulator commands and tests.

Avoid a broad backend/UI rewrite before model integration. A working small pretrained-model adapter with numerical equivalence and one measured policy tradeoff is more important than an unimplemented serving framework.

## Phase 0: audit and execution contract

Inspect git status, refs, current code, tests, packaging and hardware. Confirm whether the latest research additions are available. Run existing tests and report the actual outcome rather than copying earlier counts. Create an architecture decision record and map each proposed module to a phase and acceptance test.

Record accessible hardware and choose a genuinely pretrained open model that fits it. Verify its model card, license, context length, attention configuration and immutable revision. A tiny random transformer may be used for offline unit tests only. It is not a pretrained-model experiment. Do not incur paid compute or run an external deployment without approval.

**Exit:** a clean baseline or documented pre-existing failures; preserved user changes; reproducible environment plan; an explicit next implementation task. Do not stop with an audit alone if local implementation is possible.

## Phase 1: exact pretrained-transformer adapter

Implement a model-specific reference adapter before generalizing. First support dense KV behavior with all importance policies disabled. Verify cached versus uncached forward logits, multi-step teacher-forced continuation and greedy generation. Declare dtype-specific numerical tolerances and compare against the untouched upstream implementation.

Test original positions, RoPE, padding, masks, multi-head/GQA layout, batch isolation and cache length after append. Add cache snapshot/restore tests and model revision validation. Keep immutable prefix snapshots separate from mutable request tails. Reusing a snapshot must not mutate another request's cache.

Expose CLI commands for model inspection, dense inference and an equivalence report. Add a tiny CPU test configuration and an opt-in real pretrained checkpoint integration test with explicit skip reasons when weights are unavailable.

**Exit:** an actual pretrained-model run with equivalence evidence when a checkpoint is available; offline integration tests regardless. No efficiency claim yet. If weights/network are blocked, finish all local code/tests and report that exact missing dependency instead of replacing the experiment with fake tensors.

## Phase 2: physical page selection and exact prefix reuse

Implement page metadata, budget accounting and a policy interface. Add full, recent, sinks+recent, prompt-protected random, attention-history and query-summary baselines. Label simplified implementations as such. Ensure the selected pages determine the actual keys/values processed; multiplying dense outputs by gates is not sparse execution.

Add an exact-prefix cache with explicit ancestry and eligibility checks, immutable entries, hit/miss events and deterministic eviction. Start with LRU/LFU baselines. Test changed system text, same chunk under a different prefix, changed model/adapter, position shifts, padding variants and tenant separation. Query embeddings may help route source candidates but do not authorize reuse.

Budget by physical KV bytes, including protected regions, metadata and working buffers. Report CPU copies separately. For different model head layouts, test actual tensor shapes and no accidental KV duplication. Persist normalized JSON run records and raw per-example results.

**Exit:** real KV selection and exact reuse work on the reference adapter; causal and invalidation tests pass; full-capacity mode remains equivalent. Report quality changes and measured overhead, even when performance is worse.

## Phase 3: counterfactual dataset and learned HTIS scores

Collect public-data dense traces with state and position provenance. Offline, sample single-page and multi-page interventions and compute local attention-output perturbation, teacher-forced next-token KL, and selected longer-horizon labels. Keep the label pipeline physically separate from runtime features. Include no-lookahead tests and dataset checksums.

Split by document/session, with a task-family holdout. Tune only on validation data. Train linear and MLP baselines before attention/locality/content fusion. Measure ranking quality and calibration, but do not confuse them with end-task quality. Add value outliers, delayed relevance, contradictory updates and task-switch cases.

Evaluate all branches with identical protection rules and cache budgets. Report trainable parameters, trace-generation cost, training cost, feature-storage size and inference overhead. Account for missing attention statistics of evicted items; add a budgeted exploration/recollection experiment rather than using dense teacher attention online.

**Exit:** saved, provenance-linked trained weights and held-out results, or an explicit training blocker with executable pipeline/tests. Hand-chosen weights must remain labeled heuristic. No claim that a trained score is universally useful.

## Phase 4: recoverable memory and cost-aware routing

Implement CPU archive offload/reload with immutable same-lineage pages, transfer counters, bounded storage and measured fetch latency. Recovery of archived KV is not reconstruction from embeddings. Exact dense-prefix reuse and approximate decoding traces must have distinct namespaces.

Integrate the existing local omission bound as an optional, inspectable policy. Add adaptive fetch, tolerances, fallback status, loose-bound diagnostics and budget-aware exploration. Validate fixed-state dense comparisons separately from end-to-end trajectory quality. When a full archive exists, implement a blockwise exact-current-state fallback that respects working-memory limits. Never mark it dense-history-equivalent without proof.

Add trigger-based controller refresh, task-switch detection, hysteresis and an overhead-aware cheap-policy route. Learn or calibrate cost models from measured actions. Prefix reuse prediction and in-request output-risk prediction remain separate objectives. Evaluate whether coordinating them beats composing strong independent baselines.

**Exit:** a recoverable end-to-end example; meaningful cost accounting; tests for stale lineage, unavailable pages, task switches, thrashing, memory limits and fallback. Report cases where the bound or archive is too expensive.

## Phase 5: controlled model and hardware experiments

Build a factorial experiment: prefix reuse on/off x importance policy on/off, with additional no-recovery and no-bound ablations. Include cold and warm prefill, high- and low-reuse workloads, and equal prompt protection. Compare to exact prefix caching plus strong simple retention, not only a cold full-cache baseline.

Use official RULER and feasible long-context QA/multi-turn/code tasks; include LongBench or LongBench v2 within the supported context range. Record exact subsets and preprocessing. Do not silently truncate official tests and retain their original benchmark label. Add a second model family before making cross-family claims.

Measure task metrics, teacher-forced likelihood/KL, free-generation outcomes, TTFT, prefill time, inter-token latency, throughput, p50/p95 latency, GPU allocated/reserved peaks, CPU/archive/index bytes, transfer bytes, scorer/sorter time, page churn, cache hits and fallback frequency. Match hardware, precision, kernels, batch/concurrency, sequence lengths and generation settings. Compare against an optimized dense attention implementation, not just a slow debug loop.

Warm up, synchronize GPU measurements correctly, repeat runs, and separate profiler runs from timing runs. Use paired document/session-level uncertainty estimates and multiple training seeds. Set quality tolerance on validation data before the final run. Create quality/latency/memory Pareto plots from raw artifacts. Label smoke tests and tiny samples as such. If the system is slower, identify whether controller cost, transfers, allocations or kernels caused it and retain the result.

**Exit:** reproducible real measurements on available hardware, with no invented GPU outcomes. If local hardware is inadequate, produce tested runners and an explicit resource estimate; stop before spending money.

## Phase 6: paper and research-facing portfolio

Once the preceding evidence exists, write `paper/htis_context.tex` or an equivalent reproducible manuscript source with abstract, related work, problem, architecture, theorem/assumptions, training, implementation, experiments, ablations, limitations and reproducibility appendix. Verify bibliographic metadata from original sources. Label a pre-results version design manuscript; label an evaluated version technical report unless accepted elsewhere.

Generate tables and plots from checked run artifacts. Link every major claim to evidence. Separate existing literature, this implementation, proposed mechanisms and measured findings. Report model/hardware/configuration details and negative results. A conference-style layout is permitted; a fabricated conference acceptance is not.

Make the root README research-first with a crisp contribution statement, architecture explanation, measured results where available, quick start, paper link and status table. Keep the original HTIS prototype and SignalRank accessible under historical/applied sections. Do not advertise planned features as shipped.

Build a small research explorer only after the data pipeline works. It should show token/page selection, per-layer/KV-group budgets, source/prefix cache events, counterfactual diagnostics, and quality-resource frontiers. Clearly distinguish recorded experiment replay, synthetic demonstration and live model execution. Do not fake a live inference dashboard.

**Exit:** reproducible paper build, working links, claim/evidence audit, honest research demo, preserved original work and accurate git history.

## Execution loop and completion report

For each phase: inspect -> implement -> test -> run a real example -> inspect failure -> repair -> update status/evidence. Continue through achievable work rather than returning a plan in place of code. Stop for unresolved correctness failures, missing permissions, unavailable model/hardware, or unapproved spending, and state the specific blocker.

At the end of each run, provide changed files, commands executed and actual outcomes, new evidence, unimplemented items, failure cases, and the exact next prompt. Commit only your own reviewed changes when appropriate; do not merge, deploy, or rewrite history without authorization.

Start now with Phase 0 and Phase 1. Proceed to Phase 2 if equivalence checks pass and the resources are available. Do not try to hide an unfinished adapter behind a polished UI.

## Continuation prompts

**After the baseline adapter:** Read IMPLEMENTATION_STATUS.md and this brief. Finish Phase 2, including real KV selection, exact prefix-cache identity, negative reuse tests and actual byte accounting. Preserve the dense-equivalence tests and run a pretrained-model example.

**After page selection/reuse:** Continue with Phase 3. Produce the offline counterfactual dataset, leakage controls, trained baseline and learned HTIS ablations. Do not call configured heuristic weights trained scores. Keep a held-out evaluation split.

**After trained scoring:** Continue with Phase 4. Implement recoverable tiered KV memory and cost-aware triggers. Integrate and test the local omission bound without turning it into an unsupported end-to-end guarantee. Report all archive, metadata and transfer costs.

**After runtime integration:** Execute Phase 5 under the available hardware and approved budget. Run composed baselines and factorial ablations. Generate real per-example artifacts and quality-resource frontiers; retain regressions and negative outcomes.

**After sufficient experiments:** Execute Phase 6. Build the paper and research explorer from actual evidence, make the portfolio transformer-research-first, and audit every technical claim, citation and result link before proposing a release.
