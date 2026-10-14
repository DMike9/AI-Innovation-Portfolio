# Phase 5B preregistration

Created 2026-10-13, before validation or new held-out model execution. This is
an extended **CPU reference study**, not an official long-context leaderboard
run. Phase 0–5 evidence, runtime sources and learned weights remain unchanged.
No architecture redesign, training, paid compute or Phase 6 is authorized here.

## Data and workload selection

Use the already checksummed SQuAD v1.1 development release, CC BY-SA 4.0,
[official source](https://rajpurkar.github.io/SQuAD-explorer/). Exclude every
article used in Phase 5. Sort remaining articles by title SHA256 and eligible
QAs by ID SHA256. Eligibility is a complete original paragraph of at most 300
model tokens and first answer of 1–12 tokens. First two articles are validation;
next three are held out. Add complete same-article distractor paragraphs before,
around or after the target paragraph, without clipping paragraphs, to nominal
512/1024/2048-token buckets. Actual lengths may overshoot buckets and are recorded.
This adaptation is **not an official SQuAD or LongBench score**.

Five new authored held-out queries cover delayed relevance, multi-document
reasoning, switching tasks on a shared prefix, returning to earlier history,
and a retired/current value distinction. The two task-switch queries share
one session cluster. Add two complete official-source LongBench v1 cases as
specified below. Ten held-out cases form nine source/session clusters.
Validation and test groups are disjoint and absent from prior project data.
Public data may have occurred in foundation-model pretraining; no claim of
foundation-model contamination exclusion is made.

[LongBench's official repository](https://github.com/THUDM/LongBench) was checked.
LongBench v1 can contain supported lengths. The sandbox network refused the
first metadata connection, but a normal-network retry and pinned acquisition
succeeded before protocol freeze or model evaluation. Include one complete
HotpotQA and one complete 2WikiMultihopQA official-source case. Select by lowest
SHA256 ID among complete official prompts of 512–2400 model tokens and a first
answer of 1–12 tokens. Exclude passage titles already used by any Phase 5 or
new SQuAD validation/test article and by the other selected official case.
Record all eligibility counts and source titles. Preserve full original context,
official prompt, all answers and the official 32-token generation maximum.
Tokenize each full prompt once, then split at an existing token boundary for
prefix caching. Pin dataset revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`
and official repository revision `2e00731f8d0bff23dc4325161044d0ed8af94c1e`;
preserve release/source/prompt and selected original-row hashes. The release
archive stays in ignored local cache. This is a tiny compatible untruncated
official-source subset, **not a full LongBench benchmark score**.
LongBench v2 starts at 8k words,
which does not establish compatibility with this model's 8192-token window.
Do not truncate official cases and retain their benchmark name. RULER is not
reproduced by the authored probes.

## Model, resource and generation controls

Reuse pinned HuggingFaceTB/SmolLM2-135M revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, FP32, CPU, four Torch threads,
batch=1, concurrency=1. Model context support is 8192 original positions.
Frozen Phase 3 HTIS seeds17 and29 are loaded with existing provenance checks.
No scoring architecture or model checkpoint changes. Same tokens, masks,
positions, attention implementation and precision for bounded policies.

Primary policies: protected recency, current query-summary, prompt-protected
random and frozen HTIS. All protect opening/sink eight tokens and the latest
eight, with eight-token physical pages. Worst-case protection needs three
pages, so 4/8/16 pages are feasible: physical ceilings 1,474,560 / 2,949,120 /
5,898,240 KV bytes across 30 layers and three KV heads. Full-cache reference
has budget N/A. Its recovery-ON factorial arm is an explicit identity control:
no omitted pages exist to retrieve. Never implement it as a bounded policy.

Decode budgets begin at first answer-token computation. The established adapter
requires dense prefix and question prefill. Record its complete physical KV and
working-set estimates: **these are not end-to-end memory-bounded requests**.
Sparse policies compact before attention on each decode step; assert physical
page/byte limits for every teacher-forced and generated step.

Archive capacity 128 MiB, index capacity 64 MiB, known working estimate 512 MiB,
and independent exact-prefix cache 192 MiB. Archive, scorer, prefix and metadata
are charged as additional storage. These limits are feasibility allocations,
not equal-total-memory comparisons with archive-disabled policies.

Greedy argmax, no sampling/temperature adjustment, no tuned answer extraction.
Stop on the model EOS or a fixed case limit; do not stop on target-answer text.
Two preselected longer adapted cases (end-position public QA and historical
revisit) and both official LongBench cases allow 32 generated tokens; other
cases allow 16. Whole output is scored with the
unchanged normalized EM/token F1. Raw text, token IDs, length, EOS/limit behavior
and case regressions must be preserved. Teacher-forced NLL and KL to dense
reference are separate diagnostic metrics, never answer accuracy.

## Validation, freeze and experiment matrix

Freeze this document, data, configuration, runtime/test source hashes and model/
HTIS checkpoint hashes before validation. Two new validation articles compare
all four bounded policies at eight pages, seed17, without recovery and with
two existing-controller candidates: interval4 or8; cooldown2; query change.35;
risk/uncertainty/rank triggers.5; route admission allowance100,000 ms; score prior.1 ms.
The deliberately nonbinding route cost allowance keeps noisy measured cost
estimates from changing selection between equivalent cold/warm executions.
Actual costs remain measured and subject to the contribution B latency criterion;
this is controlled recovery validation, not a demonstrated real-time controller.
Choose one common recovery configuration by highest validation F1, then lowest
NLL, latency and candidate index. Preserve the candidate table. Freeze the
selected thresholds and calibration hash before held-out execution. No held-out
threshold tuning; any protocol amendment becomes a separately labeled experiment.

Quality matrix: ten cases × three budgets × four policies × recovery OFF/ON
× seeds17/29, plus one full reference per case: **490 trials**. Validation has
26 trials. Policies are evaluated in deterministic shuffled order per case.
Teacher continuation and free generation start fresh from the same dense fixture;
future target labels/logits are never passed into scoring. Check fixture integrity.

Systems matrix: opening public QA and historical-revisit case, all three budgets,
all four bounded policies, recovery OFF/ON and prefix OFF/ON, two repeats.
Repeat0 uses seed17; repeat1 seed29. Full reference crosses recovery identity
OFF/ON and prefix OFF/ON with two repeats, budget N/A. Each independent cell first
executes a real prefix-ON miss/admission request, then matched prefix-OFF cold and
prefix-ON warm requests in alternating order. Expected **312 requests**, plus
four actual modified-prefix full-reference requests and four shared-prefix
task-change/control requests: **320 actual requests**. Real admission requests
are retained and counted separately from cold/warm comparison requests.

Exact compatible reuse must preserve generated tokens and every captured step's
logits within predeclared atol5e-5, rtol1e-5. A modified first prefix token with
the same positions/length must miss. Retain full-prefix cache events, lookup,
admission/invalidation costs, saved prefix tokens and cold prefill measurements.
Do not report intended warm hits as a production traffic hit rate. Use measured
admission/cold/warm times to calculate traffic mixes with zero, one and eight
warm repeats; derived amortizations are not additional measured requests.

## Timing, profiling and memory

CPU perf_counter, no concurrent model benchmarks/tests/profilers, native warmup
before measurements. Record controller construction, prefill, question suffix,
TTFT, decode/ITL, end-to-end and generated tokens per second. Standard adapter
timing instrumentation stays enabled across policies. Observation capture time
is measured. Systems logit copies used for numerical verification are charged
as temporary diagnostic storage. JSON serialization is outside request timing.
Keep timings as measured; do not subtract noisy overhead to manufacture speedups.

Compare complete prompts and common-token continuations with untouched upstream
Transformers SDPA using the same model and declared tolerance. Keep optimized
SDPA performance separate from the independently implemented reference path.
Profile eight policy/recovery cells separately on a new validation source:
two plain repeats, separate cProfile/nested spans, and separate native Torch
operator allocation profile. Report insertion features, scoring features,
network/ranking/selection, copies, integrity, controller and allocation traffic;
inclusive nested spans cannot be summed. Profiling overhead is explicitly recorded.

Count actual physical KV tensors and metadata tensors; archive KV separately from
archive storage including metadata. Index/recipe structural bytes are labeled
estimates. Scorer parameter/normalizer tensors and controller query tensors are
charged. Add independent prefix cache storage to known cache-system totals.
Record known transitional/feature tensors separately from persistent storage.
Windows OS process working set/lifetime peak includes base model, research
fixtures, Python heap and allocator; it is not isolated per-request cache peak.
Untracked activation/scratch and exact Python/allocator attribution remain a
measurement limitation. Do not compare differing archive/prefix storage as equal
total memory. Physical decode capacity is not the dense-prefill peak.

Inspect local CUDA availability and installed build. If unavailable, preserve
and dry-run the existing tested GPU runner, record NOT EXECUTED and remaining
CUDA correctness/synchronization/allocator/transfer requirements. No paid hardware.

## Statistics, Pareto analysis and decisions

Use equal-case descriptive means/medians, seed-specific summaries and per-case/
workload/budget tables. For paired comparisons, average matched seeds/repeats
within case and then session; resample nine source/session clusters 10,000 times
with seed1729. Performance systems have only two source/session clusters and two
timing repeats with seeds confounded with repeat; report intervals descriptively,
not as reliable production significance. Seed variation is checkpoint/random
variation, not an independent dataset. Workload singletons support case observations.

Primary A comparisons: HTIS versus each of three simpler policies at each budget,
recovery OFF. Report ordinary 95% intervals and Bonferroni **99.444%** percentile
intervals for nine predeclared comparisons. A is SUPPORTED only if all three
adjusted paired F1 intervals exclude zero positively at a budget. A is NOT
SUPPORTED if no mean improvement over the strongest simple policy exists;
otherwise INCONCLUSIVE. No acceptable generated-quality degradation margin is
assumed (zero margin); NLL/KL cannot substitute for F1. These small bootstrap
intervals still do not establish cross-workload or cross-model generalization.

B is SUPPORTED only for a specific policy/budget when recovery has positive
adjusted paired F1 interval, non-increasing mean KL, and at most **2×** measured
paired end-to-end latency versus no recovery. The 2× allowance is an advance
research cost constraint, not a validated production utility model. Use twelve
recovery comparisons (four policies × three budgets), Bonferroni 99.583% intervals.
Report additional archive bytes without a permitted equal-memory claim. If no
mean F1 improves or all measured improved cells exceed the latency allowance,
B is NOT SUPPORTED; otherwise INCONCLUSIVE. Retain improvement/unchanged/worsened
quality and retrieval/no-retrieval counts, unnecessary/worsening recovery cases.

C concerns additional coordinated benefit beyond strong **simple policy + exact
reuse** combinations. Require preserved reuse outputs, a warm HTIS/composed cell
on the measured F1/latency/total-memory frontier, and paired evidence of F1 gain
over simpler warm compositions at matched budget. If learned compositions are
dominated, C is NOT SUPPORTED; otherwise uncertainty may make it INCONCLUSIVE.
Separately report generic exact-reuse benefit even if C fails: already-known
prefix reuse is not evidence of a novel learned coordination contribution.

Generate Pareto figures from actual artifacts for answer-F1 versus decode/request
latency, physical active KV and total storage. Use consistent cold/warm strata;
archive-enabled policies have extra storage. Report all policies, budgets and
negative results. Teacher NLL/KL frontiers are labeled separately.

Offline diagnostics use frozen estimators at layers0/14/29 on held-out dense
current-query snapshots. Compare HTIS/query ranks, seed agreement, shared-prefix
task changes, value norms and three preselected page-omission candidates per
layer. These labels never feed runtime selection. A scorer has no budget input:
same-state scores should be budget-invariant; compare actual retained-page
trajectories across budgets separately. Diagnose prediction/allocation/workload/
overhead failures without retraining or adding a scoring architecture.

## Prior work and completion boundary

Primary sources refreshed: [H2O](https://arxiv.org/abs/2306.14048),
[SnapKV](https://arxiv.org/abs/2404.14469), [Quest](https://arxiv.org/abs/2406.10774),
[RetrievalAttention](https://arxiv.org/abs/2409.10516),
[SGLang](https://arxiv.org/abs/2312.07104). Their attention-history, prompt
selection, query-aware page access, CPU retrieval and exact-prefix reuse mechanisms
are prior work. This study uses simplified existing reference baselines, not
faithful implementations of those optimized systems. No global novelty claim.

Completion requires full feasible matrices, preserved raw failures, correct
numerical/budget checks, regression/integration tests, source/checkpoint/hash
audit, plots, and evidence-linked A/B/C decisions. If a numerical/resource gate
fails, preserve completed trials and mark scope INCOMPLETE instead of inventing
results. Official benchmark or GPU paths not executed must remain explicit.
Stop at Phase 5B; Phase 6 is not started by this request.
