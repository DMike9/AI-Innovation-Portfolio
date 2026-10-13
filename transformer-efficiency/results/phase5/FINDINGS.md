# Phase 5 findings and decision

**Decision: B — useful components, excessive coordination/recovery overhead.**
On the primary free-generation F1/CPU-latency/total-storage frontier, strong
simple policies also support C. The model/hardware/sample limitations support D
for any broader inference-efficiency claim. The study does not establish A,
publication-level novelty, GPU acceleration or long-context generalization.

The final evidence is `evaluation-v1`, `analysis-v2`, the frozen
`../../configs/phase5-cpu-v2.json`, and `validation-v2/calibration.json`.
Initial analysis-v1 is preserved but superseded for matched-seed ablation
pairing and figure readability. No inference trial, test example, threshold or
trained weight was changed during/after held-out execution.

## Executed scope

- Genuine pinned SmolLM2-135M, FP32, Torch 2.9.1+cpu / Transformers 4.57.6.
- Four validation cases, eight held-out cases, complete inputs 93–226 tokens,
  six fixed greedy output tokens. Official SQuAD v1.1 paragraphs are untruncated;
  other rows are authored memory probes. This is not a leaderboard score.
- 196 held-out quality trials: 17 core arms, original trained seeds 17/29 where
  applicable, and 13 targeted ablation arms on two cases fixed before outcomes.
- 32 actual systems requests: two cases × learned HTIS/simple × recovery on/off
  × exact prefix reuse on/off × two repetitions. “Simple” was validation-selected
  prompt-protected random; query-summary is also reported across all test cases.
- 204 runtime checks passed; 32 upstream SDPA common-token comparisons passed,
  maximum absolute logit difference **3.6239624e-5**, within existing tolerances.
- Final combined suite: **186 passed**, 37.11 s, including 165 original tests,
  Phase 5 genuine pretrained checks and three corrected pairing tests.
- Held-out execution: **567.36 s** internal CPU wall time. No other tests or
  profiling jobs ran concurrently with it. Runtime sources stayed unchanged.

## Primary comparisons

Means over eight cases; trained-seed means averaged within each case.
KL/NLL are independent teacher-forced trajectories. Decode means exclude the
shared prefill fixture; actual end-to-end requests are below. All core whole-output
EM values were zero under the frozen six-token/SQuAD-normalization protocol.

| Policy | Free F1 | Teacher KL | Target NLL | CPU decode ms |
|---|---:|---:|---:|---:|
| Full reference | 0.2542 | 0 | 4.3535 | 131.22 |
| Untouched upstream SDPA, full KV | 0.2542 | equivalence checked separately | not scored separately | 99.04 |
| Protected recency | 0.0500 | 0.7380 | 7.3412 | 134.16 |
| Query-summary heuristic | 0.1333 | 0.5614 | 6.8339 | 139.64 |
| Frozen HTIS | 0.1104 | 0.6226 | 6.4347 | 148.74 |
| HTIS + recovery | 0.1104 | 0.6247 | 6.4408 | 523.19 |
| Recency + query recovery | 0.2218 | 0.4779 | 6.7805 | 373.59 |
| Bound + full supplied-state fallback | 0.2542 | near numerical zero | 4.3535 | 942.90 |
| Conditional HTIS, interval4/change0.35 | 0.0875 | 0.6549 | 6.1249 | 177.12 |

HTIS improved over recency in this pilot, unlike the preserved Phase 3/4 overall
results. It did not improve over the query-summary policy on F1/latency/storage.
Its lower target NLL creates a different tradeoff: HTIS is not dominated on the
NLL frontier. Do not substitute likelihood or KL for answer F1. The F1 frontier
identifies HTIS as dominated by query-summary, and both recovery arms and fallback
as dominated by the full CPU reference on F1/latency/total storage. These are
descriptive measured means, not statistical dominance proofs. Alias latency
differences are noise, not mechanisms.

## Recovery and controller failures

Recency-based recovery improved KL on **6/8** cases and F1 on **3/8**, with no F1
decreases. HTIS recovery improved KL on **1/8** and F1 on **0/8**. These case-level
quality outcomes, not page fetch counts, define recovery value. The former cost
373.59 ms/step versus 134.16; the latter 523.19 versus 148.74. Measured HTIS recovery
controller time averaged 295.19 ms/step, including 132.85 ms scorer time. Archived
pages can be useful, but learned ranking and reference archive management did not
coordinate economically here. History cannot observe evicted pages and carries
survivor bias; that structural limitation is not a causal ablation result.

The conservative measured-cost prior and the margin-to-ms benefit gate performed
zero reloads and matched recency quality, yet cost 427.53 and 443.54 ms/step.
Archive maintenance, integrity, summary/index work and copies remain expensive
even when a gate rejects retrieval. Profile priors include instrumentation and
are deliberately conservative; they are not fabricated GPU transfer measurements.
The validation speed rule chose recency, and the held-out calibrated arm retained
that choice. Its failure to choose a better quality tradeoff follows from its
declared fastest-feasible objective, not held-out threshold adjustment.

The query/interval wrapper added 69,120 bytes of saved current-query state. Its
249 trained-policy selection invocations across eight free-generation trials
include early returns without network evaluation. The original learned policy
already skips feature extraction/network scoring when available pages fit the
budget; checking a trigger at every layer can cost more than it saves in a short
decode. This explains why frequency gating is not automatically an optimization.

On the two predeclared cases, recomputing insertion summaries left HTIS seed17
outputs and KL unchanged but added **33.87 ms/step** on average (32.52 ms summary
work versus the cached path's roughly 14–15 ms). Preserve insertion caching.
Original branch/query/value checkpoints were evaluated without retraining.
No-query/no-value variants sometimes improved this tiny subset; do not treat
that as a general feature-removal finding. Cooldown0/always-refresh did not improve
F1 there. Eight-page HTIS reached F1 **0.8333** versus eight-page recency **0.0833**,
but only on two cases and without query/random eight-page controls. This is a
discriminating next experiment, not a confirmed benefit over strong equal-budget
alternatives. No adaptive-budget component exists to ablate.

## Exact prefix reuse and resources

Actual means over four requests per cell (two cases/two repetitions):

| Policy | Cold TTFT / end-to-end ms | Warm TTFT / end-to-end ms |
|---|---:|---:|
| Validation-selected simple | 951.58 / 1618.11 | 379.62 / 1004.71 |
| HTIS | 1038.20 / 1740.92 | 525.10 / 1202.50 |
| Simple + recovery | 1565.02 / 3225.06 | 1075.14 / 2715.76 |
| HTIS + recovery | 2120.70 / 4577.55 | 1332.44 / 3787.53 |

Warm/cold output tokens matched in every repeated cell. There were **16 exact
hits and two intentional modified-prefix misses**, not a measured natural
traffic hit rate. Warm lookup averaged about 19–22 ms; prefix admission was
136.91 ms for the official QA prefix and 59.21 ms for the authored prefix.
Initial dense prefix computation and admission are recorded separately, not
removed from the cost of populating a reusable entry. Warm simple/HTIS quality
on these two cases was F1 0/0.1 respectively; reuse itself changed no output.
Interaction estimates had mixed/noisy signs and only two repeats, so coordination
has not established a gain beyond composing independent reuse and retention.

Bounded core active KV peaked at **1,474,560 bytes** versus **10,644,480** for full
reference. Full reference cache/index storage peaked at **12,804,960 bytes**;
HTIS recovery reached **17,168,707** despite equal active capacity. Maximum archive
storage was **11,295,000**, index estimate about **4,145,300**. Independent warm
prefix storage added up to **11,525,602**, bringing warm HTIS recovery to
**28,694,249** known cache/index/scorer/prefix bytes in the factorial. Model weights
were a separate **538,060,032 bytes**. Working tensor estimates, source/Python
heap, untracked activations/scratch and allocator/RSS peaks must not be confused.
These are CPU residency/copy measurements, not GPU memory savings.

## Bound evidence, uncertainty and limitations

Raw teacher/free trajectories contain **16,470** measured fixed-state per-head
bound/error comparisons, with **zero observed violations** under the stated
numerical margin. Streaming reproduced this run's near-dense quality at high
cost; it does not certify final answers or repair earlier approximate states.
Periodically skipped bounds explicitly provided no certificate. Phase 4's
44,271 comparisons and median looseness ratio 45.25 remain unchanged.

Bootstrap resamples six document/session clusters, 2,000 repetitions, seed1729;
positional variants share one cluster. Core tables weight eight cases equally,
whereas paired bootstrap weights cluster means equally. HTIS versus recency
cluster-weighted F1 delta was **0.03796**, percentile interval **[0, 0.08056]**.
HTIS seeds17/29 had F1 **0.0875/0.1333**, KL **0.65495/0.59022**. This is exploratory,
not a significant or generalizable improvement. Single-seed ablations now pair
only the same original checkpoint seed; initial analysis mixed seeds there.

Whole-output EM/F1 penalizes explanatory text: full attention correctly generated
“The locker color is green” but its one-word target gives F1 0.4/EM0. This protocol
was frozen and not changed after outputs. Protected recency, query-summary,
HTIS seed17 and recency recovery hallucinated blue on the opening probe at four
pages. The code probe's query heuristic/recovery retained
`violet`, while HTIS seed17 generated `c`. Raw per-example outputs and every
negative case remain available. A larger instruction model and preregistered
stopping/metric protocol are needed for a stronger practical answer-quality study;
new model identities cannot silently reuse these provenance-bound trained weights.

CUDA hardware/build is unavailable. The runner's CPU dry-run, gating and timer
tests passed; CUDA numerical/memory/transfer/timing paths are **NOT EXECUTED**.
No GPU recovery support, official RULER/LongBench run, second model family, batched
sparse serving, disk archival or recomputation was added. Short contexts and a
small base model cannot establish long-context or production systems efficiency.

Official implementation feasibility was inspected without importing new serving
architectures: [SnapKV](https://github.com/FasterDecoding/SnapKV) documents an older
Transformers-tested version and FlashAttention requirement;
[Quest](https://github.com/mit-han-lab/quest) documents CUDA/custom-kernel builds
and a GQA-support TODO. Those do not fit this CPU GQA reference unchanged.
[H2O](https://github.com/FMInference/H2O) provides separate HF/FlexGen paths;
no provenance-compatible official integration was validated in this pilot.
History/query heuristics remain explicitly simplified, and SnapKV was not executed.

## Next direction and exact continuation

Prioritize exact prefix reuse, low-cost query-conditioned selection and larger
fixed-budget comparisons before more learned-controller complexity. Preserve
the current weights and all negative evidence. Do not begin Phase 6 yet.

> Continue HTIS-Context from the verified Phase 5 short CPU checkpoint. Read
> IMPLEMENTATION_STATUS.md, RESEARCH_PLAN.md, CLAIM_EVIDENCE.json, the frozen
> phase5-cpu-v2.json manifest, Phase 5 FINDINGS.md, analysis-v2 and checkpoint
> verification. Preserve all existing artifacts, user changes, trained weights,
> SignalRank and original HTIS. First reverify evidence and numerical integration.
> Continue Phase 5 as a separately versioned extension: on already available
> compatible local CUDA hardware, run the prepared numerical acceptance gates
> before any timing; do not provision paid compute. Preregister a larger paired
> context/budget study with exact reuse crossed with recency, query-summary,
> random and frozen HTIS at matched 4/8/16-page budgets, plus selective recovery.
> Use new validation sessions and a untouched final split; measure first-compaction
> scoring, archive maintenance, actual TTFT/ITL/end-to-end latency, peak allocated/
> reserved memory, total storage and free-generation quality. Validate any newly
> selected pretrained model against upstream and record its official card,
> license, revision and context support; never bypass trained-checkpoint identity
> guards or replace missing weights with random tensors. Keep all unexecuted
> GPU/long-context/generalization portions explicit. No push, merge, deployment,
> history rewriting or Phase 6 until the stronger composed-baseline evidence
> establishes a defensible contribution.
