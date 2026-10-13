# HTIS-Context
## Task-conditioned, cost-aware memory for transformer inference

Daniel Michael | Research architecture and experimental proposal | 2026-10-13

**Research question:** Can a hierarchical controller decide which computed context to reuse, retain, fetch, or recompute so that a transformer achieves a better quality/latency/memory tradeoff than exact prefix caching plus standard KV policies?

**Implementation update (2026-10-13):** the initial pretrained CPU reference,
exact-prefix cache and physical page-selection extension now exist. See
[current implementation status](../IMPLEMENTATION_STATUS.md) and
[claim/evidence ledger](../CLAIM_EVIDENCE.json). The status statement below
describes the starting specification; learned/tiered mechanisms remain proposed.

**Status:** The HTIS-KV simulator and a new fixed-state, value-sensitive omission-bound reference are implemented. The learned controller, pretrained-model adapter, tiered runtime, and hardware performance claims below are proposed work, not completed experiments. This document is a build specification, not a peer-reviewed publication or a claim of established algorithmic novelty.

## 1. Research identity

The research flagship is transformer inference memory, not SignalRank's project-priority UI. SignalRank remains a separately labeled applied demo. Its category defaults and keyword weights are not evidence about neural importance, attention, or compute savings.

The original HTIS prototype combined linear, convolutional, and feed-forward token scores and gated an already-computed attention output. Preserve it as the historical starting point. The new architecture moves selection into the path that determines which KV pages are loaded and processed. It tests learned decisions rather than treating an untrained score visualization as an efficiency result.

The core idea is **reusable computation with task-dependent value**. A page can be useful now, likely useful later, cheap to reconstruct, expensive to fetch, or invalid for the current prefix. Those are different quantities. A single timeless importance number cannot express all of them.

## 2. What is cached, and what is not interchangeable

| Object | Intended role | Correctness boundary |
|---|---|---|
| Source text and token IDs | Durable evidence and recomputation input | Does not itself eliminate transformer computation |
| Semantic embedding | Route a task toward candidate source segments | Not a replacement for text, nor a model KV tensor |
| Exact prefix KV | Reuse previously computed states for a matching prefix | Requires compatible tokens, model, positions, masks, precision and computation lineage |
| Active decoding KV | Attention inputs retained for the current sequence | Selection may approximate full attention and alter later states |
| Archived KV page | Reload a state from CPU or another tier | Reloading preserves that page's recorded state; it does not repair an earlier divergent model trajectory |
| Answer cache | Reuse a prior output | Out of scope for the core experiment; not equivalent to KV reuse |

Exact prefix caching and context retrieval solve different problems. Similarity can propose a retrieval candidate, but cannot authorize exact KV reuse. Likewise, keeping fewer GPU-resident pages is not necessarily reducing total memory when a full CPU archive remains.

Two goals are evaluated separately: repeated-prefix reuse reduces repeated **prefill**; selective KV access/retention targets **decode** bandwidth and memory. Compressing after a full cold prefill does not retroactively save that prefill.

## 3. Proposed architecture

### 3.1 Provenance and eligibility layer

Every cached object has a provenance key covering model/checkpoint revision, tokenizer and chat-template revision, token prefix hash, original positions, attention-mask semantics, RoPE configuration, adapter identity, dtype/quantizer/cache format, tenant/access scope, and computation lineage. Hashing is a lookup mechanism, not an authorization check. Validate access before retrieval.

Exact reusable prefixes are immutable. Hash-chain or trie ancestry matters: identical chunk text under a different preceding context is not the same computation. Modified source, reordered documents, changed system instructions, altered adapters, or a different position map trigger invalidation or explicitly approximate recomputation. Never advertise arbitrary chunk concatenation as exact reuse; CacheBlend and A3 are relevant prior art for approximate fusion [R6,R7].

Approximate execution traces live in a separate namespace from dense-equivalent prefix caches. A later token computed after an earlier eviction may have different K/V even if its token ID is unchanged. Track this distinction instead of marking the whole cache exact after a reload.

### 3.2 Memory tiers and hierarchy

1. **GPU working set:** bounded active KV pages plus controller features, summaries and transfer buffers.
2. **CPU archive:** recoverable same-lineage KV pages. Bound and measure its size independently. The first implementation uses unquantized pages so selection and quantization errors are not conflated.
3. **Source/recomputation store:** tokenized source, ordered-prefix recipe and content hashes. Recomputing a non-prefix chunk alone does not restore full-prefix equivalence.

The decision hierarchy is session/prefix, segment, page, layer and KV-head group. Pages use original contiguous position ranges for kernel-friendly transfers; logical positions never become compacted-array indices. Grouped-query attention shares K/V among query heads, so allocation and byte accounting must occur at the actual KV-group level. Do not duplicate the cache once per query head and then claim memory savings.

### 3.3 Two controllers, not one mixed score

**Across requests:** estimate the probability that an eligible prefix will be reused and the prefill time saved by retaining it. A practical scheduling proxy is:

`prefix_value = estimated_reuse_probability * max(0, measured_recompute_ms - measured_fetch_ms) / stored_bytes`.

This is a heuristic requiring calibration, not an optimal cache theorem. Dependency-aware trie eviction and reference counts must prevent orphaning shared prefixes. Compare against LRU and LFU before adding a trained predictor.

**Within a request:** predict action-conditioned degradation and latency. For state `s_t` and action plan `a`, target

`min_a E[output_degradation(a) | s_t] + lambda_t * E[latency_ms(a) | s_t]`

subject to explicit GPU and CPU byte budgets and eligibility constraints. Units are reconciled by the latency tradeoff parameter. Report a Pareto frontier rather than hiding all outcomes inside one score.

Actions initially are retain, offload, fetch and omit. Recompute is added only after provenance-correct prefix recovery works. Quantization is a later independent extension. Scores for individual pages are conditional marginal-utility estimates, not additive truth: interactions between omitted pages can invalidate an independent top-k intuition.

### 3.4 Hierarchical importance estimator

Preserve the original HTIS multi-signal concept, but train and ablate it:

- **Observed-attention branch:** running attention statistics available from past selected attention and inexpensive current query/page summaries. It must not request a full attention matrix merely to select a sparse subset.
- **Locality branch:** a small convolution over projected states at their original neighboring positions, computed/cached at insertion or prefill. Never treat nonadjacent surviving tokens as neighbors after compaction.
- **Content branch:** a small MLP over projected K/V summaries, value magnitudes/residuals, recency and task features.
- **Fusion:** a query/task-, layer- and KV-group-conditioned gate predicts marginal error, future reuse and an empirically calibrated uncertainty estimate. Predictions are not formal probability guarantees.

The attention query vector inside a layer is different from an external semantic query embedding. Keep them separately typed and account for both costs. External embeddings shortlist source segments; model Q/K interactions shortlist model-state pages. RetrievalAttention documents why ordinary vector-search assumptions can fail for model queries and keys [R9].

Start with a linear model and small MLP. Add the convolution or additional hierarchy only when an ablation shows an improvement. A larger controller that consumes more time than it saves is a failed efficiency intervention.

### 3.5 Conditional computation and recovery

Use a cheap baseline when metadata predicts that learned scoring will not pay for itself. Run expensive scoring on a task change, page insertion, a risk/uncertainty trigger, or a chosen refresh interval. Use hysteresis to reduce page thrashing. Validate trigger thresholds on held-out validation sessions.

Give the controller a small, budgeted exploration sample of nonpreferred archived pages to study survivor bias and delayed relevance. Exploration consumes memory and bandwidth and cannot inspect future answers. Retention should be reversible when an archive exists; permanently dropped pages cannot be magically resurrected from their semantic embeddings.

Separate operational pinning from importance estimation. Protect configured instruction/current-token/recent regions, count all protected bytes, and compare with a prompt-protected random baseline. A pinned region is a policy rule, not evidence that its attention weights are meaningful.

## 4. From attention mass to output-sensitive importance

For one fixed attention head, let `a_i` be softmax weight, `v_i` its value, and `y = sum_i a_i v_i`. Removing a set `G`, then renormalizing remaining weights, gives

`y - y_without_G = [sum_{i in G} a_i (v_i - y)] / [1 - sum_{i in G} a_i]`.

This identity is valid when the remaining attention mass is positive. It shows why mass alone is insufficient: removing a high-weight value close to the existing output can be harmless; removing a low-weight but large, different value can matter substantially. These are local counterfactuals, not causal explanations of the final answer. Value-aware eviction is existing research [R11]; the contribution must be tested beyond that observation.

Training targets progress from this fixed-state local perturbation to teacher-forced next-token KL and downstream task loss under actual cache interventions. Compute future-horizon targets only offline. Test-time features may use the available prompt and previously generated tokens, never future generated text or benchmark answers. Split by document/session and task family, not merely random token rows.

## 5. Implemented mathematical reference: pagewise omission bound

`src/htis/bounded_omission.py` supplies a float64 NumPy reference. It is not a fast kernel and does not yet select pages inside a pretrained model.

### Assumptions

One fixed, causal, single-head Q/K/V state; standard scaled dot-product softmax; nonempty retained set `S`; complete, disjoint omitted pages; exact-state summaries; no unaccounted quantization or extra positional bias. RoPE can already be applied using correct original positions. Floating-point validation is not a formal interval-arithmetic certificate.

Let `Z_S = sum_{i in S} exp(q^T k_i / sqrt(d))` and `y_S` be retained attention output. For omitted page `p`, store key-coordinate minima/maxima, value centroid `c_p`, radius `r_p >= max_i ||v_i-c_p||`, and token count `n_p`.

Compute a query-conditioned logit upper bound:

`ell_p = sum_j q_j * (key_max[p,j] if q_j >= 0 else key_min[p,j]) / sqrt(d)`.

Then page mass satisfies `0 <= M_p <= u_p = n_p * exp(ell_p)`, and every omitted value has distance from the retained output at most

`D_p = ||c_p - y_S|| + r_p`.

The exact decomposition gives the conditional local bound

`||y - y_S|| <= sum_p M_p D_p / (Z_S + sum_p M_p)`.

Maximize its right-hand side over the box `0 <= M_p <= u_p`. Sort `D_p` descending. The maximum is attained at a prefix vertex, so a computable upper bound is

`B = max_{k=0..P} [sum_{p=1..k} u_p D_p] / [Z_S + sum_{p=1..k} u_p]`,

where the `k=0` value is zero. The implementation uses log-domain cumulative sums.

**Proof sketch:** the norm bound follows from the triangle inequality. For the fractional objective `F`, its derivative with respect to one mass has the sign of `D_p-F`. Thus an optimum saturates masses with distances above its objective value and sets lower-distance masses to zero; ties do not affect the value. Sorting distances therefore contains an optimum among its prefix vertices. This solves the relaxed box problem, not the original task-level cache allocation problem.

**Validation:** tests compare the fractional solution against all vertices for small cases and compare the bound against actual dense/retained attention differences. Other cases include identical values, low-mass/high-value outliers, extreme finite logits and invalid inputs. The derivation is presented for review, not claimed as a literature-first theorem. Existing runtime-certification work must be discussed [R12,R13].

### What the bound does and does not permit

It can support an adaptive experiment: compute retained output, bound omitted-page error, fetch a useful page when the bound is too large, and recompute/merge attention until tolerance or resource limits are reached. A summary-only shortlist must still cover **all** omitted pages to claim the bound.

It bounds the current attention output relative to the supplied full KV snapshot, not the full model's answer. Previous approximate steps can already have changed the query and later KV states. A full-page fallback restores current-snapshot attention, not the historical dense trajectory. True dense recovery requires a valid dense prefix or full appropriate-prefix recomputation.

If a hard GPU budget precludes loading all pages, a future fallback can use exact streaming/blockwise softmax over CPU-resident pages. Its transfers and working buffers must be counted. Without the original pages, this fallback does not exist.

Computing summaries is `O(T*d)` at construction; a flat query pass is `O(P*d + P log P)` for `P` pages, plus retained attention. This is not a proof of sublinear inference. Metadata is approximately `(2*d_k+d_v)*metadata_bytes + counters` per page/group/layer, before Python/runtime overhead. Small pages can make the metadata cost material. A hierarchical index is an optimization to profile, not a guaranteed win.

## 6. Learning protocol

Freeze the base model first. Collect dense public-data trajectories, prefix-cache events, and sampled page interventions. Label local output perturbation, teacher-forced logit divergence, and measured action cost separately. Include multi-page interventions to expose nonadditivity. Future utility is a training label only.

Train a linear baseline, MLP and HTIS three-branch model. Use ranking/regression losses over multiple cache budgets; do not interpret rank scores as calibrated task-success probabilities. Calibrate uncertainty on untouched validation sessions. Recollect traces under the learned policy to expose distribution shift and survivor bias, with a recorded exploration budget.

A later fine-tuning experiment may train the controller or lightweight adapters with task loss, dense-teacher KL and resource penalties. It is not required for the first runtime result and must be compared against the frozen-base setting. Report trainable parameter count, training compute and amortization assumptions.

## 7. Evaluation that can establish a contribution

**E0: mathematical/reference correctness.** Bound coverage on controlled states; brute-force optimization checks; causal positions; no hidden oracle. This stage alone is not an LLM result.

**E1: exact real-model adapter.** With selection disabled, match unmodified cached/uncached forwards and greedy generation within declared tolerances. Test RoPE positions, padding, GQA, batch isolation, cache hits/misses, changed prefixes, invalidation and restart/reload.

**E2: equal-resource policy comparison.** Full cache, recent, sinks+recent, prompt-protected random, H2O-style, SnapKV-style, adaptive-budget baseline, query-page selection and learned HTIS. Use official implementations where possible; label simplified ports honestly. Include KVP as a modern learned comparison and VaSE/Random Attention in their applicable reasoning regime [R10,R11,R14].

**E3: factorial reuse experiment.** Cross exact prefix caching on/off with retention/scoring on/off; use controlled repeated-prefix traces, low-reuse traces, task switches and modified-prefix misses. Compare against exact prefix caching plus the strongest simple policy, not only a cold dense baseline.

**E4: quality.** Public long-context QA, multi-document reasoning, multi-turn memory, code-context tasks, RULER and feasible LongBench/LongBench v2 subsets [R15,R16]. Test information position, age, distractors, value outliers and forgotten-then-revisited facts. Do not silently truncate a benchmark beyond a model's supported context and keep its official name. Include teacher-forced likelihood/KL and free-generation task metrics.

**E5: systems.** Separate TTFT, prefill, inter-token latency, p50/p95 latency, throughput and actual peak GPU allocation/reservation. Report CPU archive bytes, index/metadata bytes, transferred bytes, scoring/sorting time, cache-hit distribution, page churn, bound tightness, fallback rate and total wall-clock cost. Match model, hardware, precision, kernels, batch size and output lengths. Measure optimized dense attention as well as the inspectable eager reference. No speed claim from fewer simulated QK comparisons.

**Statistics:** freeze workloads/thresholds before final evaluation; use paired examples, document/session-level bootstrap intervals, multiple training seeds and repeated timing runs. Report quality/latency/memory frontiers and failures. A noninferiority margin is a preregistered experimental choice, not a promised result.

**Ablations:** remove each HTIS branch, query conditioning, value features, adaptive budgets, cache reuse, archive recovery, uncertainty trigger and local bound. Compare equal prompt protection. Compare tuned fixed mixtures to learned fusion. If a component adds overhead without benefit, remove it.

## 8. Candidate contribution and falsification

The candidate contribution is a **task-conditioned, value-sensitive and cost-aware controller that coordinates exact reuse and recoverable in-request KV access**, with explicit provenance and local risk diagnostics. Combining named components is not automatically novel. A convincing result must show an interaction benefit over well-composed baselines, identify the workload regime, and survive an ablation that removes the claimed mechanism.

Fail H1 when HTIS adds no quality/resource Pareto improvement over exact prefix caching plus a strong simple policy. Fail the recovery claim when transfer/recomputation costs exceed its benefit. Fail the learned-score claim when a prompt-protected random or fixed heuristic policy matches it. A clear negative result still improves the research artifact; do not replace it with fabricated gains.

This is an inference-memory architecture integrated at transformer attention/cache boundaries. It is not a claim of a newly pretrained foundation model, infinite lossless context, universal speedup, or a drop-in modification to closed hosted APIs.

## 9. Literature and novelty ledger

Sources checked 2026-10-13. Recent arXiv manuscripts are research claims, not automatically peer-reviewed or independently reproduced. Repository results must be generated locally, not borrowed from these papers.

- **R1 H2O (2023):** attention-history/recency retention. https://arxiv.org/abs/2306.14048
- **R2 StreamingLLM (2023):** sinks and recent windows. https://arxiv.org/abs/2309.17453
- **R3 SnapKV (2024):** observation-window, head-specific prompt KV selection. https://arxiv.org/abs/2404.14469
- **R4 Ada-KV:** adaptive head budget allocation. https://arxiv.org/abs/2407.11550
- **R5 SGLang:** structured generation and RadixAttention prefix reuse. https://arxiv.org/abs/2312.07104
- **R6 CacheBlend:** selective recomputation for non-prefix cached-context fusion. https://arxiv.org/abs/2405.16444
- **R7 A3:** question-relevant cached-context fusion. https://arxiv.org/abs/2511.17560
- **R8 Quest:** query-dependent page selection using key summaries; selection is not necessarily permanent eviction. https://arxiv.org/abs/2406.10774
- **R9 RetrievalAttention:** attention-aware vector retrieval and CPU KV access. https://arxiv.org/abs/2409.10516
- **R10 Learning to Evict / KVP (2026):** learned per-head future-usefulness policies. https://arxiv.org/abs/2602.10238
- **R11 VaSE (2026):** value magnitude and stochastic diversity as eviction factors. https://arxiv.org/abs/2606.03928
- **R12 Runtime-Certified Bounded-Error Quantized Attention (2026):** local error controls and tiered fallback for quantization. https://arxiv.org/abs/2605.20868
- **R13 Error Certificates for KV-Cache Eviction via Randomized Design (2026):** statistical local-error certificates. https://arxiv.org/abs/2607.21475
- **R14 Random Attention (2026):** challenges the value of sophisticated scoring in studied reasoning workloads when prompt protection is controlled. Not a universal result against importance scoring. https://arxiv.org/abs/2609.03430
- **R15 RULER:** diverse synthetic effective-context tests. https://arxiv.org/abs/2404.06654
- **R16 LongBench v2:** realistic long-context understanding/reasoning tasks. https://arxiv.org/abs/2412.15204
- **R17 Semantic-aware prefix eviction (2026):** cross-request learned retention is already studied. https://arxiv.org/abs/2605.18825
- **R18 SparQ Attention:** selective history fetching and bandwidth efficiency. https://arxiv.org/abs/2312.04985
- **R19 Infini-attention:** architectural compressed memory, a distinct longer-term track that must not be conflated with exact KV caching. https://arxiv.org/abs/2404.07143
- **R20 vLLM prefix-cache design:** block ancestry and cache isolation. https://docs.vllm.ai/en/stable/design/prefix_caching/

## 10. Development and publication

Use [CODEX_BUILD_PROMPT.md](CODEX_BUILD_PROMPT.md) for staged implementation. Preserve the current runnable simulator and existing results. Build a new package/adapter incrementally rather than replacing working code with empty architecture stubs.

The paper should progress from a clearly labeled design manuscript to a technical report with model and hardware results. Its eventual title can be **HTIS-Context: Task-Conditioned Importance Control for Reusable Transformer Memory**. Maintain a claim/evidence ledger, dataset manifests, pinned dependencies and reproducible run commands. Do not manufacture publication venues, citations, trained checkpoints, deployments or performance figures.

The original prototype has a real historical commit dated **2025-08-06**: https://github.com/DMike9/AI-Innovation-Portfolio/commit/2d9e33b0fd4881237a3e38052dc873d171da9d52. Today's extension retains its actual date. Document the progression; do not rewrite new work to appear older.
