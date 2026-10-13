# HTIS-KV Research Note and Experimental Protocol

**Research status:** engineering replication of motivating *ideas* on a transparent toy system, not a peer-reviewed paper, reproduction of cited implementations, or validation on an LLM.

This note describes the preserved synthetic simulator. The new pretrained
`htis_context` package has separate [implementation status](../IMPLEMENTATION_STATUS.md),
[real-model records](results/phase2/pretrained-final.json) and a
[claim/evidence ledger](../CLAIM_EVIDENCE.json); its evidence does not relabel
these historical synthetic results as model benchmarks.

## 1. Motivation and testable claims

The original HTIS toy model calculated three untrained token scores and multiplied the **already computed** full-attention output by a sigmoid gate. Such multiplication changes representations but leaves QK matrix multiplication, softmax, and cache size essentially unchanged. It therefore **does not demonstrate accelerated attention**. Replacing the post-attention gate with actual KV eviction makes the memory–quality tradeoff measurable, although it does not yet show implementation-level speed.

**Question:** Can inexpensive observed attention history plus recency protect relevant KV positions under a hard budget more consistently than either heuristic alone?

**Hypothesis H1:** For query streams with both persistent historical references and recent references, a hybrid policy will reduce attention-output approximation error versus sliding-window-only and attention-only policies at the same capacity.

**Null H0:** The hybrid offers no reproducible approximation-error advantage in that scenario.

**Counterhypothesis / failure test:** When a token matters only after a long inactive interval, past attention cannot anticipate its usefulness, so an online importance cache can permanently evict it.

## 2. Prior work and exactly what it supports

| Source | What the paper supports | What this repository implements (and does not) |
|---|---|---|
| [H2O: Heavy-Hitter Oracle (2023)](https://arxiv.org/abs/2306.14048) | Frequently attended tokens and recency can be useful cache-retention signals. | A simple accumulated **sparse** attention score. Does **not** reimplement H2O's optimization or claim its latency. |
| [Scissorhands (2023)](https://arxiv.org/abs/2305.17118) | Attention importance often persists across generation steps; fixed KV budgets are meaningful. | Decayed history score with no trained predictor. Persistence is tested, not assumed. |
| [StreamingLLM / Attention Sinks (2023)](https://arxiv.org/abs/2309.17453) | Early tokens can act as attention sinks even when not semantically meaningful. | Protects first `sink_tokens` positions. Not a faithful streaming attention algorithm. |
| [SnapKV (2024)](https://arxiv.org/abs/2404.14469) | A prompt observation window and head-specific selection can compress prefill caches. | Future research direction only; not implemented. |
| [Ada-KV (2024/2025)](https://arxiv.org/abs/2407.11550) | Adaptive allocation across heads can outperform fixed per-head budgets. | Future multi-head extension only; this prototype is one head. |
| [FlashAttention (2022)](https://arxiv.org/abs/2205.14135) | IO-aware tiling can improve **exact** attention efficiency. | Different optimization axis; we make **no** FlashAttention/GPU speed comparison. |
| [Attention is not Explanation (2019)](https://aclanthology.org/N19-1357/) | Attention weights need not be faithful explanations of model predictions. | Scores are retention heuristics, **not explanations**. |
| [Lost in the Middle (2023)](https://arxiv.org/abs/2307.03172) | Long-context utility varies by information position. | Motivates position-stratified failure tests. No LLM evaluation yet. |
| [RULER (2024)](https://arxiv.org/abs/2404.06654) | Evaluating effective long context requires more than a single retrieval task. | Future external evaluation benchmark, not used in synthetic experiments. |

**Novelty boundary:** A weighted combination of retention signals and reserved slots is not claimed as a novel, publishable algorithm. The useful contribution here is a reproducible, inspectable benchmark with explicit failure modes and a transition plan toward more realistic evaluation.

## 3. Online protocol (no future leakage)

1. Generate a deterministic sequence of keys and values and query targets (synthetic data generation may choose targets in advance, but a policy cannot inspect future targets).
2. At step `t`, insert current K/V position, select at most `B` positions using *previously observed sparse attention* and `t`-available metadata, and calculate sparse attention.
3. Update attention mass **after** the sparse attention has been computed. Never use full-history attention to choose cache survivors.
4. Separately and **offline**, compute exact full-history single-head attention for the same `q_t` and prefix `K_{0:t}, V_{0:t}`.
5. Measure fidelity and target retention only from `t >= B`, comparing policies with exactly the same capacity. Aggregate separately for each scenario and seed.

**Selection score**: `h_i = decay * h_i + sparse_attention_i` after observing each step. Before the next step, normalize by the maximum retained `h_i`, compute `r_i = exp(-(t-i)/tau)`, rank `alpha*h_norm + (1-alpha)*r_i`; initial sinks and a short recent window are reserved in the hybrid. New current position is always preserved. Already evicted keys cannot be recalled, so the score has survivor/selection bias.

**Offline counterfactual probe**: for full attention output `y = sum_i a_i * v_i`, deleting token `i` from the softmax distribution changes output by `delta_i = (a_i/(1-a_i)) * ||v_i-y||`, provided `a_i < 1`. We compute this *analytically* (with numerical floor) solely for measurement. This is a **single-attention-output perturbation**, NOT causal model attribution, and is not available to the cache online.

## 4. Experimental controls

- Workload families: `recency`, `persistent`, `mixed`, `late_needle`.
- Reference conditions: same normalized random keys, values, queries, dimensionality and budget per seed, with fixed seeds and deterministic ties.
- Baselines: recent, sink+recent, heavy-hitter, weighted hybrid.
- Capacity: one `B`-token hard budget per policy, including reserved tokens and the current token.
- Metrics: post-warmup target recall, output error, counterfactual coverage, comparison count, final theoretical FP16 KV footprint.
- Record raw per-seed results and across-seed mean and population standard deviation.
- Weight ablation uses separate **validation** and **test** seeds. Select `alpha` only using validation output error; do not choose it based on test results.

Runnable examples:

```bash
python -m htis.experiment --scenario all --steps 160 --budget 24 --seeds 7 17 29 --output results/reproduce.json
python -m htis.ablation --scenario mixed --validation-seeds 1 2 3 --test-seeds 101 102 103 --output results/ablation-reproduce.json
pytest -q
```

## 5. Threats to validity

**Synthetic construction and oversimplified attention.** Queries deliberately point to token vectors, making a target-recall measure possible but not representative of semantic tasks. No trained language model or token embeddings are used. This demo does not establish perplexity, answer quality, or factual recall in real text.

**One head, one query at a time.** A production model has multiple layers/heads, grouped-query attention possibilities, positional encodings and complex token interactions. A per-head policy and total-budget allocation are nontrivial. Inference with RoPE requires the correct original token positions; simply reindexing keys is incorrect.

**Online feedback and irreversibility.** Attention mass of survivors is measured after eviction, causing selection bias. A never-before-important token can be deleted even if essential later (see `late_needle`). Heavy hitters can crowd out local context without explicit reservations.

**Attention versus explanations.** Attention is not validated causal importance; the offline removal probe only measures a local mathematical change, not downstream generated text quality.

**Arithmetic proxies are not deployment savings.** Deleting vectors in the simulator reduces theoretical K/V bytes and QK comparisons. Python/NumPy bookkeeping, allocator overhead, kernel efficiency, memory transfers, prefill cost and end-to-end runtime are excluded. Actual GPU latency can even regress.

**Small sample and tuned workloads.** Three seeds are provided for reproducibility, not statistical significance. A held-out seed split within the same synthetic workload family is weaker than held-out natural data.

## 6. Research development plan

| Stage | Goal | Acceptance criteria |
|---|---|---|
| 1. Synthetic causal cache simulator (**done**) | Establish reproducible policy tradeoffs. | Four baselines; fixed budget; multi-seed raw artifacts; unit tests. |
| 2. Ablations (**done, synthetic**) | Test importance vs recency mixing. | Tune only on validation seeds; report untouched test-seed scores. |
| 3. Real-model adapter (**not done**) | Add model-aware KV eviction to an open decoder. | Accurate original positional indices, correct cache semantics, reference full-cache reproducibility, numerical tests. |
| 4. Model fidelity (**not done**) | Measure generation degradation. | LongBench/RULER-style diverse tasks; perplexity and answer metrics; stratify by needle age/location and context length. |
| 5. Systems benchmark (**not done**) | Determine actual efficiency. | Matched model/hardware/batch/precision/kernels, prefill vs decode timing, peak allocated GPU memory, throughput/tokens/s, profiler evidence. |
| 6. Head-wise budgets (**not done**) | Test adaptive allocation like Ada-KV. | Equal global budget, adaptive/fixed baselines, ablation against layer and head effects. |

## 7. Attribution and reproduction

Independent study and implementation by this repository's maintainers. The cited studies belong to their respective authors. Benchmarks in `results/` are generated locally using synthetic data, not copied performance claims. The code is intentionally readable and dependency-light so the conclusions can be challenged and extended.
