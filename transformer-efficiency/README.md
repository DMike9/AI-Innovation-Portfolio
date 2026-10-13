# HTIS-KV | Token Importance and Budgeted Transformer Inference

**Status: reproducible research sandbox, not a model optimization or a published method.**

This project asks a concrete transformer-efficiency question: **under a fixed decoding KV-cache budget, which past token positions should be retained to preserve useful attention outputs?** It replaces an untrained post-attention *soft gate* as the primary experiment with a measurable *hard KV-retention* simulator. The original `src/filename.py` is retained as a legacy educational demo, not presented as evidence of computational savings.

## The research question

A transformer decoder stores past key and value vectors so future queries can attend to them. For a single head of dimension `d`, a `T`-position FP16 KV cache requires `2 × T × d × 2` bytes, excluding allocator overhead and other model state. Retaining only `B` positions reduces the **theoretical KV vector storage** to `2 × B × d × 2` bytes, but can discard evidence required by later queries. A lower cache budget alone does not prove GPU latency improvements or preserve language-model quality.

We compare four online policies under exactly the same token budget:

| Policy | Retention rule | Main motivation |
|---|---|---|
| `recent` | Keep last `B` positions | Sliding-window baseline |
| `sink_recent` | Keep initial sink positions and remaining newest positions | StreamingLLM-inspired sink retention |
| `heavy_hitter` | Keep current position and tokens with high observed decayed attention | H2O / Scissorhands-inspired history |
| `hybrid` | Reserve sinks + recent tokens; fill the remaining budget using weighted attention history and recency | Experimental HTIS-KV combination |

These are **independent simplified baselines**, **not faithful implementations** of H2O, StreamingLLM, Scissorhands, or Ada-KV. Attention-sink positions are reserved without claiming semantic importance. Cache decisions see only previous *sparse* attention scores, never the future query or offline full-history attention.

### Experimental importance score

For cached position `i` at streaming step `t`:

```text
past_mass_i(t-1) = exponentially decayed sum of previously observed sparse attention
attention_component_i = past_mass_i / max_j(past_mass_j)   (zero when all are zero)
recency_component_i = exp(-(t - i) / tau)
importance_i = alpha * attention_component_i + (1-alpha) * recency_component_i
```

Initial sink positions and the last `recent_tokens` positions are **hard protected** in `hybrid`; their survival is not solely determined by the numeric score. Eviction uses deterministic ties (newer positions first). `alpha=0.7`, `tau=16`, `decay=0.98`, `sink_tokens=2`, and `recent_tokens=4` are **configurable research choices**, not learned/calibrated parameters. Stored scores are heuristics, **not ground-truth token relevance, causal explanations, or confidence probabilities**.

## Quick start (VS Code / Python 3.10+)

```bash
cd transformer-efficiency
python -m venv .venv
# Activate the environment: Windows: .venv\Scripts\activate ; Unix: source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
python -m htis.experiment --output results/my-synthetic-run.json
python -m htis.ablation --output results/my-heldout-ablation.json
```

No model weights, API keys, GPUs, or dataset downloads are needed. The original PyTorch demo can still be run separately after installing `pip install -e '.[legacy]'`, using `python src/filename.py` (it opens a plot).

### Repository layout

```text
transformer-efficiency/
  src/filename.py              # Preserved legacy dense gating demo (no speedup claim)
  src/htis/cache.py            # Online importance scores + hard bounded KV cache
  src/htis/synthetic.py        # Synthetic known-target workload generator
  src/htis/experiment.py       # Dense teacher vs four causal retention policies
  src/htis/ablation.py         # Validation-only weight selection; held-out seeds
  tests/                       # Causality, retention, invariants and fidelity checks
  results/                     # Reproducible example metrics (synthetic only)
  RESEARCH.md                  # Hypotheses, literature, validity and next steps
  pyproject.toml
```

## What is measured?

- **Target recall:** Was the designated synthetic query target retained? (higher is better)
- **Normalized attention-output error:** `mean(||y_sparse - y_full||²/(||y_full||²+1e-12))` over post-warmup steps. Smaller is better; may exceed 1.
- **Counterfactual coverage:** fraction of the full-history, single-token removal effects attributable to cached tokens. This is a *local attention-output* diagnostic computed by the **offline evaluator**, not a model-level attribution method. Higher is better.
- **Attention pair ratio:** number of sparse query-key comparisons divided by full-history comparisons. This is a rough **arithmetic count**, **not a speed benchmark**.
- **Final FP16 KV bytes:** theoretical K and V vector bytes, excluding all other memory use.

All comparisons use the **same budget**. The warmup period (`t < B`) is excluded from quality metrics because no eviction has occurred. By construction, all policies use the same budget and the same pair count, so a policy is not faster than its alternatives on this metric.

## Reproduced synthetic results

Run with 160 steps, head dimension 32, budget 24, seeds 7/17/29. The following are **averages over three deterministic synthetic seeds**, NOT large-language-model results. See [`results/synthetic-baseline.json`](results/synthetic-baseline.json) for raw trials and standard deviations.

| Workload | Recent target recall | Heavy-hitter target recall | Hybrid target recall | Hybrid output error |
|---|---:|---:|---:|---:|
| Mostly recent | 1.000 | 0.025 | 0.799 | 0.238 |
| Persistent early targets | 0.059 | 1.000 | 1.000 | 0.005 |
| Mixed historical / recent | 0.463 | 0.564 | 0.892 | 0.145 |
| Late first-use needle | 0.412 | 0.613 | **0.338** | **0.647** |

**Failure is part of the result:** the hybrid policy can discard a previously irrelevant position before a future query needs it, and does poorly on the deliberately late-used needle. The heavy-hitter-only strategy fails on recency-heavy streams. Results depend strongly on how synthetic queries are constructed.

For a separate *mixed-workload* ablation, three validation seeds selected `alpha=0.5` from a preset grid; three different held-out seeds produced hybrid target recall **0.934**, versus recent **0.471**, with a normalized attention-output error of **0.087** versus recent **0.629**. This is a held-out *synthetic* result, **not proof of superiority on real language models**. See [`results/ablation-mixed.json`](results/ablation-mixed.json).

The cache ends with 24 of 160 positions, corresponding to a theoretical **85% reduction in KV vector storage at the final step**. The post-warmup comparison count is approximately **25.9%** of full-history attention. Neither number is a GPU speedup, and neither includes the cost of maintaining scores.

## Research boundaries and next evaluation

The simulator is single-head, single-layer, nontraining, and operates on synthetic Q/K/V. It omits actual LLM tokenization, RoPE/position handling, prefill semantics, attention kernels, model perplexity, multi-head adaptive budgets, generation quality, fragmentation, and throughput. A random untrained token gate (the legacy file) cannot establish efficiency gains. Attention weights are not faithful explanations of predictions; compare with *Attention is not Explanation* before making attribution claims.

To test the approach as a candidate inference optimization, the next stage must integrate an established decoder with correct per-layer/per-head KV cache handling and position semantics; compare full cache, recent window, sink + recent, and head-aware baselines at equal budget; run LongBench/RULER-style tasks; report held-out accuracy/perplexity alongside real `tokens/s`, peak allocated GPU bytes and latency; record hardware, model, precision, kernels, batch size and token lengths. Follow the experiment protocol in [RESEARCH.md](RESEARCH.md).

## Sources

Primary papers, their specific influence, and limitations are indexed in [RESEARCH.md](RESEARCH.md). No algorithms or paper results are claimed as original inventions or copied implementations.
