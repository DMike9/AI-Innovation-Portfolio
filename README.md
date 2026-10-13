# AI Innovation Portfolio | Applied AI Research & Engineering

A public portfolio of **inspectable implementations and reproducible experiments**, with a focus on how AI systems rank information, allocate attention and operate under limited compute or memory budgets.

The flagship research track is **HTIS-KV (Transformer Efficiency / Token Importance)**. A separate applied track, **SignalRank**, adapts transparent scoring to prioritizing project information. They share a high-level interest in ranking information, but **they are not the same algorithm** and SignalRank does not implement neural attention or a language model.

## Portfolio tracks

### 1. [HTIS-KV: Transformer efficiency research](transformer-efficiency/)

**Question:** Under a bounded decoding KV-cache budget, can online attention-history plus recency keep more useful token positions than straightforward retention rules?

**Implemented and testable:**

- Hard-cap single-head KV retention that physically removes simulated key/value pairs (rather than only applying a gate after expensive attention).
- Explainable per-position importance components: observed attention mass, normalized history, recency, and explicitly protected sink/recent positions.
- Four equal-budget policies: recent-only, attention sinks + recent, heavy-hitter-only, and hybrid.
- Reproducible causal synthetic workload suite with multiple seeds and raw machine-readable metrics.
- Dense attention comparison, counterfactual single-token removal diagnostics, attention comparison counts, and theoretical KV-cache footprint.
- Separate validation/test seed ablation of attention weighting; automated unit tests.
- Paper-by-paper literature discussion, limitations, hypothesis, and a roadmap to actual model/GPU measurements.

**Interpretation:** This is **an honest research prototype**, not a novel published method or a proven speedup of a pretrained transformer. The current measurements use synthetic Q/K/V vectors, not a pretrained LLM. [Read research limitations and citations](transformer-efficiency/RESEARCH.md).

**Run in VS Code:**

```bash
cd transformer-efficiency
python -m pip install -e '.[test]'
pytest -q
python -m htis.experiment --output results/local-run.json
python -m htis.ablation --output results/local-ablation.json
```

### 2. [SignalRank: Explainable priority scoring](signalrank/)

A lightweight React/Vite prototype that ranks project updates and work items using transparent, deterministic rule-based signals (impact, urgency, risk, dependency, strategic alignment and confidence). It runs locally, explains score contributions and recommends follow-up actions.

SignalRank is a **workflow-product concept**, not a transformer model, RAG system, calibrated prediction model or empirical extension of HTIS-KV. Its heuristic scores are designed for demonstration and would require domain validation before operational use.

```bash
cd signalrank
npm ci
npm test
npm run build
npm run dev
```

## Why this portfolio is structured this way

An impressive demo is not the same as evidence. The research track therefore includes **baseline comparisons, configurable parameters, controlled ablations, negative results, raw reproducible outputs, and explicit validity boundaries**. The applied track explores human-facing usability and traceability without misleading readers about model sophistication.

## Research foundations

The research track discusses [H2O](https://arxiv.org/abs/2306.14048), [Scissorhands](https://arxiv.org/abs/2305.17118), [StreamingLLM](https://arxiv.org/abs/2309.17453), [SnapKV](https://arxiv.org/abs/2404.14469), [Ada-KV](https://arxiv.org/abs/2407.11550), [FlashAttention](https://arxiv.org/abs/2205.14135), [Attention is not Explanation](https://aclanthology.org/N19-1357/), and [RULER](https://arxiv.org/abs/2404.06654). These are source material and comparisons, **not claims that this code reimplements their published methods**.

## Future scope

The next meaningful result would be model-integrated, hardware-measured KV eviction with real language-model tasks and positional correctness. That work is explicitly marked **not implemented** in the [research protocol](transformer-efficiency/RESEARCH.md).
