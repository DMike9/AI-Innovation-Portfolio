# HTIS-Context | Transformer Inference & Adaptive KV Memory

**Daniel Michael · AI Innovation Portfolio**

Research into how transformer systems decide **which computed context to reuse, which memory to retain, and when additional computation is worth its cost**.

The flagship direction is **HTIS-Context: Task-Conditioned Importance Control for Reusable Transformer Memory**. It develops the original Hierarchical Token Importance Scoring concept into a research program spanning query-conditioned importance, reusable KV states, recoverable memory, and measurable quality/latency/memory tradeoffs.

[Architecture and mathematical reference](transformer-efficiency/HTIS_CONTEXT_ARCHITECTURE.md) · [Experimental protocol](transformer-efficiency/RESEARCH.md) · [Phased implementation brief](transformer-efficiency/CODEX_BUILD_PROMPT.md)

## The research question

A transformer may see the same context repeatedly, yet need different parts of it for each task. Can a hierarchical controller coordinate exact prefix reuse and in-request KV access better than a strong combination of conventional caching and retention policies?

The target is not a universal token-importance number. It is **conditional utility**: the expected effect of retaining, omitting, fetching or recomputing a piece of model state, given the current query, layer, memory budget and action cost.

## Research components

| Component | Current evidence or status |
|---|---|
| HTIS-KV online retention sandbox | Implemented single-head synthetic simulator with equal-budget policies and reproducible results |
| Value-sensitive omission bound | Implemented NumPy reference, derivation, and numerical/property-style tests against dense attention and brute-force optimization |
| Learned page-importance controller | Phase 3 small CPU implementation: counterfactual labels, 16 trained baseline/fusion/ablation checkpoints; held-out benefit over recency was not established |
| Pretrained reference adapter | Implemented CPU FP32 SmolLM2-135M adapter; 97 upstream-equivalence checks and 115 combined tests passed |
| Exact prefix reuse and physical KV selection | Implemented initial reference with provenance validation, LRU/LFU and six heuristic policies; 105 real-model smoke checks passed |
| Recoverable tiered KV memory | Specified for Phase 4; in-request archive/fetch and runtime bound are not implemented |
| Model-quality and GPU efficiency study | Evaluation protocol and composed baselines specified; no pretrained-model speedup claim yet |
| SignalRank | Preserved as a separate applied workflow demo, not evidence of transformer efficiency |

## What makes the next experiment substantive

**Output-sensitive importance.** Attention mass alone does not quantify the effect of removing a value. The new mathematical reference bounds a fixed attention head's omission error using key-coordinate bounds and value centroid/radius summaries. Its tests include a low-attention, high-value outlier and an identical-value case. The bound is local to a supplied Q/K/V snapshot, not a guarantee about the final generated answer.

**Two timescales.** Exact prefix reuse targets repeated prefill; query-conditioned memory selection targets decode access and residency. Their effects will be measured separately and together.

**Recoverable memory.** The proposed controller distinguishes GPU residency from CPU archival and source recomputation. It accounts for transfer and metadata costs rather than treating external memory as free.

**Architectural correctness.** Original positions, RoPE, masks, grouped-query attention, cache ancestry, access scope and computation lineage are explicit implementation requirements. Semantic embeddings route candidates; they do not replace source content or authorize exact KV reuse.

**A high comparison bar.** The study will compare against exact prefix caching plus strong simple policies, including prompt-protected random retention where applicable. Learned scoring must demonstrate benefit after paying for its own computation.

## Run the implemented research

```bash
cd transformer-efficiency
python -m pip install -e '.[test]'
pytest -q
python -m htis.experiment --output results/local-run.json
python -m htis.ablation --output results/local-ablation.json
```

The existing simulator compares recent-only, sinks+recent, attention-history and hybrid retention. It includes fixed seeds, negative workloads, counterfactual diagnostics and validation/test separation. These outputs are synthetic Q/K/V experiments, not language-model accuracy or measured GPU acceleration.

The additional bound implementation lives in [`bounded_omission.py`](transformer-efficiency/src/htis/bounded_omission.py), with [tests](transformer-efficiency/tests/test_bounded_omission.py) and a [derivation and assumptions](transformer-efficiency/HTIS_CONTEXT_ARCHITECTURE.md#5-implemented-mathematical-reference-pagewise-omission-bound).

## Build the real-model extension in VS Code

The initial real-model extension now lives in [`htis_context`](transformer-efficiency/src/htis_context/).
Read [implementation status](IMPLEMENTATION_STATUS.md), [research plan](RESEARCH_PLAN.md),
[claim/evidence ledger](CLAIM_EVIDENCE.json), and [actual commands/outcomes](transformer-efficiency/COMMANDS_EXECUTED.md)
before continuing. Phase 3 counterfactual labels and learned baselines are now
recorded under `transformer-efficiency/results/phase3`; Phase 4 recovery/routing is next.

```bash
cd transformer-efficiency
# In an activated, isolated environment; CPU-only pinned dependencies:
python -m pip install -r requirements-context.lock --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps
python -m htis_context.cli download --model-dir ../.model-cache/smollm2-135m
python -m htis_context.cli equivalence --output results/phase1/my-equivalence.json
python -m htis_context.cli phase2 --output results/phase2/my-smoke.json
```

The [dense-equivalence report](transformer-efficiency/results/phase1/equivalence-final.json)
and [Phase 2 report](transformer-efficiency/results/phase2/pretrained-final.json) use
genuine pretrained weights. The latter links losslessly compressed per-example
records: three authored short-text sources, two repeats, reuse on/off and six
policies. This is a CPU correctness/quality smoke example, not an official
benchmark. Query-summary selection produced nonzero divergence and slightly
slower mean forward time than the full reference in this run. That Phase 2
experiment did not train HTIS and establishes no long-context task correctness
or GPU speedup.

Open the repository in VS Code and give Codex the [implementation brief](transformer-efficiency/CODEX_BUILD_PROMPT.md). The sequence is: establish dense-model equivalence, implement physical KV selection and exact reuse, train counterfactual importance estimators, add recoverable/cost-aware memory, run controlled experiments, and build the paper and research explorer from recorded evidence.

The architecture note is a research proposal with an implemented mathematical component. A publishable performance contribution remains a hypothesis to test, not a result inferred from the design.

## Literature

The architecture includes a paper-by-paper [literature and novelty ledger](transformer-efficiency/HTIS_CONTEXT_ARCHITECTURE.md#9-literature-and-novelty-ledger): H2O, StreamingLLM, SnapKV, Ada-KV, SGLang, CacheBlend, A3, Quest, RetrievalAttention, KVP, VaSE, Random Attention, SparQ and local-error certification work, with RULER and LongBench v2 informing evaluation.

Published mechanisms are attributed. The proposed contribution must be established through implementation, composed baselines, controlled ablations, actual hardware measurements and explicit failure analysis.

## Research progression

- **August 6, 2025:** the [original HTIS transformer prototype](https://github.com/DMike9/AI-Innovation-Portfolio/commit/2d9e33b0fd4881237a3e38052dc873d171da9d52) was committed. It remains available as the starting artifact.
- **October 13, 2026:** HTIS-KV reproducible simulations were added, followed by the HTIS-Context architecture and value-sensitive omission-bound reference.

New work retains its real dates. The historical prototype demonstrates the origin of the research direction, not that today's proposed architecture was already deployed then.

## Applied demonstration: SignalRank

[SignalRank](signalrank/) is a React/Vite demo that ranks project updates with deterministic category and keyword rules. It explores workflow usability and explainable score contributions. It is neither a trained transformer nor a validation of the HTIS-Context architecture.

```bash
cd signalrank
npm ci
npm test
npm run build
npm run dev
```

The demo and its [research boundaries](signalrank/RESEARCH_BOUNDARIES.md) remain available without shifting the flagship away from transformer research.
