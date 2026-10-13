# SignalRank: Boundaries Between Workflow Scores and Transformer Importance

SignalRank is a deterministic, human-workflow scoring demo. It assigns heuristic 1–5 dimension values based on categories and text rules, then computes an overall priority. Despite conceptual inspiration from limited attention, **it is not transformer attention, not an embedding model, not RAG, not a probability model, and not trained on labeled priorities**.

## What the current score does and does not mean

- The six dimensions are hand-defined business concepts, not learned neural features.
- Current weights were not empirically calibrated; the product score is not a probability or uncertainty interval.
- Confidence currently contributes to the weighted importance score, which is a product design choice, not a validated statistical confidence estimate. Separating **priority** from **evidence quality** is a sensible future experiment.
- Keyword matching is brittle: negation, lexical collisions, missing evidence, and organization-specific language can produce mistakes. Humans must review results before acting on critical decisions.
- "Why" explanations trace which rules fired. This is rule provenance, not causal model interpretability.

## Route to a rigorously evaluated product

Create a labeled evaluation set with multiple human raters; benchmark against simple keyword rules and unweighted baselines; report rank correlation (e.g., Kendall's tau), pairwise agreement, top-k recall for critical items, false negatives for severe risks, and calibration of any predicted probabilities separately from subjective priority scores. Freeze a held-out test set before tuning weights. Track subgroup and domain drift, and provide an explicit reviewer override path.

For genuine transformer memory/importance experiments, see [HTIS-KV](../transformer-efficiency/). The research and product demos deliberately have separate claims and metrics.
