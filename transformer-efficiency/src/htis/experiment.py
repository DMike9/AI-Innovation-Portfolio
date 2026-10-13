"""Reproducible sparse-vs-dense single-head attention approximation benchmark.

Runs deterministic causal retention *without* any pretrained LLM. The full
history is the offline teacher, never consulted by a streaming cache policy.
Neither output error nor theoretical bytes are proxies for downstream accuracy
or actual GPU throughput. See README for limitations and interpretation.
"""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from .cache import CachePolicy, StreamingKVCache, attention
from .synthetic import make_stream


def counterfactual_effects(weights: np.ndarray, values: np.ndarray,
                           output: np.ndarray) -> np.ndarray:
    """Exact per-token *local attention-output* change when each key/value is removed.

    For softmax attention, y_{-i}=(y-a_i v_i)/(1-a_i). Therefore
    ||y-y_{-i}|| = a_i * ||v_i-y||/(1-a_i). The metric is an offline,
    single-layer diagnostic, NOT model-level attribution or a retention input.
    """
    if len(weights) != len(values) or output.shape != values.shape[1:]:
        raise ValueError("Incompatible counterfactual shapes")
    return (weights * np.linalg.norm(values - output, axis=1)
            / np.maximum(1.0 - weights, 1e-12))


POLICIES = (
    CachePolicy.RECENT, CachePolicy.SINK_RECENT,
    CachePolicy.HEAVY_HITTER, CachePolicy.HYBRID
)
SCENARIOS = ("recency", "persistent", "mixed", "late_needle")


def run_trial(*, scenario: str, seed: int, steps: int = 160,
              head_dim: int = 32, budget: int = 24,
              attention_weight: float = 0.7, sinks: int = 2,
              recent: int = 4) -> dict[str, dict]:
    if budget > steps:
        raise ValueError("Use budget <= steps so the trial includes eviction")
    stream = make_stream(steps, head_dim, seed, scenario)
    caches = {
        p.value: StreamingKVCache(
            budget, policy=p, sink_tokens=sinks, recent_tokens=recent,
            attention_weight=attention_weight
        ) for p in POLICIES
    }
    observations = {p.value: {"error": [], "hit": [], "pairs": [], "kv_bytes": [], "counterfactual_coverage": []}
                    for p in POLICIES}
    full_pair_count = 0
    for t in range(steps):
        # The teacher evaluates the full causal history, but it never drives eviction.
        dense_out, dense_weights = attention(stream.queries[t], stream.keys[:t+1],
                                             stream.values[:t+1])
        effects = counterfactual_effects(dense_weights, stream.values[:t+1], dense_out)
        if t >= budget:
            full_pair_count += t + 1
        for name, cache in caches.items():
            sparse_out, _, positions = cache.step(
                stream.queries[t], stream.keys[t], stream.values[t]
            )
            if t >= budget:  # Exclude trivial warm-up where full and sparse agree.
                sample = observations[name]
                sample["error"].append(float(np.linalg.norm(dense_out - sparse_out) ** 2
                                             / (np.linalg.norm(dense_out) ** 2 + 1e-12)))
                sample["hit"].append(int(stream.targets[t] in positions))
                sample["pairs"].append(len(positions))
                sample["kv_bytes"].append(cache.bytes_used())
                effect_total = effects.sum()
                sample["counterfactual_coverage"].append(
                    float(effects[list(positions)].sum() / effect_total) if effect_total > 0 else 1.0
                )
    results = {}
    for name, obs in observations.items():
        cache = caches[name]
        results[name] = {
            "normalized_output_mse": round(mean(obs["error"]), 6),
            "target_recall": round(mean(obs["hit"]), 6),
            "counterfactual_coverage": round(mean(obs["counterfactual_coverage"]), 6),
            "attention_pair_ratio": round(sum(obs["pairs"]) / full_pair_count, 6),
            "final_kv_bytes_fp16": cache.bytes_used(bytes_per_scalar=2),
            "final_kv_reduction_fraction": round(1 - cache.stored_tokens / steps, 6),
            "tokens_retained": cache.stored_tokens,
            "final_importance_top5": [
                {"position": r.position, "score": round(r.score, 4),
                 "past_attention": round(r.observed_attention, 4),
                 "recency": round(r.recency, 4), "protected": r.protected}
                for r in sorted(cache.importance, key=lambda row: -row.score)[:5]
            ],
        }
    return results


def run_suite(*, scenarios=SCENARIOS, seeds=(7, 17, 29), steps=160,
              head_dim=32, budget=24, attention_weight=0.7,
              sinks=2, recent=4) -> dict:
    if not seeds or not scenarios:
        raise ValueError("Provide at least one seed and scenario")
    trials = {}
    aggregates = {}
    metrics = ("normalized_output_mse", "target_recall", "counterfactual_coverage", "attention_pair_ratio",
               "final_kv_reduction_fraction")
    for scenario in scenarios:
        items = [run_trial(scenario=scenario, seed=int(seed), steps=steps,
                           head_dim=head_dim, budget=budget,
                           attention_weight=attention_weight, sinks=sinks,
                           recent=recent) for seed in seeds]
        trials[scenario] = {str(seed): trial for seed, trial in zip(seeds, items)}
        aggregates[scenario] = {}
        for policy in (p.value for p in POLICIES):
            aggregates[scenario][policy] = {
                metric: {"mean": round(mean(values := [v[policy][metric] for v in items]), 6),
                         "std": round(pstdev(values), 6)}
                for metric in metrics
            }
    return {
        "experiment": "synthetic-single-head-causal-kv-retention",
        "disclaimer": "Synthetic attention proxy ONLY; no LM quality or latency claim.",
        "configuration": {"scenarios": list(scenarios), "seeds": list(seeds),
                          "steps": steps, "head_dim": head_dim, "budget": budget,
                          "attention_weight": attention_weight,
                          "sinks": sinks, "recent": recent},
        "definitions": {
            "normalized_output_mse": "Mean across post-warmup steps of ||sparse-dense||^2/(||dense||^2+1e-12); lower is better.",
            "target_recall": "Fraction of post-warmup steps retaining synthetic query target; higher is better.",
            "counterfactual_coverage": "Fraction of full-history, one-token deletion effects attributable to retained positions; offline diagnostic, higher is better.",
            "attention_pair_ratio": "Number of sparse dot products / full-history dot products for post-warmup steps; not runtime.",
            "final_kv_reduction_fraction": "1 - final_retained_tokens/total_stream_tokens; theoretical K/V storage.",
        },
        "aggregate": aggregates,
        "trials": trials,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--attention-weight", type=float, default=0.7)
    parser.add_argument("--sinks", type=int, default=2)
    parser.add_argument("--recent", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    selected = SCENARIOS if args.scenario == "all" else (args.scenario,)
    result = run_suite(scenarios=selected, seeds=args.seeds, steps=args.steps,
                       head_dim=args.head_dim, budget=args.budget,
                       attention_weight=args.attention_weight,
                       sinks=args.sinks, recent=args.recent)
    for scenario, policies in result["aggregate"].items():
        print(f"\nScenario: {scenario} (seed count={len(args.seeds)})")
        print(f"{'policy':<15} {'target_recall':>14} {'output_error':>14} {'pair_ratio':>12}")
        for policy, metrics in policies.items():
            print(f"{policy:<15} {metrics['target_recall']['mean']:>14.3f} "
                  f"{metrics['normalized_output_mse']['mean']:>14.3f} "
                  f"{metrics['attention_pair_ratio']['mean']:>12.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")
    return result


if __name__ == "__main__":
    main()
