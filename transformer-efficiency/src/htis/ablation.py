"""Small held-out ablation of attention-mass vs recency mixing weight.

Select using validation seeds only, then report results on separate test seeds.
This is a synthetic diagnostic rather than a generalization claim for LLMs.
"""

import argparse
import json
from pathlib import Path

from .experiment import run_suite

WEIGHTS = (0.0, 0.25, 0.5, 0.7, 0.9, 1.0)


def ablate(*, scenario="mixed", validation_seeds=(1, 2, 3), test_seeds=(101, 102, 103),
           weights=WEIGHTS, steps=160, head_dim=32, budget=24,
           sinks=2, recent=4):
    if not validation_seeds or not test_seeds or not weights:
        raise ValueError("Provide seeds and weights")
    if set(validation_seeds) & set(test_seeds):
        raise ValueError("Validation and held-out test seeds must not overlap")
    candidates = []
    for weight in weights:
        report = run_suite(scenarios=(scenario,), seeds=validation_seeds,
                           steps=steps, head_dim=head_dim, budget=budget,
                           attention_weight=weight, sinks=sinks, recent=recent)
        score = report["aggregate"][scenario]["hybrid"]
        candidates.append({"attention_weight": weight,
                           "validation_normalized_output_mse": score["normalized_output_mse"]["mean"],
                           "validation_target_recall": score["target_recall"]["mean"],
                           "validation_counterfactual_coverage": score["counterfactual_coverage"]["mean"]})
    selected = min(candidates, key=lambda c: (c["validation_normalized_output_mse"],
                                              c["attention_weight"]))
    held_out = run_suite(scenarios=(scenario,), seeds=test_seeds,
                         steps=steps, head_dim=head_dim, budget=budget,
                         attention_weight=selected["attention_weight"], sinks=sinks,
                         recent=recent)
    return {
        "protocol": "Select min validation normalized-output-MSE; report all policies on untouched test seeds.",
        "scenario": scenario, "validation_seeds": list(validation_seeds),
        "test_seeds": list(test_seeds), "candidates": candidates,
        "selected_weight": selected["attention_weight"],
        "held_out": held_out["aggregate"][scenario],
        "limitation": "Synthetic selection and evaluation; hyperparameter choice is workload-dependent."
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["recency", "persistent", "mixed", "late_needle"],
                        default="mixed")
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--sinks", type=int, default=2)
    parser.add_argument("--recent", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = ablate(scenario=args.scenario, validation_seeds=args.validation_seeds,
                    test_seeds=args.test_seeds, steps=args.steps, head_dim=args.head_dim,
                    budget=args.budget, sinks=args.sinks, recent=args.recent)
    print("Selected weight:", report["selected_weight"])
    for name, metrics in report["held_out"].items():
        print(f"{name:<15} error={metrics['normalized_output_mse']['mean']:.4f} "
              f"hit={metrics['target_recall']['mean']:.4f} "
              f"counterfactual_coverage={metrics['counterfactual_coverage']['mean']:.4f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Wrote", args.output)
    return report


if __name__ == "__main__":
    main()
