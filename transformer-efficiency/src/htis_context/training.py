"""Offline supervised fitting, validation-only selection and held-out ablations."""
from __future__ import annotations
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .learned import RiskEstimator
from .runtime_features import SCHEMA, WIDTH
from .provenance import digest


KINDS = ("linear", "mlp", "htis", "no_attention", "no_locality", "no_content", "no_query", "no_value")


def read_dataset(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "htis-counterfactual-dataset-v1" or manifest["feature_schema"] != SCHEMA:
        raise ValueError("Dataset schema mismatch")
    data = []
    for file_key, sha_key in (("features_file", "features_sha256"), ("labels_file", "labels_sha256")):
        path = directory / manifest[file_key]
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[sha_key]:
            raise ValueError("Trace artifact checksum mismatch")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            data.append([json.loads(x) for x in stream])
    features, labels = data
    mapping = {r["feature_id"]: r for r in features}
    if len(mapping) != len(features):
        raise ValueError("Duplicate feature IDs")
    document_splits = {}
    for row in features:
        if document_splits.setdefault(row["document_id"], row["split"]) != row["split"]:
            raise ValueError("Document split leakage")
        if row["schema"] != SCHEMA or len(row["values"]) != WIDTH:
            raise ValueError("Unexpected feature fields or width")
    singles = []
    for label in labels:
        selected = [mapping[i] for i in label["feature_ids"]]
        if any(row["split"] != label["split"] or row["trace_id"] != label["trace_id"] or row["protected"] for row in selected):
            raise ValueError("Invalid label join or protected-page intervention")
        if len(selected) == 1:
            singles.append({**selected[0], "label": label})
    return manifest, singles, labels


def average_ranks(values):
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[start]] == values[order[stop]]:
            stop += 1
        result[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return result


def metrics(rows, prediction, residual_radius):
    actual = np.array([r["label"]["horizon_kl"] for r in rows])
    target = np.log1p(1000 * actual)
    if len(actual) == 0:
        return {"count": 0}
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row["trace_id"], row["layer"]), []).append(i)
    correlations, capture = [], []
    for indices in groups.values():
        y, score = actual[indices], prediction[indices]
        rank_y, rank_p = average_ranks(y), average_ranks(score)
        if rank_y.std() > 0 and rank_p.std() > 0:
            correlations.append(float(np.corrcoef(rank_y, rank_p)[0, 1]))
        k = min(2, len(indices))
        best = np.sort(y)[-k:].sum()
        chosen = y[np.argsort(score, kind="stable")[-k:]].sum()
        capture.append(float(chosen / best) if best > 0 else 1.0)
    return {"count": len(rows), "log_target_mse": float(np.mean((prediction-target)**2)),
            "mean_group_spearman": float(np.mean(correlations)) if correlations else None,
            "top2_damage_capture_on_labeled_candidates": float(np.mean(capture)),
            "validation_residual_interval_coverage": float(np.mean(np.abs(prediction-target) <= residual_radius)),
            "note": "Scores target intervention damage; ranking/calibration are not end-task accuracy."}


def fit(directory: Path, output: Path, config: dict) -> dict:
    from safetensors.torch import save_file
    if output.exists():
        raise FileExistsError("Do not overwrite trained weights or reports")
    manifest, singles, labels = read_dataset(directory)
    partitions = {name: [r for r in singles if r["split"] == name] for name in ("train", "validation", "test")}
    if any(not rows for rows in partitions.values()):
        raise ValueError("Missing train/validation/test examples")
    if any(r["task_family"] == config["heldout_task"] for name in ("train", "validation") for r in partitions[name]):
        raise ValueError("Task-family holdout leakage")
    torch.set_num_threads(4)
    xs = {name: torch.tensor([r["values"] for r in rows]) for name, rows in partitions.items()}
    ys = {name: torch.tensor([np.log1p(1000 * r["label"]["horizon_kl"]) for r in rows], dtype=torch.float32) for name, rows in partitions.items()}
    mean = xs["train"].mean(0)
    std = xs["train"].std(0, unbiased=False).clamp(min=1e-4)
    xs = {name: (x - mean) / std for name, x in xs.items()}
    output.mkdir(parents=True)
    records = []
    start = time.perf_counter()
    for kind in KINDS:  # linear and MLP finish before HTIS/fusion complexity.
        for seed in config["training_seeds"]:
            torch.manual_seed(seed)
            model = RiskEstimator(kind, config["hidden_width"])
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0.001)
            best_loss, best_epoch, best_weights = float("inf"), -1, None
            began = time.perf_counter()
            curve = []
            for epoch in range(config["epochs"]):
                model.train(); optimizer.zero_grad()
                loss = torch.mean((model(xs["train"]) - ys["train"]) ** 2)
                loss.backward(); optimizer.step(); model.eval()
                with torch.no_grad():
                    validation_loss = float(torch.mean((model(xs["validation"]) - ys["validation"]) ** 2))
                curve.append({"epoch": epoch, "train_loss": float(loss.detach()), "validation_loss": validation_loss})
                if validation_loss < best_loss:
                    best_loss, best_epoch = validation_loss, epoch
                    best_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
            model.load_state_dict(best_weights); model.eval()
            with torch.no_grad():
                validation_prediction = model(xs["validation"]).numpy()
                test_prediction = model(xs["test"]).numpy()
            radius = float(np.quantile(np.abs(validation_prediction - ys["validation"].numpy()), config["calibration_quantile"]))
            model_dir = output / f"{kind}-seed-{seed}"
            model_dir.mkdir()
            save_file({**best_weights, "feature_mean": mean, "feature_std": std}, str(model_dir / "weights.safetensors"))
            runtime_files = ("adapter.py", "pages.py", "runtime_features.py", "learned.py")
            record = {"kind": kind, "seed": seed, "hidden_width": config["hidden_width"], "feature_schema": SCHEMA,
                "feature_width": WIDTH, "target": "log1p(1000 * two-step-mean teacher-forced intervention KL)",
                "best_epoch": best_epoch, "selection": "minimum validation log-target MSE; no test tuning",
                "trainable_parameters": sum(p.numel() for p in model.parameters()), "training_wall_seconds": time.perf_counter()-began,
                "execution_identity": manifest["execution_identity"], "dataset_manifest_sha256": hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
                "weights_sha256": hashlib.sha256((model_dir/'weights.safetensors').read_bytes()).hexdigest(),
                "weights_bytes": (model_dir/'weights.safetensors').stat().st_size,
                "runtime_source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in runtime_files},
                "validation_residual_log_target_radius": radius, "calibration_quantile": config["calibration_quantile"],
                "validation": metrics(partitions["validation"], validation_prediction, radius), "held_out": {},
                "curve": curve}
            for group, predicate in [("seen_tasks", lambda r: r["task_family"] in config["train_tasks"]),
                                      ("task_family_holdout", lambda r: r["task_family"] == config["heldout_task"]),
                                      ("stress_holdout", lambda r: r["challenge"] != "natural-public-text")]:
                indices = [i for i, r in enumerate(partitions["test"]) if predicate(r)]
                record["held_out"][group] = metrics([partitions["test"][i] for i in indices], test_prediction[indices], radius)
            (model_dir / "model.json").write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding="utf-8")
            records.append({k: v for k, v in record.items() if k != "curve"} | {"directory": model_dir.name})
            print(f"Trained {kind} seed={seed}: epoch={best_epoch}, validation MSE={best_loss:.4f}", flush=True)
    # Pair labels expose nonadditivity; never pretend summed single-page risk
    # equals the joint intervention, or use joint effects as runtime features.
    single_map = {(r["trace_id"], r["layer"], r["omitted_pages"][0]): r["horizon_kl"] for r in labels if len(r["omitted_pages"]) == 1}
    interactions = []
    for row in labels:
        if len(row["omitted_pages"]) > 1:
            summed = sum(single_map[(row["trace_id"], row["layer"], p)] for p in row["omitted_pages"])
            interactions.append({"action_id": row["action_id"], "split": row["split"], "joint_kl": row["horizon_kl"],
                                 "sum_single_kl": summed, "interaction_residual": row["horizon_kl"] - summed})
    report = {"experiment": "phase3-trained-counterfactual-scoring-small-study", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config, "dataset_manifest_sha256": hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
        "split_documents": {name: sorted({r["document_id"] for r in rows}) for name, rows in partitions.items()},
        "split_single_rows": {name: len(rows) for name, rows in partitions.items()}, "models": records,
        "multi_page_interactions": interactions, "total_training_wall_seconds": time.perf_counter()-start,
        "limitations": ["Small sample and two seeds; no universal ranking, calibrated probability or task-quality claim.",
            "Page-level locality uses original page adjacency; no token-level locality convolution.",
            "Labels cover three sampled layers; runtime all-layer scores extrapolate elsewhere.",
            "Model/epoch selection and normalization use training/validation only; task-family holdout remains test-only."]}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding="utf-8")
    return report
