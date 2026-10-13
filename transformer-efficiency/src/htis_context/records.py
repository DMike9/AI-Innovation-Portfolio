"""Normalized JSON summaries and compressed, lossless per-example trial records."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path


def write_report(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if "trials" in result:
        trials = result.pop("trials")
        raw_path = path.with_suffix(".trials.jsonl.gz")
        with gzip.open(raw_path, "wt", encoding="utf-8") as stream:
            for trial in trials:
                stream.write(json.dumps(trial, separators=(",", ":"), allow_nan=False) + "\n")
        result["raw_trial_count"] = len(trials)
        result["raw_trials_artifact"] = raw_path.name
        result["raw_trials_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
