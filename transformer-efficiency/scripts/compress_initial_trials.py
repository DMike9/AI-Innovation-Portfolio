"""Losslessly compact this session's initial report; retain every raw trial."""
from pathlib import Path
import json
from htis_context.records import write_report

path = Path("transformer-efficiency/results/phase2/pretrained-initial.json")
record = json.loads(path.read_text(encoding="utf-8"))
write_report(record, path)
print(f"Retained {record['raw_trial_count']} raw trials in {record['raw_trials_artifact']}; summary bytes={path.stat().st_size}")
