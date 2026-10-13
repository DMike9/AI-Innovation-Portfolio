"""Reproducible model download, inspection, dense inference and acceptance reports."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .checkpoint import download, load
from .equivalence import check_equivalence
from .experiment import run_phase2
from .records import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "inspect", "dense", "equivalence", "phase2"])
    parser.add_argument("--config", type=Path, default=Path("configs/smollm2-135m.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("../.model-cache/smollm2-135m"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--workload", type=Path, default=Path("configs/phase2-smoke.json"))
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command == "download":
        result = download(config, args.model_dir)
    else:
        model, tokenizer, adapter, manifest = load(config, args.model_dir)
        if args.command == "equivalence":
            result = check_equivalence(model, tokenizer, adapter, config)
        elif args.command == "phase2":
            result = run_phase2(model, tokenizer, adapter, config,
                                json.loads(args.workload.read_text(encoding="utf-8")))
        elif args.command == "inspect":
            result = {"identity": asdict(adapter.identity), "model_config": model.config.to_dict(),
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                      "device": str(next(model.parameters()).device), "checkpoint_manifest": manifest}
        else:
            ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(config["device"])
            all_ids = ids.clone()
            state = None
            metrics = []
            for _ in range(args.tokens):
                run = adapter.forward(ids, state=state)
                ids = run.logits[:, -1].argmax(-1, keepdim=True)
                all_ids = torch.cat((all_ids, ids), dim=1)
                state = run.state
                metrics.append(run.metrics)
            result = {"experiment": "pretrained-dense-greedy-smoke", "configuration": config,
                      "prompt": args.prompt, "token_ids": all_ids[0].tolist(),
                      "text": tokenizer.decode(all_ids[0]), "steps": metrics,
                      "identity": asdict(adapter.identity)}
    result["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    if args.output:
        write_report(result, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "steps", "model_config", "trials")}, indent=2))
    if args.command in ("equivalence", "phase2"):
        print(f"Checks: {len(result['checks'])}; failures: {[x['name'] for x in result['checks'] if not x['passed']]}")
        return 0 if result["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
