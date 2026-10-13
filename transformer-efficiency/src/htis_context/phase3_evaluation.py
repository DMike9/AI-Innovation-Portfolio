"""Held-out learned-policy ablations and budgeted survivor-bias recollection."""
from __future__ import annotations
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time

import torch

from .adapter import LlamaReferenceAdapter
from .learned import TrainedPagePolicy, load_estimator
from .offline_labels import CapturePolicy, InterventionPolicy, kl, local_damage
from .pages import BaselinePolicy, MemoryBudget
from .provenance import digest
from .runtime_features import protected_pages


def trained_policy(entry, training_dir, identity, config, layer_count, *, exploration=False):
    model, mean, std, record = load_estimator(training_dir / entry["directory"], identity)
    policy = TrainedPagePolicy(model, mean, std, page_size=config["page_size"], max_pages=config["budget"],
        layer_count=layer_count, prompt_tokens=config["prompt_tokens"], sink_tokens=config["sink_tokens"],
        recent_tokens=config["recent_tokens"], exploration_seed=config["seed"] if exploration else None)
    return policy, record


@torch.inference_mode()
def evaluate(model, tokenizer, dense_adapter, model_config, config, dataset_dir: Path, training_dir: Path, output: Path):
    if output.exists():
        raise FileExistsError("Never overwrite a held-out evaluation")
    output.mkdir(parents=True)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    training = json.loads((training_dir / "report.json").read_text(encoding="utf-8"))
    trials, checks = [], []
    began = time.perf_counter()
    entries = [{"name": name} for name in ("recent", "prompt_random", "attention_history", "query_summary")]
    entries += [{"name": row["directory"], "trained": row} for row in training["models"]]
    # The exploration comparison uses each saved HTIS seed, not an oracle.
    entries += [{"name": row["directory"] + "/explore", "trained": row, "exploration": True}
                for row in training["models"] if row["kind"] == "htis"]
    heldout = [row for row in manifest["traces"] if row["split"] == "test" and row["challenge"] != "value-outlier"]
    # Artificial KV stress is already evaluated in scoring labels; do not call
    # an unperturbed runtime replay of that case the same value-outlier test.
    for number, trace in enumerate(heldout):
        prefix_ids = torch.tensor(trace["prefix_recipe"]["tokens"], device=model_config["device"])
        positions = torch.tensor(trace["prefix_recipe"]["positions"], device=prefix_ids.device)
        prefix = dense_adapter.forward(prefix_ids, position_ids=positions).state
        current = torch.tensor([[trace["current_token_id"]]], device=prefix_ids.device)
        current_tokens = [int(current.item()), *trace["future_token_ids"][:config["horizon"]-1]]
        teacher = []
        state = prefix
        for token in current_tokens:
            full = dense_adapter.forward(torch.tensor([[token]], device=prefix_ids.device), state=state)
            teacher.append(full.logits[:, -1]); state = full.state
        for budget in config["budgets_pages"]:
            local_config = {**config, "budget": budget}
            kv_limit = budget * config["page_size"] * model_config["layers"] * model_config["kv_heads"] * model_config["head_dim"] * 2 * 4
            for entry in entries:
                if "trained" in entry:
                    policy, record = trained_policy(entry["trained"], training_dir, dense_adapter.identity,
                            local_config, model_config["layers"], exploration=entry.get("exploration", False))
                    controller_bytes = sum(p.numel()*p.element_size() for p in policy.estimator.parameters())
                    controller_bytes += policy.mean.numel()*policy.mean.element_size() + policy.std.numel()*policy.std.element_size()
                else:
                    policy = BaselinePolicy(entry["name"], page_size=config["page_size"], max_pages=budget,
                        prompt_tokens=config["prompt_tokens"], sink_tokens=config["sink_tokens"], recent_tokens=config["recent_tokens"], seed=config["seed"])
                    controller_bytes = 0
                adapter = LlamaReferenceAdapter(model, dense_adapter.identity, policy=policy,
                              memory_budget=MemoryBudget(kv_limit, 64 * 1024 * 1024))
                state = prefix
                observations = []
                for step, token in enumerate(current_tokens):
                    run = adapter.forward(torch.tensor([[token]], device=prefix_ids.device), state=state)
                    state = run.state
                    feature_workspace = getattr(policy, "feature_bytes", 0)
                    working = run.metrics["estimated_cache_and_attention_working_tensor_bytes"] + controller_bytes + feature_workspace
                    if working > 64 * 1024 * 1024:
                        raise MemoryError("Controller/feature tensors exceed the charged working estimate")
                    observations.append({"step": step, "kl_to_dense": kl(teacher[step], run.logits[:, -1]),
                        "forward_ms": run.metrics["elapsed_ms"], "selection_ms": run.metrics["selection_ms"],
                        "feature_update_ms": run.metrics["feature_update_ms"], "active_kv_bytes": state.kv_bytes,
                        "cache_metadata_tensor_bytes": run.metrics["metadata_tensor_bytes"],
                        "controller_persistent_tensor_bytes": controller_bytes,
                        "controller_peak_feature_tensor_bytes": feature_workspace,
                        "known_cache_attention_controller_working_estimate_bytes": working,
                        "available_tokens_per_layer": [len(x.slots) for x in state.layers],
                        "omitted_original_slots_per_layer": [state.recipe.length-len(x.slots) for x in state.layers]})
                checks.append({"name": f"{trace['id']}/{entry['name']}/budget-{budget}",
                               "passed": state.kv_bytes <= kv_limit and state.lineage != "dense"})
                trials.append({"trace_id": trace["trace_id"], "document_id": trace["document_id"], "task_family": trace["task_family"],
                    "challenge": trace["challenge"], "policy": entry["name"], "budget_pages": budget, "kv_limit_bytes": kv_limit,
                    "mean_KL_to_dense": sum(x["kl_to_dense"] for x in observations)/len(observations),
                    "steps": observations, "exploration_events": getattr(policy, "exploration_events", [])})
        print(f"Held-out runtime {number+1}/{len(heldout)} {trace['id']} complete", flush=True)
    # Recollect diagnostic labels under trained-policy survivor states on TRAIN
    # documents only. These are NOT fed back into this run's held-out fitting.
    recollection = recollect(model, dense_adapter, model_config, config, manifest, training, training_dir, output)
    aggregate = []
    for budget in config["budgets_pages"]:
        for entry in entries:
            rows = [r for r in trials if r["policy"] == entry["name"] and r["budget_pages"] == budget]
            aggregate.append({"policy": entry["name"], "budget_pages": budget, "examples": len(rows),
                "mean_KL_to_dense": sum(r["mean_KL_to_dense"] for r in rows)/len(rows),
                "mean_forward_ms": sum(s["forward_ms"] for r in rows for s in r["steps"])/sum(len(r["steps"]) for r in rows),
                "mean_selection_ms": sum(s["selection_ms"] for r in rows for s in r["steps"])/sum(len(r["steps"]) for r in rows)})
    report = {"experiment": "phase3-heldout-policy-ablation-small-study", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config, "execution_identity": manifest["execution_identity"], "checks": checks,
        "passed": all(c["passed"] for c in checks), "aggregate": aggregate, "trials": trials,
        "recollection": recollection, "wall_seconds": time.perf_counter()-began,
        "limitations": ["CPU eager timings are diagnostic, with no optimized-kernel or speedup claim.",
            "Runtime includes all-layer score extrapolation from three training layers.",
            "Equal physical KV protection and budgets; learned feature/weight overhead is separately charged.",
            "Exploration replaces one free page among currently available pages; no archive fetch or evicted-page recovery.",
            "Recollection is train-document diagnostic evidence, not a second round of training or test-set tuning."]}
    from .records import write_report
    write_report(report, output / "report.json")
    return report


@torch.inference_mode()
def recollect(model, dense_adapter, model_config, config, manifest, training, training_dir, output):
    entry = next(r for r in training["models"] if r["kind"] == "htis" and r["seed"] == config["training_seeds"][0])
    train_traces = [r for r in manifest["traces"] if r["split"] == "train" and r["task_family"] == config["train_tasks"][0]][:2]
    features_path, labels_path = output/"recollection.features.jsonl.gz", output/"recollection.labels.jsonl.gz"
    diagnostics = []
    with gzip.open(features_path, "wt", encoding="utf-8") as feature_stream, gzip.open(labels_path, "wt", encoding="utf-8") as label_stream:
        for trace in train_traces:
            for exploration in (False, True):
                ids = torch.tensor(trace["prefix_recipe"]["tokens"], device=model_config["device"])
                prefix = dense_adapter.forward(ids).state
                policy, _ = trained_policy(entry, training_dir, dense_adapter.identity, {**config,"budget":max(config['budgets_pages'])}, model_config["layers"], exploration=exploration)
                adapter = LlamaReferenceAdapter(model, dense_adapter.identity, policy=policy)
                current_tokens = [trace["current_token_id"], *trace["future_token_ids"][:config["horizon"]-1]]
                for token in current_tokens:
                    prefix = adapter.forward(torch.tensor([[token]],device=ids.device), state=prefix).state
                next_token = torch.tensor([[trace["future_token_ids"][-1]]],device=ids.device)
                capture = CapturePolicy(config["page_size"], model_config["layers"])
                reference = LlamaReferenceAdapter(model,dense_adapter.identity,policy=capture).forward(next_token,state=prefix)
                layer_index = config["sample_layers"][1]
                observation = capture.calls[layer_index]
                batch, kv = observation["features"], reference.state.layers[layer_index]
                protected = protected_pages(kv.slots, now=batch.now, page_size=config["page_size"],
                            prompt_tokens=config["prompt_tokens"],sink_tokens=config["sink_tokens"],recent_tokens=config["recent_tokens"])
                candidates = [p for p in batch.page_ids if p not in protected][:1]
                record_id = digest([trace["trace_id"], exploration, "recollection"])
                for i,page in enumerate(batch.page_ids):
                    feature_stream.write(json.dumps({"recollection_id":record_id,"document_id":trace["document_id"],"split":"train-diagnostic",
                        "exploration":exploration,"page_id":page,"values":batch.values[i].tolist(),"available_history_only":True})+'\n')
                if candidates:
                    removed = tuple(candidates)
                    intervention=InterventionPolicy(config["page_size"],layer_index,removed,len(batch.page_ids))
                    changed=LlamaReferenceAdapter(model,dense_adapter.identity,policy=intervention).forward(next_token,state=prefix)
                    label_stream.write(json.dumps({"recollection_id":record_id,"split":"train-diagnostic","omitted_pages":removed,
                        "next_token_kl":kl(reference.logits,changed.logits),"local_output_l2":local_damage(observation["query"],kv,removed,config["page_size"]),
                        "reference":"same supplied approximate survivor state; not historical dense equivalence"})+'\n')
                diagnostics.append({"document_id":trace["document_id"],"exploration":exploration,
                    "sampled_layer":layer_index,"available_original_pages":list(batch.page_ids),
                    "missing_original_pages":len(set(range((batch.now+config['page_size'])//config['page_size']))-set(batch.page_ids)),
                    "charged_exploration_events":len(policy.exploration_events),"lineage":prefix.lineage,
                    "feature_tensor_bytes":batch.values.numel()*batch.values.element_size()})
    import hashlib
    return {"diagnostics":diagnostics,"features_file":features_path.name,"labels_file":labels_path.name,
            "features_sha256":hashlib.sha256(features_path.read_bytes()).hexdigest(),"labels_sha256":hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            "not_used_for_model_selection_or_training":True}
