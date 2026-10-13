"""Offline pretrained intervention dataset built on the existing Phase 2 adapter."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import random
import time

import torch

from .adapter import LlamaReferenceAdapter
from .offline_labels import CapturePolicy, InterventionPolicy, kl, local_damage
from .pages import update_features
from .provenance import digest
from .public_corpus import read_corpus
from .runtime_features import SCHEMA, WIDTH, protected_pages
from .state import _fingerprint


TASKS = {"continue": "Continue this passage in its original style.\nContinuation:",
         "entities": "List the characters or objects mentioned in this passage.\nList:",
         "compare": "Compare the beginning and ending of this passage.\nComparison:"}


def make_sessions(corpus: dict, tokenizer, config: dict):
    sessions = []
    for doc in corpus["documents"]:
        source_ids = tokenizer(doc["text"], add_special_tokens=False).input_ids
        limit = config["max_source_tokens"]
        passage = tokenizer.decode(source_ids[:limit])
        future = source_ids[limit:limit + config["horizon"]]
        if len(future) < config["horizon"]:
            raise ValueError("Public source too short for declared trace horizon")
        tasks = config["train_tasks"] + ([config["heldout_task"]] if doc["split"] == "test" else [])
        for task in tasks:
            prompt = f"Passage:\n{passage}\n\nTask: {TASKS[task]}"
            sessions.append({"id": f"{doc['id']}/{task}", "document_id": doc["id"], "split": doc["split"],
                "task_family": task, "challenge": "natural-public-text", "prompt": prompt,
                "future_token_ids": future, "source_sha256": doc["extract_sha256"]})
    # Authored stress instructions appended to actual public passages. They are
    # explicitly challenge cases, not mislabeled public benchmarks or gold QA.
    doc = next(x for x in corpus["documents"] if x["split"] == "test")
    passage = tokenizer.decode(tokenizer(doc["text"], add_special_tokens=False).input_ids[:config["max_source_tokens"]])
    future = tokenizer(" The updated record is relevant.", add_special_tokens=False).input_ids[:config["horizon"]]
    challenges = {
        "delayed-relevance": "Archive fact: the meeting was in Room Four.\n" + passage + "\nReturn to the early archive fact. Where was the meeting?\nAnswer:",
        "contradictory-update": passage + "\nOld notice: Room Four. Revised notice: Room Seven. Use the revised notice.\nAnswer:",
        "task-switch": passage + "\nEarlier task: list names. Current task: explain the sequence of events.\nResponse:",
        "value-outlier": passage + "\nTask: Continue the passage.\nContinuation:"
    }
    for name, prompt in challenges.items():
        sessions.append({"id": f"{doc['id']}/stress/{name}", "document_id": doc["id"], "split": "test",
            "task_family": "stress-holdout", "challenge": name, "prompt": prompt,
            "future_token_ids": future, "source_sha256": doc["extract_sha256"]})
    return sessions


def validate_splits(sessions, heldout_task):
    mapping = {}
    for row in sessions:
        previous = mapping.setdefault(row["document_id"], row["split"])
        if row["split"] != previous:
            raise ValueError("Document leakage across train/validation/test")
        if row["task_family"] == heldout_task and row["split"] != "test":
            raise ValueError("Held-out task family appears in training/validation")
    if set(mapping.values()) != {"train", "validation", "test"}:
        raise ValueError("Need disjoint train, validation and test documents")


@torch.inference_mode()
def collect(model, tokenizer, dense_adapter, model_config: dict, config: dict,
            corpus_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Never overwrite a trace dataset; choose a new run directory")
    corpus = read_corpus(corpus_path)
    sessions = make_sessions(corpus, tokenizer, config)
    validate_splits(sessions, config["heldout_task"])
    output.mkdir(parents=True)
    (output / "runtime_features").mkdir()
    (output / "offline_labels").mkdir()
    features_path = output / "runtime_features/pages.jsonl.gz"
    labels_path = output / "offline_labels/actions.jsonl.gz"
    trace_records = []
    feature_count = action_count = local_ms = forwards_ms = 0
    started = time.perf_counter()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with gzip.open(features_path, "wt", encoding="utf-8") as feature_stream, gzip.open(labels_path, "wt", encoding="utf-8") as label_stream:
        for number, session in enumerate(sessions):
            ids = tokenizer(session["prompt"], return_tensors="pt").input_ids.to(model_config["device"])
            prefix = dense_adapter.forward(ids[:, :-1]).state
            stress = session["challenge"] == "value-outlier"
            if stress:
                # Controlled value scaling on a *real pretrained* trace; this is
                # not a natural-state experiment and cannot enter exact reuse.
                prefix = prefix.fork()
                layer = prefix.layers[config["sample_layers"][1]]
                chosen = torch.nonzero(layer.slots // config["page_size"] == 1).flatten()
                layer.values[:, :, chosen[:1]] *= 100
                layer.features = update_features(layer.keys, layer.values, layer.slots, config["page_size"], None, layer.slots)
                prefix.lineage = "approximate:controlled-pretrained-value-outlier"
            capture = CapturePolicy(config["page_size"], model_config["layers"])
            teacher_adapter = LlamaReferenceAdapter(model, dense_adapter.identity, policy=capture)
            full = teacher_adapter.forward(ids[:, -1:], state=prefix)
            teacher_logits = [full.logits[:, -1]]
            teacher_state = full.state
            for token in session["future_token_ids"][:config["horizon"] - 1]:
                step = dense_adapter.forward(torch.tensor([[token]], device=ids.device), state=teacher_state)
                teacher_logits.append(step.logits[:, -1]); teacher_state = step.state
            state_hash = _fingerprint(prefix)
            trace_id = digest([session["id"], state_hash, ids[0, -1].item()])
            trace_record = {k: v for k, v in session.items() if k != "prompt"}
            trace_record.update(trace_id=trace_id, prefix_recipe=asdict(prefix.recipe), prefix_state_sha256=state_hash,
                                current_token_id=int(ids[0, -1]), current_slot=ids.shape[1]-1,
                                provenance=asdict(dense_adapter.identity), lineage=prefix.lineage,
                                dense_cpu_kv_bytes=full.state.kv_bytes, page_size=config["page_size"],
                                feature_tensor_bytes=0, intervention_count=0)
            for layer_index in config["sample_layers"]:
                observation = capture.calls[layer_index]
                batch = observation["features"]
                kv = full.state.layers[layer_index]
                protected = protected_pages(kv.slots, now=batch.now, page_size=config["page_size"],
                    prompt_tokens=config["prompt_tokens"], sink_tokens=config["sink_tokens"], recent_tokens=config["recent_tokens"])
                candidates = [p for p in batch.page_ids if p not in protected]
                rng = random.Random(config["seed"] + number * 100 + layer_index)
                chosen = sorted(rng.sample(candidates, min(len(candidates), config["max_single_pages_per_layer"])))
                for page_index, page in enumerate(batch.page_ids):
                    row = {"feature_id": digest([trace_id, layer_index, page]), "trace_id": trace_id,
                        "document_id": session["document_id"], "split": session["split"], "task_family": session["task_family"],
                        "challenge": session["challenge"], "layer": layer_index, "page_id": page, "protected": page in protected,
                        "schema": SCHEMA, "values": batch.values[page_index].cpu().tolist()}
                    feature_stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                    feature_count += 1
                trace_record["feature_tensor_bytes"] += batch.values.numel() * batch.values.element_size()
                interventions = [(p,) for p in chosen]
                if len(chosen) >= 2:
                    for _ in range(config["pairs_per_layer"]):
                        interventions.append(tuple(sorted(rng.sample(chosen, 2))))
                for omitted in interventions:
                    local_start = time.perf_counter()
                    local = local_damage(observation["query"], kv, omitted, config["page_size"])
                    local_ms += (time.perf_counter() - local_start) * 1000
                    policy = InterventionPolicy(config["page_size"], layer_index, omitted, len(batch.page_ids))
                    intervention_adapter = LlamaReferenceAdapter(model, dense_adapter.identity, policy=policy)
                    action_start = time.perf_counter()
                    run = intervention_adapter.forward(ids[:, -1:], state=prefix)
                    kls = [kl(teacher_logits[0], run.logits[:, -1])]
                    changed_state = run.state
                    for horizon_index, token in enumerate(session["future_token_ids"][:config["horizon"] - 1], 1):
                        run = dense_adapter.forward(torch.tensor([[token]], device=ids.device), state=changed_state)
                        changed_state = run.state
                        kls.append(kl(teacher_logits[horizon_index], run.logits[:, -1]))
                    elapsed = (time.perf_counter() - action_start) * 1000
                    forwards_ms += elapsed
                    row = {"action_id": digest([trace_id, layer_index, omitted]), "trace_id": trace_id,
                        "document_id": session["document_id"], "split": session["split"], "task_family": session["task_family"],
                        "challenge": session["challenge"], "layer": layer_index, "omitted_pages": list(omitted),
                        "feature_ids": [digest([trace_id, layer_index, p]) for p in omitted],
                        "local_attention_output_l2": local, "next_token_kl": kls[0], "horizon_kl": sum(kls) / len(kls),
                        "horizon_step_kl": kls, "action_wall_ms": elapsed,
                        "removed_layer_kv_bytes": len([x for x in kv.slots.tolist() if x // config['page_size'] in omitted]) * 2 * model_config["kv_heads"] * model_config["head_dim"] * 4,
                        "lineage_after_intervention": changed_state.lineage}
                    label_stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    action_count += 1; trace_record["intervention_count"] += 1
            trace_records.append(trace_record)
            print(f"Trace {number+1}/{len(sessions)} {session['id']}: {trace_record['intervention_count']} actions", flush=True)
    def sha(path):
        import hashlib
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema": "htis-counterfactual-dataset-v1", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config, "configuration_sha256": digest(config), "model_configuration": model_config,
        "execution_identity": asdict(dense_adapter.identity), "corpus_sha256": corpus["documents_sha256"],
        "feature_schema": SCHEMA, "feature_width": WIDTH, "features_file": "runtime_features/pages.jsonl.gz",
        "labels_file": "offline_labels/actions.jsonl.gz", "features_sha256": sha(features_path), "labels_sha256": sha(labels_path),
        "traces": trace_records, "feature_rows": feature_count, "action_rows": action_count,
        "costs": {"total_wall_seconds": time.perf_counter()-started, "local_label_ms": local_ms,
                  "intervention_and_KL_ms": forwards_ms, "feature_artifact_bytes": features_path.stat().st_size,
                  "label_artifact_bytes": labels_path.stat().st_size, "device": model_config["device"]},
        "limitations": ["Small public-text CPU study; no official QA/benchmark score.",
            "Only three sampled layers train labels; other-layer policy scores are extrapolations.",
            "Locality uses original adjacent pages, not a token-level convolution.",
            "Future teacher-forced text is label-only and not a verified task answer.",
            "Value-outlier stress scales one real pretrained value tensor by 100; not a natural model state.",
            "Trace snapshots are reproducible from tokens/config/source identity; full KV tensors are not persisted."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
