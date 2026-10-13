"""Small real-pretrained factorial smoke example; not an external benchmark.

Offline dense logits are computed only for measurement and never reach runtime
selection. All bounded policies use identical prompt/sink/recent protection.
Costs include feature updates, selection, prefix copies, and cache compaction.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import time

import torch

from .adapter import LlamaReferenceAdapter
from .pages import BaselinePolicy, MemoryBudget
from .prefix_cache import ExactPrefixCache
from .provenance import SourceRecipe, digest


@torch.inference_mode()
def run_phase2(model, tokenizer, dense_adapter, config: dict, workload: dict) -> dict:
    policy_names = ["full", "recent", "sinks_recent", "prompt_random", "attention_history", "query_summary"]
    rows, checks = [], []
    atol, rtol = config["atol"], config["rtol"]
    for example in workload["examples"]:
        ids = tokenizer(example["text"], return_tensors="pt").input_ids.to(config["device"])
        cut = ids.shape[1] - workload["teacher_forced_steps"] - 1
        if cut <= workload["page_size"] * workload["max_pages"]:
            raise ValueError("Workload too short to exercise actual page omission")
        prefix_ids = ids[:, :cut]
        dense_prefill = dense_adapter.forward(prefix_ids)
        teacher = model(ids, use_cache=False).logits  # measurement only, not a policy input
        cache = ExactPrefixCache(workload["cpu_prefix_budget_bytes"])
        admitted = cache.put(dense_prefill.state)
        checks.append({"name": f"{example['id']}/prefix-admitted", "passed": admitted})
        # Changed source/identity misses. Same final chunk cannot authorize reuse.
        original = dense_prefill.state.recipe
        changed = SourceRecipe(((original.tokens[0][0] + 1,) + original.tokens[0][1:],),
                               original.positions, original.mask)
        checks.append({"name": f"{example['id']}/changed-prefix-miss", "passed": cache.get(dense_adapter.identity, changed) is None})
        checks.append({"name": f"{example['id']}/tenant-miss", "passed": cache.get(replace(dense_adapter.identity, access_scope="tenant-other"), original) is None})
        # Full-capacity mode of every baseline must remain exact on real weights.
        for name in policy_names:
            policy = BaselinePolicy(name, page_size=workload["page_size"], max_pages=10000,
                                    prompt_tokens=workload["prompt_tokens"], sink_tokens=workload["sink_tokens"],
                                    recent_tokens=workload["recent_tokens"], seed=workload["seed"])
            adapter = LlamaReferenceAdapter(model, dense_adapter.identity, policy=policy)
            run = adapter.forward(ids[:, cut:cut+1], state=dense_prefill.state)
            checks.append({"name": f"{example['id']}/{name}/full-capacity-equivalence",
                           "passed": bool(torch.allclose(run.logits[:, -1], teacher[:, cut], atol=atol, rtol=rtol)),
                           "max_abs_error": float((run.logits[:, -1] - teacher[:, cut]).abs().max())})
        for repeat in range(workload["repeats"]):
            for name in policy_names:
                policy = BaselinePolicy(name, page_size=workload["page_size"], max_pages=workload["max_pages"],
                                        prompt_tokens=workload["prompt_tokens"], sink_tokens=workload["sink_tokens"],
                                        recent_tokens=workload["recent_tokens"], seed=workload["seed"])
                # Limits are for selected cache and known attention/cache tensors,
                # not the model's total process memory. Cold prefill is dense.
                per_token = config["layers"] * config["kv_heads"] * config["head_dim"] * 2 * 4
                kv_limit = (ids.shape[1] if name == "full" else workload["max_pages"] * workload["page_size"]) * per_token
                adapter = LlamaReferenceAdapter(model, dense_adapter.identity, policy=policy,
                                               memory_budget=MemoryBudget(kv_limit, 64 * 1024 * 1024))
                for reuse in (False, True):
                    started = time.perf_counter()
                    if reuse:
                        state = cache.get(dense_adapter.identity, original, device=config["device"])
                        if state is None:
                            raise RuntimeError("Expected exact prefix hit")
                    else:
                        state = dense_adapter.forward(prefix_ids).state
                    prefill_or_restore_ms = (time.perf_counter() - started) * 1000
                    raw_steps = []
                    decode_start = time.perf_counter()
                    for step in range(workload["teacher_forced_steps"]):
                        slot = cut + step
                        run = adapter.forward(ids[:, slot:slot+1], state=state)
                        state = run.state
                        reference = teacher[:, slot].float()
                        selected = run.logits[:, -1].float()
                        log_p = reference.log_softmax(-1)
                        log_q = selected.log_softmax(-1)
                        kl = (log_p.exp() * (log_p - log_q)).sum(-1).item()
                        target = int(ids[0, slot+1])
                        raw_steps.append({"step": step, "original_slot": slot,
                            "teacher_forced_token": int(ids[0, slot]), "next_target": target,
                            "next_token_KL_dense_to_policy": kl,
                            "dense_next_token_nll": -float(log_p[0, target]),
                            "policy_next_token_nll": -float(log_q[0, target]),
                            "max_abs_logit_error": float((reference - selected).abs().max()),
                            "dense_argmax": int(reference.argmax(-1)), "policy_argmax": int(selected.argmax(-1)),
                            "metrics": run.metrics, "lineage": state.lineage})
                    decode_ms = (time.perf_counter() - decode_start) * 1000
                    if name == "full":
                        checks.append({"name": f"{example['id']}/full/reuse-{reuse}/repeat-{repeat}",
                                       "passed": all(x["max_abs_logit_error"] <= atol + rtol * teacher.abs().max().item() for x in raw_steps)})
                    else:
                        checks.append({"name": f"{example['id']}/{name}/physical-selection/reuse-{reuse}/repeat-{repeat}",
                            "passed": all(event["key_tokens"] < event["before_selection_tokens"]
                                          for event in raw_steps[0]["metrics"]["attention_inputs"])
                                      and state.kv_bytes <= kv_limit})
                    rows.append({"example_id": example["id"], "source_sha256": digest(example["text"]),
                        "prefix_recipe_sha256": original.key, "prefix_tokens": cut, "token_ids": ids[0].tolist(),
                        "policy": asdict(policy), "reuse": reuse, "repeat": repeat,
                        "cold_prefill_or_warm_restore_ms": prefill_or_restore_ms,
                        "teacher_forced_decode_ms_including_diagnostics": decode_ms,
                        "mean_next_token_KL": sum(x["next_token_KL_dense_to_policy"] for x in raw_steps) / len(raw_steps),
                        "mean_nll_delta": sum(x["policy_next_token_nll"] - x["dense_next_token_nll"] for x in raw_steps) / len(raw_steps),
                        "cpu_prefix_cache_tensor_and_recipe_bytes": cache.bytes_used,
                        "kv_limit_bytes": kv_limit, "raw_steps": raw_steps})
        # One live greedy demonstration using the query-summary heuristic.
        generation_policy = BaselinePolicy("query_summary", page_size=workload["page_size"],
                                max_pages=workload["max_pages"], prompt_tokens=workload["prompt_tokens"],
                                sink_tokens=workload["sink_tokens"], recent_tokens=workload["recent_tokens"])
        generator = LlamaReferenceAdapter(model, dense_adapter.identity, policy=generation_policy)
        dense_state, sparse_state = dense_prefill.state, cache.get(dense_adapter.identity, original)
        dense_token = sparse_token = dense_prefill.logits[:, -1].argmax(-1, keepdim=True)
        dense_tokens, sparse_tokens = [], []
        for _ in range(workload["greedy_steps"]):
            dense_tokens.append(int(dense_token.item())); sparse_tokens.append(int(sparse_token.item()))
            full = dense_adapter.forward(dense_token, state=dense_state)
            selected = generator.forward(sparse_token, state=sparse_state)
            dense_state, sparse_state = full.state, selected.state
            dense_token, sparse_token = full.logits[:, -1].argmax(-1, keepdim=True), selected.logits[:, -1].argmax(-1, keepdim=True)
        checks.append({"name": f"{example['id']}/greedy-record", "passed": True,
                       "dense_tokens": dense_tokens, "query_summary_tokens": sparse_tokens,
                       "dense_text": tokenizer.decode(dense_tokens), "query_summary_text": tokenizer.decode(sparse_tokens),
                       "note": "Recorded behavior, not a correctness assertion about generated text"})
        checks.append({"name": f"{example['id']}/prefix-cache-events", "passed": True, "events": cache.events})
    aggregates = []
    for name in policy_names:
        samples = [x for x in rows if x["policy"]["name"] == name]
        aggregates.append({"policy": name, "mean_next_token_KL": sum(x["mean_next_token_KL"] for x in samples) / len(samples),
            "mean_nll_delta": sum(x["mean_nll_delta"] for x in samples) / len(samples),
            "mean_final_active_kv_bytes": sum(x["raw_steps"][-1]["metrics"]["kv_bytes"] for x in samples) / len(samples),
            "mean_decode_step_ms": sum(s["metrics"]["elapsed_ms"] for x in samples for s in x["raw_steps"]) / sum(len(x["raw_steps"]) for x in samples),
            "mean_selection_ms": sum(s["metrics"]["selection_ms"] for x in samples for s in x["raw_steps"]) / sum(len(x["raw_steps"]) for x in samples)})
    return {"experiment": "pretrained-phase2-factorial-smoke", "passed": all(x["passed"] for x in checks),
            "configuration": config, "workload": workload, "workload_sha256": digest(workload),
            "identity": asdict(dense_adapter.identity), "checks": checks, "aggregate": aggregates, "trials": rows,
            "limitations": ["Three authored short-text examples and two timing repeats; no official benchmark or significance claim.",
                "CPU FP32 eager reference, not optimized dense kernels; timing is diagnostic, not speedup evidence.",
                "Sparse decode batch=1; one common page subset across KV groups in each layer. No per-group adaptive allocation.",
                "Prefix snapshots store independent full prefixes, not a shared radix trie. Restore hashes and copies are timed.",
                "Physical omission is permanent within a request; no tiered archive, fetch, recomputation, trained scorer, or runtime bound yet.",
                "Active KV bytes exclude CPU prefix copies, source recipes, features, model weights, MLP/logit activations and allocator overhead."]}
