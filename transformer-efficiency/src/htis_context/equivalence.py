"""Numerical acceptance checks against untouched upstream pretrained inference."""
from __future__ import annotations

from dataclasses import asdict, replace
import platform
from pathlib import Path
import tempfile

import torch
import transformers

from .state import PrefixSnapshot


@torch.inference_mode()
def check_equivalence(model, tokenizer, adapter, config: dict) -> dict:
    checks = []
    atol, rtol = config["atol"], config["rtol"]

    def compare(name, actual, expected):
        delta = (actual - expected).abs()
        passed = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
        checks.append({"name": name, "passed": passed,
                       "max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean()),
                       "shape": list(actual.shape)})

    def assertion(name, passed, **extra):
        checks.append({"name": name, "passed": bool(passed), **extra})

    device = next(model.parameters()).device
    texts = ["The capital of France is Paris. The capital of Italy is",
             "A reusable cache stores states computed from the exact preceding context.",
             "def add(a, b):\n    return a + b\n\nadd(2, 3)",
             "The library closes at six. On Tuesday, it closes at eight. Tuesday closing time is"]
    for example, text in enumerate(texts):
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        positions = torch.arange(ids.shape[1], device=device)[None]
        upstream = model(ids, position_ids=positions, use_cache=False).logits
        dense = adapter.forward(ids, position_ids=positions)
        compare(f"example-{example}/uncached-all-logits", dense.logits, upstream)
        cut = ids.shape[1] - 4
        cached = adapter.forward(ids[:, :cut])
        upstream_cached = model(ids[:, :cut], use_cache=True)
        for step in range(cut, ids.shape[1]):
            token = ids[:, step:step + 1]
            pos = positions[:, step:step + 1]
            cached = adapter.forward(token, state=cached.state, position_ids=pos)
            upstream_cached = model(token, past_key_values=upstream_cached.past_key_values,
                                    position_ids=pos, use_cache=True)
            compare(f"example-{example}/cached-upstream-step-{step}", cached.logits, upstream_cached.logits)
            compare(f"example-{example}/cached-uncached-step-{step}", cached.logits[:, -1], upstream[:, step])
            assertion(f"example-{example}/append-{step}", cached.state.recipe.length == step + 1
                      and all(x.keys.shape[2] == step + 1 for x in cached.state.layers))
        # Multi-token cache append tests causal alignment, independently of one-step decode.
        block = adapter.forward(ids[:, cut:], state=adapter.forward(ids[:, :cut]).state)
        compare(f"example-{example}/multi-token-append", block.logits, upstream[:, cut:])
        # Upstream model is never monkey patched or replaced by the adapter.
        assertion(f"example-{example}/greedy-continuation",
                  torch.equal(cached.logits[:, -1].argmax(-1), upstream[:, -1].argmax(-1)))
    ids = tokenizer(texts[0], return_tensors="pt").input_ids.to(device)
    for offset in (0, 37, 511, model.config.max_position_embeddings - ids.shape[1]):
        positions = torch.arange(ids.shape[1], device=device)[None] + offset
        compare(f"original-RoPE-positions-offset-{offset}", adapter.forward(ids, position_ids=positions).logits,
                model(ids, position_ids=positions, use_cache=False).logits)
        cut = ids.shape[1] // 2
        head = adapter.forward(ids[:, :cut], position_ids=positions[:, :cut])
        tail = adapter.forward(ids[:, cut:], state=head.state, position_ids=positions[:, cut:])
        compare(f"shifted-RoPE-cached-offset-{offset}", tail.logits,
                model(ids, position_ids=positions, use_cache=False).logits[:, cut:])
    # Mixed padding patterns, explicit original positions, both sides and interior keys.
    short = ids[:, :8]
    for side in ("left", "right", "interior"):
        tokens = torch.cat((short, short), dim=0)
        mask = torch.ones_like(tokens)
        mask[0, {"left": slice(0, 2), "right": slice(6, 8), "interior": slice(2, 4)}[side]] = 0
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        actual = adapter.forward(tokens, position_ids=pos, attention_mask=mask)
        expected = model(tokens, position_ids=pos, attention_mask=mask, use_cache=False).logits
        compare(f"{side}-padding-visible-logits", actual.logits[mask.bool()], expected[mask.bool()])
        new = ids[:, 8:9].expand(2, -1)
        new_mask = torch.cat((mask, torch.ones((2, 1), dtype=torch.long, device=device)), dim=1)
        new_pos = mask.sum(-1, keepdim=True)
        appended = adapter.forward(new, state=actual.state, attention_mask=new_mask, position_ids=new_pos)
        combined_pos = torch.cat((pos, new_pos), dim=1)
        compare(f"{side}-padding-cached-tail", appended.logits[:, -1],
                model(torch.cat((tokens, new), dim=1), position_ids=combined_pos,
                      attention_mask=new_mask, use_cache=False).logits[:, -1])
        compare(f"{side}-batch-isolation", actual.logits[1], adapter.forward(short, position_ids=pos[1:]).logits[0])
    # Padding invariance at visible positions versus the unpadded source.
    padded = torch.cat((torch.zeros((1, 2), dtype=torch.long, device=device), short), dim=1)
    pad_mask = torch.cat((torch.zeros((1, 2), dtype=torch.long, device=device), torch.ones_like(short)), dim=1)
    pad_pos = (pad_mask.cumsum(-1) - 1).clamp(min=0)
    compare("left-padding-versus-unpadded", adapter.forward(padded, attention_mask=pad_mask,
            position_ids=pad_pos).logits[:, 2:], adapter.forward(short).logits)
    prefix = adapter.forward(ids[:, :8]).state
    snapshot = PrefixSnapshot(prefix)
    restored_a = snapshot.restore(adapter.identity, prefix.recipe)
    restored_b = snapshot.restore(adapter.identity, prefix.recipe)
    before = restored_b.layers[0].keys.clone()
    restored_a.layers[0].keys.zero_()
    assertion("request-isolation", torch.equal(before, restored_b.layers[0].keys)
              and torch.equal(before, snapshot.restore(adapter.identity, prefix.recipe).layers[0].keys))
    a = adapter.forward(ids[:, 8:], state=snapshot.restore(adapter.identity, prefix.recipe))
    compare("snapshot-restore-continuation", a.logits, adapter.forward(ids).logits[:, 8:])
    with tempfile.TemporaryDirectory(prefix="htis-exact-prefix-") as directory:
        path = Path(directory) / "snapshot"
        snapshot.save(path)
        loaded = PrefixSnapshot.load(path, adapter.identity, prefix.recipe)
        restored = loaded.restore(adapter.identity, prefix.recipe)
        reloaded = adapter.forward(ids[:, 8:], state=restored)
        compare("safe-snapshot-restart-reload", reloaded.logits, adapter.forward(ids).logits[:, 8:])
    assertion("physical-GQA-layout", all(x.keys.shape[1] == config["kv_heads"]
                                          and x.values.shape[1] == config["kv_heads"] for x in prefix.layers))
    for field, value in (("revision", "0" * 40), ("access_scope", "other-tenant"),
                         ("rope_hash", "different"), ("adapters", ("changed-adapter",)),
                         ("dtype", "torch.float16"), ("implementation", "different-lineage")):
        try:
            snapshot.restore(replace(adapter.identity, **{field: value}), prefix.recipe)
        except ValueError:
            passed = True
        else:
            passed = False
        assertion(f"reject-incompatible-{field}", passed)
    # Eight greedy steps versus full upstream recomputation and upstream generate.
    generated = ids.clone()
    state = None
    next_input = ids
    for step in range(8):
        result = adapter.forward(next_input, state=state)
        full = model(generated, use_cache=False).logits[:, -1]
        compare(f"greedy-multistep-logits-{step}", result.logits[:, -1], full)
        next_input = result.logits[:, -1].argmax(-1, keepdim=True)
        generated = torch.cat((generated, next_input), dim=1)
        state = result.state
    upstream_generation = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=8,
                                           do_sample=False, eos_token_id=None, pad_token_id=0)
    assertion("greedy-versus-upstream-generate", torch.equal(generated, upstream_generation),
              token_ids=generated[0].tolist(), text=tokenizer.decode(generated[0]))
    return {"experiment": "pretrained-dense-adapter-equivalence", "passed": all(x["passed"] for x in checks),
            "identity": asdict(adapter.identity), "configuration": config, "checks": checks,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "torch": torch.__version__, "transformers": transformers.__version__,
                            "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads()},
            "limitation": "CPU FP32 smoke validation on short public-authored prompts; no speed or benchmark-quality claim."}
