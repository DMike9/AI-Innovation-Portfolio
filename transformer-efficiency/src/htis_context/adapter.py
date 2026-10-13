"""Model-specific Llama reference forward, independent of upstream cache code.

Uses unmodified pretrained projections, RMSNorm, MLP, and original-position RoPE.
Grouped attention never stores repeated K/V per query head. No training here.
Only causal slot-order and 2-D binary key-padding masks are supported.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .provenance import ExecutionIdentity, SourceRecipe, digest
from .state import LayerKV, RequestState
from .pages import BaselinePolicy, MemoryBudget, PagePolicy, update_features


@dataclass
class ForwardResult:
    logits: torch.Tensor
    state: RequestState
    metrics: dict


class LlamaReferenceAdapter:
    def __init__(self, model, identity: ExecutionIdentity, *, policy: PagePolicy | None = None,
                 memory_budget: MemoryBudget | None = None):
        config = model.config
        if config.model_type != "llama":
            raise ValueError("Only the explicitly tested Llama architecture is supported")
        if config.rope_scaling is not None:
            raise ValueError("Scaled/dynamic RoPE requires independent equivalence validation")
        if config.pretraining_tp != 1 or config.hidden_act != "silu":
            raise ValueError("Unsupported tensor-parallel or activation configuration")
        if identity.config_hash != digest(config.to_dict()):
            raise ValueError("Model configuration does not match execution identity")
        if identity.dtype != str(next(model.parameters()).dtype):
            raise ValueError("Precision identity does not match weights")
        if next(model.parameters()).dtype != torch.float32:
            raise ValueError("Only FP32 is validated; other precisions require separate tolerance/equivalence tests")
        if identity.rope_hash != digest({"theta": config.rope_theta, "scaling": config.rope_scaling}):
            raise ValueError("RoPE configuration does not match execution identity")
        if identity.adapters or hasattr(model, "peft_config"):
            raise ValueError("Adapter-modified weights are not supported by this reference")
        if identity.mask_semantics != "causal-slot-order+binary-key-padding-v1" or identity.cache_format != "B,KV-head,original-slot,head-dim-v1":
            raise ValueError("Unsupported mask semantics or cache format")
        if any(layer.self_attn.head_dim != config.hidden_size // config.num_attention_heads for layer in model.model.layers):
            raise ValueError("Nonstandard head dimensions require independent validation")
        self.model = model.eval()
        self.identity = identity
        self.policy = policy or BaselinePolicy()
        self.memory_budget = memory_budget
        self._parameter_signature = self._weights_signature()

    def _weights_signature(self):
        # Cheap trusted-process mutation guard. Checkpoint content checksums are
        # validated at load; parameter versions detect subsequent in-place edits.
        if any(p.is_inference() for p in self.model.parameters()):
            raise ValueError("Model weights lack mutation counters; construct the model outside inference_mode")
        return tuple((name, id(p), p._version, str(p.dtype), str(p.device), tuple(p.shape))
                     for name, p in self.model.named_parameters())

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, *, state: RequestState | None = None,
                position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> ForwardResult:
        start = time.perf_counter()
        if self.model.training or digest(self.model.config.to_dict()) != self.identity.config_hash or self._weights_signature() != self._parameter_signature:
            raise ValueError("Model configuration or weights changed; cached computation identity is stale")
        if hasattr(self.model, "peft_config"):
            raise ValueError("Adapter-modified execution requires a separately validated identity")
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or 0 in input_ids.shape:
            raise ValueError("Expected nonempty [batch, tokens] int64 IDs")
        if input_ids.min() < 0 or input_ids.max() >= self.model.config.vocab_size:
            raise ValueError("Token IDs outside the model vocabulary")
        b, q_len = input_ids.shape
        if self.policy.name != "full" and (b != 1 or q_len != 1):
            raise ValueError("Sparse execution currently supports batch=1, one new token; use full policy for prefill")
        past = state.recipe.length if state is not None else 0
        if state is not None and (state.identity != self.identity or len(state.recipe.tokens) != b):
            raise ValueError("Request identity or batch does not match")
        if state is not None and state.page_size != self.policy.page_size:
            raise ValueError("Page size changed during request; recompute features explicitly")
        if state is not None:
            if len(state.layers) != len(self.model.model.layers):
                raise ValueError("Incomplete layer cache")
            for old in state.layers:
                expected = (b, self.model.config.num_key_value_heads, len(old.slots), self.model.config.hidden_size // self.model.config.num_attention_heads)
                if old.keys.shape != expected or old.values.shape != expected or old.keys.dtype != next(self.model.parameters()).dtype:
                    raise ValueError("Cache KV layout or precision does not match model")
                if old.slots.ndim != 1 or len(old.slots) == 0 or old.slots.min() < 0 or old.slots.max() >= state.recipe.length:
                    raise ValueError("Invalid original slot map")
                if len(old.slots) > 1 and not (old.slots[1:] > old.slots[:-1]).all():
                    raise ValueError("Cache slots must be unique and in original order")
                if state.lineage == "dense" and not torch.equal(old.slots, torch.arange(state.recipe.length, device=old.slots.device)):
                    raise ValueError("A compacted state cannot be labeled dense")
        total = past + q_len
        device = input_ids.device
        if state is not None and any(x.keys.device != device or x.values.device != device for x in state.layers):
            raise ValueError("Cache must be restored onto the request device")
        if attention_mask is None:
            old = state.recipe.mask if state is not None else tuple(() for _ in range(b))
            attention_mask = torch.tensor([list(row) + [1] * q_len for row in old], device=device)
        if attention_mask.shape != (b, total) or not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("Mask must be binary [batch, entire prefix plus new tokens]")
        if state is not None and attention_mask[:, :past].tolist() != [list(x) for x in state.recipe.mask]:
            raise ValueError("Earlier masks changed; recompute the correct prefix")
        if position_ids is None:
            position_ids = torch.arange(past, total, device=device).unsqueeze(0).expand(b, -1)
        if position_ids.shape != (b, q_len) or position_ids.dtype != torch.long:
            raise ValueError("Explicit positions must be [batch, new tokens] int64")
        if position_ids.min() < 0 or position_ids.max() >= self.model.config.max_position_embeddings:
            raise ValueError("Original positions exceed the validated context range")
        if total > self.model.config.max_position_embeddings:
            raise ValueError("Token count exceeds validated context range")
        def append(old, new):
            return tuple(tuple(x) + tuple(y) for x, y in zip(old, new.tolist()))
        empty = tuple(() for _ in range(b))
        recipe = SourceRecipe(append(state.recipe.tokens if state else empty, input_ids),
                              append(state.recipe.positions if state else empty, position_ids),
                              tuple(tuple(x) for x in attention_mask.tolist()))
        hidden = self.model.model.embed_tokens(input_ids)
        # PHASE4 EXTENSION BEGIN
        recovery = getattr(self.policy, "recovery", None)
        if state is not None and state.archive is not None and recovery is None:
            raise ValueError("Archived request requires its recovery controller")
        if recovery is not None:
            recovery.begin(state, self.identity, recipe)
        # PHASE4 EXTENSION END
        cos, sin = self.model.model.rotary_emb(hidden, position_ids)
        query_slots = torch.arange(past, total, device=device)
        layers = []
        lineage = state.lineage if state else "dense"
        metrics = {"attention_inputs": [], "elapsed_ms": 0.0, "selection_ms": 0.0,
                   "feature_update_ms": 0.0, "policy": self.policy.name, "compaction_copy_bytes": 0,
                   "peak_attention_workspace_tensor_bytes": 0,
                   "peak_cache_transition_tensor_bytes": 0}
        for index, layer in enumerate(self.model.model.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attn = layer.self_attn
            d = attn.head_dim
            hq, hkv = self.model.config.num_attention_heads, self.model.config.num_key_value_heads
            groups = hq // hkv
            query = attn.q_proj(normalized).view(b, q_len, hq, d).transpose(1, 2)
            key = attn.k_proj(normalized).view(b, q_len, hkv, d).transpose(1, 2)
            value = attn.v_proj(normalized).view(b, q_len, hkv, d).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            slots = query_slots
            history = torch.zeros((hkv, q_len), device=device)
            previous_features = None
            if state is not None:
                old = state.layers[index]
                key = torch.cat((old.keys, key), dim=2)
                value = torch.cat((old.values, value), dim=2)
                slots = torch.cat((old.slots, slots))
                history = torch.cat((old.history, history), dim=1)
                previous_features = old.features
                if previous_features is not None and not torch.equal(previous_features.ids, torch.unique(old.slots // self.policy.page_size)):
                    raise ValueError("Page size changed during request; recompute features explicitly")
            feature_start = time.perf_counter()
            features = update_features(key, value, slots, self.policy.page_size, previous_features, query_slots)
            metrics["feature_update_ms"] += (time.perf_counter() - feature_start) * 1000
            # PHASE4 EXTENSION BEGIN
            if recovery is not None:
                key, value, slots, history, features = recovery.route(
                    index, LayerKV(key, value, slots, history, features), query, total - 1)
                if len(slots) < total:
                    lineage = "approximate:physical-page-omission"
            # PHASE4 EXTENSION END
            selection_start = time.perf_counter()
            before_tokens = len(slots)
            indices, page_indices = self.policy.select(slots, history, features, query, now=total - 1)
            transition_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()
            if len(indices) < len(slots):
                key, value = key[:, :, indices].contiguous(), value[:, :, indices].contiguous()
                slots, history = slots[indices], history[:, indices]
                features = features.subset(page_indices)
                copied_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()
                transition_bytes += copied_bytes
                metrics["compaction_copy_bytes"] += copied_bytes
                lineage = "approximate:physical-page-omission"
            metrics["selection_ms"] += (time.perf_counter() - selection_start) * 1000
            metrics["peak_cache_transition_tensor_bytes"] = max(metrics["peak_cache_transition_tensor_bytes"], transition_bytes)
            if self.memory_budget is not None:
                scalar_bytes = key.element_size()
                per_token_bytes = 2 * b * hkv * d * scalar_bytes
                future_tokens = []
                for remaining in range(index + 1, len(self.model.model.layers)):
                    count = (len(state.layers[remaining].slots) if state else 0) + q_len
                    if self.policy.name != "full":
                        count = min(count, self.policy.max_pages * self.policy.page_size)
                    future_tokens.append(count)
                # Conservative capacity check before allocating attention scores.
                projected_kv = sum(x.kv_bytes for x in layers) + per_token_bytes * (len(slots) + sum(future_tokens))
                estimated_workspace = scalar_bytes * (3 * b * hq * q_len * len(slots)
                                                       + 3 * b * hq * q_len * d + b * q_len * len(slots))
                known_metadata = sum(x.metadata_bytes for x in layers) + features.bytes
                estimated_working = (projected_kv + (state.kv_bytes if state else 0)
                                     + known_metadata + transition_bytes + estimated_workspace)
                self.memory_budget.validate(projected_kv, estimated_working)
            # Fold query groups into rows rather than broadcasting KV along a
            # group batch dimension (matmul can materialize expanded KV there).
            grouped_q = query.reshape(b, hkv, groups * q_len, d)
            logits = (torch.matmul(grouped_q, key.transpose(-2, -1)) * attn.scaling
                      ).reshape(b, hkv, groups, q_len, len(slots))
            visible = ((slots[None, :] <= query_slots[:, None])[None, None, None]
                       & attention_mask[:, slots].bool()[:, None, None, None, :])
            causal_mask = torch.zeros(visible.shape, dtype=query.dtype, device=device)
            causal_mask.masked_fill_(~visible, torch.finfo(query.dtype).min)
            weights = torch.softmax(logits + causal_mask, dim=-1, dtype=torch.float32).to(query.dtype)
            out = torch.matmul(weights.reshape(b, hkv, groups * q_len, len(slots)), value)
            # PHASE4 EXTENSION BEGIN
            if recovery is not None:
                key, value, slots, history, features, weights, out, logits = recovery.attention(
                    index, LayerKV(key, value, slots, history, features), query,
                    weights, out, logits, attn.scaling)
            # PHASE4 EXTENSION END
            out = out.reshape(b, hq, q_len, d).transpose(1, 2).contiguous().reshape(b, q_len, -1)
            hidden = residual + attn.o_proj(out)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            history = history + weights.float().mean(dim=(0, 2, 3))
            layers.append(LayerKV(key, value, slots, history, features))
            workspace = sum(x.numel() * x.element_size() for x in (logits, weights, causal_mask, query, grouped_q, out)) + logits.numel() * logits.element_size()
            metrics["peak_attention_workspace_tensor_bytes"] = max(metrics["peak_attention_workspace_tensor_bytes"], workspace)
            metrics["attention_inputs"].append({"layer": index, "query_tokens": q_len,
                                                "key_tokens": len(slots), "kv_heads": hkv,
                                                "query_heads": hq, "before_selection_tokens": before_tokens,
                                                "original_slots": slots.tolist(), "page_ids": features.ids.tolist()})
        logits = self.model.lm_head(self.model.model.norm(hidden))
        result_state = RequestState(self.identity, recipe, layers, lineage, self.policy.page_size)
        # PHASE4 EXTENSION BEGIN
        if recovery is not None:
            result_state.archive = recovery.finish(result_state)
            metrics["recovery"] = recovery.report()
        # PHASE4 EXTENSION END
        metrics.update(kv_bytes=result_state.kv_bytes,
                       metadata_tensor_bytes=sum(x.metadata_bytes for x in layers),
                       elapsed_ms=(time.perf_counter() - start) * 1000)
        metrics["input_state_kv_bytes"] = state.kv_bytes if state else 0
        metrics["source_recipe_json_bytes"] = len(str(recipe).encode())
        metrics["estimated_cache_and_attention_working_tensor_bytes"] = (
            result_state.kv_bytes + metrics["input_state_kv_bytes"] + metrics["metadata_tensor_bytes"]
            + metrics["peak_cache_transition_tensor_bytes"] + metrics["peak_attention_workspace_tensor_bytes"])
        metrics["accounting_notes"] = "Tensor estimates, not allocator/RSS peaks. Excludes model weights, MLP/logit activations, Python overhead and untracked kernel scratch. CPU copies are separate."
        # PHASE4 EXTENSION BEGIN
        if recovery is not None:
            recovery.validate_working(metrics, result_state)
        # PHASE4 EXTENSION END
        if self.memory_budget is not None:
            self.memory_budget.validate(result_state.kv_bytes, metrics["estimated_cache_and_attention_working_tensor_bytes"])
        return ForwardResult(logits, result_state, metrics)
