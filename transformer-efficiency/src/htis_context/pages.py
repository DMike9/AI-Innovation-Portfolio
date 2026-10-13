"""Insertion-cached page features and simplified, untrained causal baselines.

Physical pages cover original contiguous source slots; compact indices never
become RoPE positions or locality coordinates. No archive recovery in Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Protocol

import torch


@dataclass
class PageFeatures:
    ids: torch.Tensor
    key_min: torch.Tensor
    key_max: torch.Tensor
    value_center: torch.Tensor
    value_radius: torch.Tensor

    def clone(self) -> PageFeatures:
        return PageFeatures(*(x.detach().clone() for x in self.tensors()))

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.ids, self.key_min, self.key_max, self.value_center, self.value_radius

    @property
    def bytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in self.tensors())

    def subset(self, indices: torch.Tensor) -> PageFeatures:
        return PageFeatures(self.ids[indices], self.key_min[:, indices], self.key_max[:, indices],
                            self.value_center[:, indices], self.value_radius[:, indices])


def update_features(keys: torch.Tensor, values: torch.Tensor, slots: torch.Tensor,
                    page_size: int, previous: PageFeatures | None, new_slots: torch.Tensor) -> PageFeatures:
    """Only summarize newly inserted or extended pages; reuse all other features.

    Features aggregate over batch for dense accounting; sparse policies currently
    reject batches larger than one to avoid mixing per-request task selection.
    """
    ids = torch.unique(slots // page_size, sorted=True)
    touched = set((new_slots // page_size).tolist())
    old = {int(x): i for i, x in enumerate(previous.ids.tolist())} if previous is not None else {}
    minima, maxima, centers, radii = [], [], [], []
    for page_id in ids.tolist():
        if page_id not in touched and page_id in old:
            i = old[page_id]
            minima.append(previous.key_min[:, i])
            maxima.append(previous.key_max[:, i])
            centers.append(previous.value_center[:, i])
            radii.append(previous.value_radius[:, i])
        else:
            index = torch.nonzero(slots // page_size == page_id).flatten()
            k = keys[:, :, index].transpose(0, 1).flatten(1, 2)
            v = values[:, :, index].transpose(0, 1).flatten(1, 2)
            center = v.mean(dim=1)
            minima.append(k.amin(dim=1))
            maxima.append(k.amax(dim=1))
            centers.append(center)
            radii.append(torch.linalg.vector_norm(v - center[:, None], dim=-1).amax(dim=1))
    return PageFeatures(ids, torch.stack(minima, dim=1), torch.stack(maxima, dim=1),
                        torch.stack(centers, dim=1), torch.stack(radii, dim=1))


class PagePolicy(Protocol):
    page_size: int
    name: str
    max_pages: int | None
    def select(self, slots: torch.Tensor, history: torch.Tensor, features: PageFeatures,
               query: torch.Tensor, *, now: int) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class BaselinePolicy:
    name: str = "full"
    page_size: int = 8
    max_pages: int | None = None
    prompt_tokens: int = 0
    sink_tokens: int = 0
    recent_tokens: int = 1
    seed: int = 17

    def __post_init__(self):
        for value in (self.page_size, self.prompt_tokens, self.sink_tokens, self.recent_tokens, self.seed):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("Page sizes, protection counts and seeds must be integers")
        if self.max_pages is not None and (not isinstance(self.max_pages, int) or isinstance(self.max_pages, bool)):
            raise ValueError("Page count must be an integer")
        if self.name not in {"full", "recent", "sinks_recent", "prompt_random", "attention_history", "query_summary"}:
            raise ValueError("Unknown policy; learned policies are not implemented")
        if self.page_size < 1 or self.recent_tokens < 1 or min(self.prompt_tokens, self.sink_tokens) < 0:
            raise ValueError("Invalid page/protection configuration")
        if self.name != "full" and (self.max_pages is None or self.max_pages < 1):
            raise ValueError("Bounded policies require max_pages >= 1")

    def select(self, slots, history, features, query, *, now):
        if self.name == "full" or len(features.ids) <= self.max_pages:
            return torch.arange(len(slots), device=slots.device), torch.arange(len(features.ids), device=slots.device)
        protected_slots = (slots < max(self.prompt_tokens, self.sink_tokens)) | (slots > now - self.recent_tokens)
        protected = set((slots[protected_slots] // self.page_size).tolist())
        if len(protected) > self.max_pages:
            raise ValueError("Protected pages exceed page budget; cannot silently drop instructions/current tokens")
        ids = features.ids.tolist()
        if self.name in ("recent", "sinks_recent"):
            scores = [float(i) for i in ids]
        elif self.name == "prompt_random":
            # A per-step deterministic causal RNG, never future tokens or labels.
            rng = random.Random(self.seed + now)
            scores = [rng.random() for _ in ids]
        elif self.name == "attention_history":
            scores = [float(history[:, slots // self.page_size == i].sum()) for i in ids]
        else:
            # Query-conditioned coordinate bounds over insertion-cached keys.
            # Simplified Quest-inspired heuristic, not a faithful implementation,
            # learned importance, or the value-sensitive omission bound.
            b, hq, _, d = query.shape
            hkv = features.key_min.shape[0]
            grouped = query[:, :, -1].reshape(b, hkv, hq // hkv, 1, d)
            bounds = torch.where(grouped >= 0, features.key_max[None, :, None],
                                 features.key_min[None, :, None])
            scores = (grouped * bounds).sum(-1).amax(dim=(0, 1, 2)).div(math.sqrt(d)).tolist()
        candidates = [i for i in range(len(ids)) if ids[i] not in protected]
        candidates.sort(key=lambda i: (-scores[i], -ids[i]))
        chosen = protected | {ids[i] for i in candidates[:self.max_pages - len(protected)]}
        feature_indices = torch.tensor([i for i, page in enumerate(ids) if page in chosen], device=slots.device)
        token_indices = torch.nonzero(torch.isin(slots // self.page_size, features.ids[feature_indices])).flatten()
        return token_indices, feature_indices


@dataclass(frozen=True)
class MemoryBudget:
    """Explicit tensor-byte ceilings; Python/allocator overhead reported separately.

    KV allocation limit and estimated transient tensor working-set limit are
    separate. Neither is represented as measured peak GPU allocation on CPU.
    """
    max_kv_bytes: int
    max_working_tensor_bytes: int

    def __post_init__(self):
        if any(not isinstance(x, int) or isinstance(x, bool) for x in (self.max_kv_bytes, self.max_working_tensor_bytes)):
            raise ValueError("Byte budgets must be integers")
        if self.max_kv_bytes < 1 or self.max_working_tensor_bytes < self.max_kv_bytes:
            raise ValueError("Working tensor budget must cover the positive KV budget")

    def validate(self, kv_bytes: int, working_tensor_bytes: int):
        if kv_bytes > self.max_kv_bytes:
            raise MemoryError("Physical retained KV bytes exceed budget")
        if working_tensor_bytes > self.max_working_tensor_bytes:
            raise MemoryError("Estimated transient tensor working-set bytes exceed budget")
