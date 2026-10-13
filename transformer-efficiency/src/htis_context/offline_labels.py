"""Offline intervention labels; never imported by runtime feature/policy modules.

Counterfactual actions use the existing adapter/policy interface. Dense-current
attention is computed ONLY here for local measurement, not sparse selection.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from .runtime_features import extract


class CapturePolicy:
    name = "full"
    max_pages = None
    def __init__(self, page_size: int, layer_count: int):
        self.page_size, self.layer_count = page_size, layer_count
        self.calls = []

    def select(self, slots, history, features, query, *, now):
        index = len(self.calls)
        batch = extract(slots, history, features, query, layer=index,
                        layer_count=self.layer_count, now=now, page_size=self.page_size)
        self.calls.append({"features": batch, "query": query.detach().clone(), "history": history.detach().clone()})
        return torch.arange(len(slots), device=slots.device), torch.arange(len(features.ids), device=slots.device)


class InterventionPolicy:
    name = "offline-intervention"
    def __init__(self, page_size: int, layer: int, omitted_pages: tuple[int, ...], total_pages: int):
        self.page_size, self.layer = page_size, layer
        self.omitted_pages, self.max_pages = omitted_pages, total_pages
        self.index = 0

    def select(self, slots, history, features, query, *, now):
        layer = self.index
        self.index += 1
        if layer == self.layer:
            retain = torch.tensor([i for i, page in enumerate(features.ids.tolist()) if page not in self.omitted_pages], device=slots.device)
            indices = torch.nonzero(torch.isin(slots // self.page_size, features.ids[retain])).flatten()
        else:
            indices, retain = torch.arange(len(slots), device=slots.device), torch.arange(len(features.ids), device=slots.device)
        if len(indices) == 0:
            raise ValueError("Counterfactual cannot omit every visible key")
        return indices, retain


@dataclass(frozen=True)
class OfflineLabel:
    omitted_pages: tuple[int, ...]
    local_output_l2: float
    next_token_kl: float
    horizon_kl: float
    action_wall_ms: float


def kl(dense_logits: torch.Tensor, intervention_logits: torch.Tensor) -> float:
    log_p = dense_logits.double().log_softmax(-1)
    log_q = intervention_logits.double().log_softmax(-1)
    return float((log_p.exp() * (log_p - log_q)).sum(-1).mean().clamp(min=0))


@torch.no_grad()
def local_damage(query, layer, omitted_pages, page_size):
    """Fixed-state attention perturbation, averaged L2 over query heads."""
    b, hq, q, d = query.shape
    hkv = layer.keys.shape[1]
    if b != 1 or q != 1:
        raise ValueError("Local label supports one current query")
    grouped = query.reshape(b, hkv, hq // hkv, d)
    def output(indices):
        logits = torch.matmul(grouped, layer.keys[:, :, indices].transpose(-2, -1)) / d ** 0.5
        weights = logits.float().softmax(-1).to(query.dtype)
        return torch.matmul(weights, layer.values[:, :, indices])
    full = torch.arange(len(layer.slots), device=query.device)
    retained = torch.tensor([i for i, slot in enumerate(layer.slots.tolist()) if slot // page_size not in omitted_pages], device=query.device)
    if len(retained) == 0:
        raise ValueError("Local label must retain keys")
    return float((output(full) - output(retained)).norm(dim=-1).mean())
