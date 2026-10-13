"""Runtime-only page features. This module has no offline-label dependency.

Inputs are current query, retained insertion-cached summaries, *previous*
observed attention, original slots, layer index and current time. No labels,
future tokens, dense-current-attention weights or benchmark answers are accepted.
Locality is a page-level adaptation: convolution windows follow original page
adjacency, with explicit missing-neighbor masks rather than compact adjacency.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import torch

from .pages import PageFeatures


SCHEMA = "htis-runtime-page-features-v1"
SLICES = {"attention": (0, 2), "content": (2, 14), "value": (14, 17),
          "query": (17, 24), "locality": (24, 39)}
WIDTH = 39


@dataclass(frozen=True)
class RuntimePageBatch:
    page_ids: tuple[int, ...]
    values: torch.Tensor
    layer: int
    now: int
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA or self.values.shape != (len(self.page_ids), WIDTH):
            raise ValueError("Runtime feature schema/shape mismatch")
        if not torch.isfinite(self.values).all():
            raise ValueError("Runtime features must be finite")


def projection(d: int, device, dtype):
    # Fixed analytic projection, neither pretrained weights nor learned scores.
    row = torch.arange(1, d + 1, dtype=dtype, device=device)[:, None]
    col = torch.arange(1, 5, dtype=dtype, device=device)[None]
    return torch.sin(row * col) / math.sqrt(d)


@torch.no_grad()
def extract(slots: torch.Tensor, history: torch.Tensor, features: PageFeatures,
            query: torch.Tensor, *, layer: int, layer_count: int, now: int, page_size: int) -> RuntimePageBatch:
    if query.shape[0] != 1 or query.shape[2] != 1:
        raise ValueError("Runtime learned scorer currently needs a single request/query")
    hkv, p, d = features.key_min.shape
    if history.shape != (hkv, len(slots)):
        raise ValueError("Observed-history layout differs from actual KV groups")
    mid = (features.key_min + features.key_max) * 0.5
    proj = projection(d, query.device, query.dtype)
    key_proj = (mid @ proj).mean(0)
    value_proj = (features.value_center @ proj).mean(0)
    counts = torch.stack([(slots // page_size == i).sum() for i in features.ids])
    mass = torch.stack([history[:, slots // page_size == i].sum() for i in features.ids])
    attention = torch.stack([torch.log1p(mass.clamp(min=0)), (mass > 0).float()], -1)
    content = torch.cat([key_proj, value_proj,
                        torch.log1p(mid.norm(dim=-1).mean(0))[:, None],
                        torch.log1p((features.key_max - features.key_min).abs().mean((0, 2)))[:, None],
                        torch.log1p(features.value_center.std(dim=0, unbiased=False).norm(dim=-1))[:, None],
                        torch.log1p(counts.float())[:, None]], -1)
    value = torch.stack([torch.log1p(features.value_center.norm(dim=-1).mean(0)),
                         torch.log1p(features.value_radius.mean(0)),
                         torch.log1p(features.value_center.norm(dim=-1).amax(0))], -1)
    grouped = query[:, :, -1].reshape(1, hkv, query.shape[1] // hkv, 1, d)
    bounds = torch.where(grouped >= 0, features.key_max[None, :, None], features.key_min[None, :, None])
    upper = (grouped * bounds).sum(-1).amax((0, 1, 2)) / math.sqrt(d)
    mean_q = grouped.mean((0, 2, 3))
    dot = (mean_q[:, None] * mid).sum(-1).mean(0) / math.sqrt(d)
    age = ((now - features.ids * page_size).float() / max(now + 1, 1)).clamp(0, 1)
    qproject = (query[0, :, -1].mean(0) @ proj).expand(p, -1)
    condition = torch.cat([upper[:, None], dot[:, None], age[:, None],
                          torch.full((p, 1), layer / max(layer_count - 1, 1), device=query.device), qproject[:, :3]], -1)
    # Each page's 3-position window stores 4 value projections + availability.
    # Original page IDs guarantee that compacted gaps are not treated as neighbors.
    lookup = {int(i): idx for idx, i in enumerate(features.ids.tolist())}
    windows = torch.zeros((p, 3, 5), device=query.device, dtype=query.dtype)
    for index, page in enumerate(features.ids.tolist()):
        for offset in (-1, 0, 1):
            neighbor = lookup.get(page + offset)
            if neighbor is not None:
                windows[index, offset + 1, :4] = value_proj[neighbor]
                windows[index, offset + 1, 4] = 1
    values = torch.cat([attention, content, value, condition, windows.flatten(1)], -1).float()
    return RuntimePageBatch(tuple(features.ids.tolist()), values, layer, now)


def protected_pages(slots, *, now: int, page_size: int, prompt_tokens: int, sink_tokens: int, recent_tokens: int):
    mask = (slots < max(prompt_tokens, sink_tokens)) | (slots > now - recent_tokens)
    return set((slots[mask] // page_size).tolist())
