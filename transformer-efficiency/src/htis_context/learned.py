"""Small trainable page-risk models and runtime policies; no label imports."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .provenance import digest
from .runtime_features import SCHEMA, WIDTH, extract, protected_pages


class RiskEstimator(nn.Module):
    def __init__(self, kind: str, hidden: int = 24):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.net = nn.Linear(WIDTH, 1)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(WIDTH, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        else:
            if kind not in {"htis", "no_attention", "no_locality", "no_content", "no_query", "no_value"}:
                raise ValueError("Unsupported learned model/ablation")
            self.branches = nn.ModuleDict()
            if kind != "no_attention":
                self.branches["attention"] = nn.Sequential(nn.Linear(2, 8), nn.SiLU())
            if kind != "no_content":
                self.branches["content"] = nn.Sequential(nn.Linear(15, 8), nn.SiLU())
            if kind != "no_locality":
                self.locality = nn.Conv1d(5, 8, kernel_size=3)
            count = len(self.branches) + (kind != "no_locality")
            self.gate = nn.Linear(7, count) if kind != "no_query" else nn.Parameter(torch.zeros(count))
            self.net = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, values):
        if values.shape[-1] != WIDTH:
            raise ValueError("Runtime feature schema width differs")
        if self.kind in ("linear", "mlp"):
            return self.net(values).squeeze(-1)
        x = values
        if self.kind == "no_value":
            x = x.clone()
            x[:, 6:10] = 0; x[:, 12] = 0; x[:, 14:17] = 0
            local = x[:, 24:39].reshape(-1, 3, 5)
            local[:, :, :4] = 0
        outputs = []
        for name, branch in self.branches.items():
            outputs.append(branch(x[:, :2] if name == "attention" else x[:, 2:17]))
        if self.kind != "no_locality":
            window = x[:, 24:39].reshape(-1, 3, 5).transpose(1, 2)
            outputs.append(torch.nn.functional.silu(self.locality(window).squeeze(-1)))
        weights = (self.gate.softmax(-1).expand(len(x), -1) if self.kind == "no_query"
                   else self.gate(x[:, 17:24]).softmax(-1))
        fused = (torch.stack(outputs, dim=1) * weights[:, :, None]).sum(dim=1)
        return self.net(fused).squeeze(-1)


class TrainedPagePolicy:
    """Use the established selection signature and physical cache; never labels."""
    name = "learned-page-risk"
    def __init__(self, estimator: RiskEstimator, mean: torch.Tensor, std: torch.Tensor, *,
                 page_size: int, max_pages: int, layer_count: int,
                 prompt_tokens: int, sink_tokens: int, recent_tokens: int, exploration_seed: int | None = None):
        self.estimator, self.mean, self.std = estimator.eval(), mean, std
        self.page_size, self.max_pages, self.layer_count = page_size, max_pages, layer_count
        self.prompt_tokens, self.sink_tokens, self.recent_tokens = prompt_tokens, sink_tokens, recent_tokens
        self.exploration_seed = exploration_seed
        self.layer_cursor = 0
        self.feature_bytes = 0
        self.exploration_events = []

    @torch.no_grad()
    def select(self, slots, history, features, query, *, now):
        layer = self.layer_cursor
        self.layer_cursor = (self.layer_cursor + 1) % self.layer_count
        if len(features.ids) <= self.max_pages:
            return torch.arange(len(slots), device=slots.device), torch.arange(len(features.ids), device=slots.device)
        protected = protected_pages(slots, now=now, page_size=self.page_size,
                    prompt_tokens=self.prompt_tokens, sink_tokens=self.sink_tokens, recent_tokens=self.recent_tokens)
        if len(protected) > self.max_pages:
            raise ValueError("Protected pages exceed physical page budget")
        batch = extract(slots, history, features, query, layer=layer,
                        layer_count=self.layer_count, now=now, page_size=self.page_size)
        self.feature_bytes = max(self.feature_bytes, batch.values.numel() * batch.values.element_size())
        scores = self.estimator((batch.values - self.mean.to(query.device)) / self.std.to(query.device)).tolist()
        candidates = [i for i, page in enumerate(batch.page_ids) if page not in protected]
        candidates.sort(key=lambda i: (-scores[i], -batch.page_ids[i]))
        free = self.max_pages - len(protected)
        selected = candidates[:free]
        # Budgeted exploration among *still available* nonpreferred pages.
        # No evicted KV is recovered: archive/fetch remains Phase 4.
        if self.exploration_seed is not None and free > 0 and len(candidates) > free:
            import random
            rng = random.Random(self.exploration_seed + now * self.layer_count + layer)
            sampled = rng.choice(candidates[free:])
            selected[-1] = sampled
            self.exploration_events.append({"layer": layer, "now": now, "page": batch.page_ids[sampled], "charged_pages": 1})
        chosen = protected | {batch.page_ids[i] for i in selected}
        page_indices = torch.tensor([i for i, p in enumerate(batch.page_ids) if p in chosen], device=slots.device)
        indices = torch.nonzero(torch.isin(slots // self.page_size, features.ids[page_indices])).flatten()
        return indices, page_indices


def load_estimator(directory: Path, identity):
    from safetensors.torch import load_file
    record = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if record["feature_schema"] != SCHEMA or record["feature_width"] != WIDTH:
        raise ValueError("Learned checkpoint feature schema mismatch")
    # Numerical model identity is separate from conservative whole-package
    # lookup namespace. Runtime feature and adapter-source hashes are checked.
    expected = asdict(identity)
    for field, value in record["execution_identity"].items():
        if field != "implementation" and digest(value) != digest(expected[field]):
            raise ValueError(f"Learned checkpoint model provenance mismatch: {field}")
    for name, sha in record["runtime_source_sha256"].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != sha:
            raise ValueError("Learned checkpoint runtime source changed")
    path = directory / "weights.safetensors"
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["weights_sha256"]:
        raise ValueError("Learned weights checksum mismatch")
    model = RiskEstimator(record["kind"], record["hidden_width"])
    tensors = load_file(str(path))
    mean, std = tensors.pop("feature_mean"), tensors.pop("feature_std")
    model.load_state_dict(tensors, strict=True)
    return model.eval(), mean, std, record
