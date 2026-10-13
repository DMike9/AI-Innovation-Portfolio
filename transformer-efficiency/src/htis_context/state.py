"""Request-owned KV tensors and copy-isolated, integrity-checked snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .recovery import CPUArchive

import torch

from .provenance import ExecutionIdentity, SourceRecipe
from .pages import PageFeatures


@dataclass
class LayerKV:
    keys: torch.Tensor
    values: torch.Tensor
    slots: torch.Tensor
    history: torch.Tensor
    features: PageFeatures | None = None

    def clone(self) -> LayerKV:
        return LayerKV(*(x.detach().clone() for x in
                         (self.keys, self.values, self.slots, self.history)),
                       self.features.clone() if self.features is not None else None)

    @property
    def kv_bytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in (self.keys, self.values))

    @property
    def metadata_bytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in (self.slots, self.history)) + (
            self.features.bytes if self.features is not None else 0)


@dataclass
class RequestState:
    identity: ExecutionIdentity
    recipe: SourceRecipe
    layers: list[LayerKV]
    lineage: str = "dense"
    page_size: int = 8
    archive: CPUArchive | None = None
    memory_chain: str | None = None

    def fork(self) -> RequestState:
        return RequestState(self.identity, self.recipe,
                            [layer.clone() for layer in self.layers], self.lineage, self.page_size,
                            self.archive.fork() if self.archive is not None else None, self.memory_chain)

    @property
    def kv_bytes(self) -> int:
        return sum(x.kv_bytes for x in self.layers)


def _fingerprint(state: RequestState) -> str:
    hasher = hashlib.sha256()
    hasher.update((state.identity.key + state.recipe.key + state.lineage + str(state.page_size)).encode())
    if state.memory_chain is not None:
        hasher.update(state.memory_chain.encode())
    for layer in state.layers:
        features = layer.features.tensors() if layer.features is not None else ()
        for tensor in (layer.keys, layer.values, layer.slots, layer.history, *features):
            hasher.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            hasher.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


class PrefixSnapshot:
    """Owns private copies. Restores always clone; mutation is detected before use.

    Python/Torch tensors cannot be made read-only. The public API never exposes
    owned tensors, and an integrity hash catches accidental private mutation.
    This is a local trusted-process correctness guard, not an adversarial vault.
    """
    def __init__(self, state: RequestState):
        if state.lineage != "dense":
            raise ValueError("Approximate trajectories cannot enter exact prefix namespace")
        if not state.layers or any(not torch.equal(x.slots, torch.arange(state.recipe.length, device=x.slots.device))
                                   for x in state.layers):
            raise ValueError("Exact snapshots require the complete contiguous prefix in every layer")
        self._state = state.fork()
        # Complete dense prefixes have no request-specific memory decisions.
        # Do not export routing state into the exact across-request namespace.
        self._state.archive = None
        self._state.memory_chain = None
        self._fingerprint = _fingerprint(self._state)

    @property
    def identity(self) -> ExecutionIdentity:
        return self._state.identity

    @property
    def recipe(self) -> SourceRecipe:
        return self._state.recipe

    @property
    def kv_bytes(self) -> int:
        return self._state.kv_bytes

    def restore(self, identity: ExecutionIdentity, recipe: SourceRecipe) -> RequestState:
        if identity != self.identity or recipe != self.recipe:
            raise ValueError("Incompatible identity, access scope, ancestry, positions or masks")
        if _fingerprint(self._state) != self._fingerprint:
            raise ValueError("Snapshot integrity check failed")
        return self._state.fork()

    def save(self, directory: Path) -> None:
        """Safe tensor persistence. Never overwrite an existing snapshot directory."""
        from safetensors.torch import save_file
        state = self.restore(self.identity, self.recipe)
        directory.mkdir(parents=True, exist_ok=False)
        tensors = {}
        for i, layer in enumerate(state.layers):
            for name in ("keys", "values", "slots", "history"):
                tensors[f"{i}.{name}"] = getattr(layer, name).cpu().contiguous()
            if layer.features is not None:
                for name in ("ids", "key_min", "key_max", "value_center", "value_radius"):
                    tensors[f"{i}.features.{name}"] = getattr(layer.features, name).cpu().contiguous()
        save_file(tensors, str(directory / "kv.safetensors"))
        record = {"format": "htis-context-exact-prefix-v1", "identity": asdict(state.identity),
                  "recipe": asdict(state.recipe), "identity_key": state.identity.key,
                  "recipe_key": state.recipe.key, "layers": len(state.layers),
                  "page_size": state.page_size, "fingerprint": self._fingerprint}
        (directory / "provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, identity: ExecutionIdentity, recipe: SourceRecipe) -> PrefixSnapshot:
        """Validate scope/provenance before reading KV; no pickle or remote code."""
        from safetensors.torch import load_file
        record = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        if record["format"] != "htis-context-exact-prefix-v1":
            raise ValueError("Unsupported snapshot format")
        if record["identity_key"] != identity.key or record["recipe_key"] != recipe.key:
            raise ValueError("Incompatible snapshot identity/access scope or original prefix")
        from .provenance import digest
        if digest(record["identity"]) != identity.key or digest(record["recipe"]) != recipe.key:
            raise ValueError("Snapshot provenance integrity mismatch")
        tensors = load_file(str(directory / "kv.safetensors"), device="cpu")
        layers = []
        for i in range(record["layers"]):
            features = None
            if f"{i}.features.ids" in tensors:
                features = PageFeatures(*(tensors[f"{i}.features.{name}"] for name in
                                          ("ids", "key_min", "key_max", "value_center", "value_radius")))
            layers.append(LayerKV(*(tensors[f"{i}.{name}"] for name in ("keys", "values", "slots", "history")), features))
        state = RequestState(identity, recipe, layers, "dense", record["page_size"])
        if _fingerprint(state) != record["fingerprint"]:
            raise ValueError("Snapshot tensor integrity mismatch")
        return cls(state)
