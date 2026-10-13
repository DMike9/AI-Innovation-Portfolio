"""Bounded exact-prefix entries with access checks and deterministic LRU/LFU.

Whole prefixes are stored independently; no shared-block deduplication/trie yet.
Every entry includes its entire ancestry, so eviction cannot orphan a parent.
"""
from __future__ import annotations

from dataclasses import dataclass

from .provenance import ExecutionIdentity, SourceRecipe, digest
from .state import PrefixSnapshot, RequestState


@dataclass
class _Entry:
    snapshot: PrefixSnapshot
    touched: int
    hits: int
    size: int


class ExactPrefixCache:
    def __init__(self, max_bytes: int, *, eviction: str = "lru"):
        if max_bytes < 1 or eviction not in ("lru", "lfu"):
            raise ValueError("Require positive storage budget and LRU/LFU eviction")
        self.max_bytes = max_bytes
        self.eviction = eviction
        self._entries: dict[str, _Entry] = {}
        self._clock = 0
        self.events: list[dict] = []

    @property
    def bytes_used(self) -> int:
        return sum(x.size for x in self._entries.values())

    def put(self, state: RequestState) -> bool:
        snapshot = PrefixSnapshot(state)
        # These are real independent CPU tensor copies, charged separately.
        stored = snapshot.restore(state.identity, state.recipe)
        for layer in stored.layers:
            layer.keys, layer.values = layer.keys.cpu(), layer.values.cpu()
            layer.slots, layer.history = layer.slots.cpu(), layer.history.cpu()
            if layer.features is not None:
                from .pages import PageFeatures
                layer.features = PageFeatures(*(x.cpu() for x in layer.features.tensors()))
        snapshot = PrefixSnapshot(stored)
        size = stored.kv_bytes + sum(x.metadata_bytes for x in stored.layers)
        size += len(str(stored.recipe).encode()) + len(str(stored.identity).encode())
        if size > self.max_bytes:
            self.events.append({"event": "admission-rejected", "reason": "entry exceeds CPU tensor+recipe budget", "bytes": size})
            return False
        key = digest([state.identity.key, state.recipe.key])
        self._clock += 1
        self._entries.pop(key, None)
        while self.bytes_used + size > self.max_bytes:
            victim = min(self._entries, key=lambda k: ((self._entries[k].hits if self.eviction == "lfu" else 0),
                                                     self._entries[k].touched, k))
            old = self._entries.pop(victim)
            self.events.append({"event": "eviction", "key": victim, "bytes": old.size, "policy": self.eviction})
        self._entries[key] = _Entry(snapshot, self._clock, 0, size)
        self.events.append({"event": "admission", "key": key, "bytes": size,
                            "cpu_kv_copy_bytes": stored.kv_bytes, "prefix_tokens": state.recipe.length})
        return True

    def get(self, identity: ExecutionIdentity, recipe: SourceRecipe, *, device="cpu") -> RequestState | None:
        # Caller must supply full validated identity/scope; similarity objects are not accepted.
        if not isinstance(identity, ExecutionIdentity) or not isinstance(recipe, SourceRecipe):
            raise TypeError("Exact reuse needs an execution identity and ordered source recipe")
        key = digest([identity.key, recipe.key])
        entry = self._entries.get(key)
        if entry is None:
            self.events.append({"event": "miss", "key": key, "scope": identity.access_scope})
            return None
        state = entry.snapshot.restore(identity, recipe)
        self._clock += 1
        entry.touched = self._clock
        entry.hits += 1
        for layer in state.layers:
            layer.keys, layer.values = layer.keys.to(device), layer.values.to(device)
            layer.slots, layer.history = layer.slots.to(device), layer.history.to(device)
            if layer.features is not None:
                from .pages import PageFeatures
                layer.features = PageFeatures(*(x.to(device) for x in layer.features.tensors()))
        self.events.append({"event": "hit", "key": key, "scope": identity.access_scope,
                            "prefix_tokens": recipe.length, "restore_cpu_copy_bytes": state.kv_bytes,
                            "device_transfer_kv_bytes": state.kv_bytes if str(device) != "cpu" else 0})
        return state
