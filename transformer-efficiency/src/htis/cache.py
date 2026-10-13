"""Online, causal, budgeted KV retention for a single attention head.

Research inspirations (NOT reimplementations): H2O (2306.14048), StreamingLLM
(2309.17453), Scissorhands (2305.17118), and Ada-KV (2407.11550).

The selector sees *only* past sparse attention and cache metadata, not future
queries or dense attention. Unlike multiplying a dense attention output by a
soft gate, deleting cached KV pairs reduces the simulated memory footprint.
This code does not implement model positional embeddings, kernels, or HF caches.
"""

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class CachePolicy(str, Enum):
    RECENT = "recent"
    SINK_RECENT = "sink_recent"
    HEAVY_HITTER = "heavy_hitter"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class TokenImportance:
    position: int
    observed_attention: float
    normalized_attention: float
    recency: float
    score: float
    protected: str | None


def attention(query: np.ndarray, keys: np.ndarray, values: np.ndarray):
    """Compute a single-query, single-head attention reference with stable softmax.

    Parameters: query (D,), keys and values (T,D). Returns (output (D,), weights (T,)).
    """
    query = np.asarray(query, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if query.ndim != 1 or keys.ndim != 2 or values.ndim != 2:
        raise ValueError("Expected query (D,), keys and values (T,D)")
    if keys.shape != values.shape or keys.shape[1] != query.shape[0] or not len(keys):
        raise ValueError("Incompatible or empty attention shapes")
    if not (np.all(np.isfinite(query)) and np.all(np.isfinite(keys)) and np.all(np.isfinite(values))):
        raise ValueError("Attention inputs must be finite")
    logits = keys @ query / math.sqrt(query.size)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights @ values, weights


class StreamingKVCache:
    """Bounded single-head KV cache with deterministic causal eviction.

    `step` inserts the current position, evicts until under budget *before*
    attention, computes sparse attention and then updates observed mass. It
    never computes full-history attention for its retention decisions.
    `importance` captures the pre-query ranking of currently retained tokens.

    Heavy hitter score is an *observed, exponentially decayed* attention proxy.
    It is not ground-truth causal importance, and sparse attention introduces
    selection bias (an evicted key cannot recover).
    """

    def __init__(self, budget: int, *, policy: CachePolicy | str = CachePolicy.HYBRID,
                 sink_tokens: int = 2, recent_tokens: int = 4,
                 attention_weight: float = 0.7, decay: float = 0.98,
                 recency_tau: float = 16.0):
        self.policy = CachePolicy(policy)
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("budget must be a positive integer")
        if sink_tokens < 0 or recent_tokens < 1 or sink_tokens + recent_tokens > budget:
            raise ValueError("Require sink>=0, recent>=1, sink+recent<=budget")
        if not 0 <= attention_weight <= 1 or not 0 <= decay <= 1 or recency_tau <= 0:
            raise ValueError("Require weight/decay in [0,1] and positive tau")
        self.budget = budget
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.attention_weight = float(attention_weight)
        self.decay = float(decay)
        self.recency_tau = float(recency_tau)
        self._keys: dict[int, np.ndarray] = {}
        self._values: dict[int, np.ndarray] = {}
        self._mass: dict[int, float] = {}
        self._next_position = 0
        self._d: int | None = None
        self.importance: list[TokenImportance] = []

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(sorted(self._keys))

    @property
    def stored_tokens(self) -> int:
        return len(self._keys)

    def bytes_used(self, bytes_per_scalar: int = 2) -> int:
        """Theoretical K+V storage; excludes Python, indices, metadata, GPU allocator."""
        if bytes_per_scalar < 1:
            raise ValueError("bytes_per_scalar must be >=1")
        return 2 * self.stored_tokens * (self._d or 0) * bytes_per_scalar

    def _protected(self, now: int) -> dict[int, str]:
        indices = sorted(self._keys)
        if self.policy is CachePolicy.RECENT:
            return {i: "recent" for i in indices[-self.budget:]}
        if self.policy is CachePolicy.SINK_RECENT:
            sinks = [i for i in indices if i < self.sink_tokens]
            keep_recent = self.budget - len(sinks)
            protected = {i: "sink" for i in sinks}
            if keep_recent:
                protected.update({i: "recent" for i in indices[-keep_recent:]})
            return protected
        if self.policy is CachePolicy.HEAVY_HITTER:
            return {now: "current"}
        protected = {i: "sink" for i in indices if i < self.sink_tokens}
        protected.update({i: "recent" for i in indices[-self.recent_tokens:]})
        return protected

    def _rank(self, now: int) -> list[TokenImportance]:
        protected = self._protected(now)
        peak_mass = max(self._mass.values(), default=0.0)
        ranking = []
        for idx in self.positions:
            mass = self._mass[idx]
            normalized = mass / peak_mass if peak_mass > 0 else 0.0
            recency = math.exp(-(now - idx) / self.recency_tau)
            if self.policy in (CachePolicy.RECENT, CachePolicy.SINK_RECENT):
                score = recency
            elif self.policy is CachePolicy.HEAVY_HITTER:
                score = normalized
            else:
                score = self.attention_weight * normalized + (1 - self.attention_weight) * recency
            ranking.append(TokenImportance(idx, mass, normalized, recency, score,
                                           protected.get(idx)))
        return ranking

    def step(self, query: np.ndarray, key: np.ndarray, value: np.ndarray):
        """Append a token and return (sparse_output, local_weights, retained_positions)."""
        query, key, value = [np.asarray(x, dtype=np.float64).copy() for x in (query, key, value)]
        if any(x.ndim != 1 or not np.all(np.isfinite(x)) for x in (query, key, value)):
            raise ValueError("query, key and value must be finite vectors")
        if not (query.size == key.size == value.size and query.size > 0):
            raise ValueError("query, key and value must have matching nonempty dimensions")
        if self._d is not None and query.size != self._d:
            raise ValueError("Attention head dimension cannot change during streaming")
        self._d = query.size
        now = self._next_position
        self._next_position += 1
        self._keys[now], self._values[now], self._mass[now] = key, value, 0.0
        ranking = self._rank(now)
        protected = {r.position for r in ranking if r.protected}
        if len(protected) > self.budget:
            raise AssertionError("protected positions exceed budget")
        free_slots = self.budget - len(protected)
        others = [r for r in ranking if r.position not in protected]
        others.sort(key=lambda r: (-r.score, -r.position))
        retain = protected | {r.position for r in others[:free_slots]}
        for idx in tuple(self._keys):
            if idx not in retain:
                del self._keys[idx], self._values[idx], self._mass[idx]
        self.importance = [r for r in ranking if r.position in retain]
        retained = self.positions
        keys = np.stack([self._keys[i] for i in retained])
        values = np.stack([self._values[i] for i in retained])
        output, weights = attention(query, keys, values)
        for idx, weight in zip(retained, weights):
            self._mass[idx] = self.decay * self._mass[idx] + float(weight)
        return output, weights, retained
