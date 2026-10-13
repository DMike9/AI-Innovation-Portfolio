"""Deterministic synthetic attention workloads, NOT pretrained language models."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SyntheticStream:
    queries: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    targets: np.ndarray
    scenario: str
    seed: int


def make_stream(steps: int = 160, head_dim: int = 32, seed: int = 7,
                scenario: str = "mixed") -> SyntheticStream:
    """Create answer-seeking queries with known target positions.

    'persistent': repeatedly references early positions (heavy hitters).
    'recency': predominantly references recent tokens.
    'late_needle': an early token is not queried until later (eviction failure).
    'mixed': a blend of the above signals and sporadic long-range targets.
    Queries are generated with knowledge of their targets, but cache policies
    ONLY observe each query when its step arrives; no future information leaks.
    """
    if steps < 2 or head_dim < 4:
        raise ValueError("steps>=2 and head_dim>=4 are required")
    if scenario not in {"recency", "persistent", "late_needle", "mixed"}:
        raise ValueError(f"Unknown scenario: {scenario}")
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((steps, head_dim))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    values = rng.standard_normal((steps, head_dim))
    targets = np.zeros(steps, dtype=np.int64)
    for t in range(1, steps):
        recent = max(0, t - int(rng.integers(1, min(t, 5) + 1)))
        hot = [i for i in (3, 8, 13) if i < t]
        if scenario == "recency":
            targets[t] = recent
        elif scenario == "persistent":
            targets[t] = hot[t % len(hot)] if hot else recent
        elif scenario == "late_needle":
            targets[t] = 7 if t >= max(30, steps // 2) else recent
        else:
            u = rng.random()
            if u < 0.45 or not hot:
                targets[t] = recent
            elif u < 0.85:
                targets[t] = hot[t % len(hot)]
            elif u < 0.95:
                targets[t] = 0
            else:
                targets[t] = int(rng.integers(0, t))
    queries = (8.0 * math.sqrt(head_dim) * keys[targets]
               + rng.normal(0, 0.1, (steps, head_dim)))
    return SyntheticStream(queries, keys, values, targets, scenario, seed)
