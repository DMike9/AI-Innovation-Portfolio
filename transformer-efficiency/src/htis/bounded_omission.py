"""Value-sensitive upper bound for omitting fixed single-head KV pages.

This is a NumPy/float64 research reference, not an inference kernel. The bound
holds in real arithmetic for the supplied Q/K/V state. Floating-point tests are
not an outward-rounded numerical certificate or an end-to-end LLM guarantee.
See HTIS_CONTEXT_ARCHITECTURE.md for the proof, assumptions, and prior art.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _array(value: ArrayLike, ndim: int, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or 0 in result.shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty, finite {ndim}-D array")
    return result


@dataclass(frozen=True)
class PageSummary:
    """Summaries must cover every omitted, causally visible KV exactly once."""

    count: int
    key_min: FloatArray
    key_max: FloatArray
    value_center: FloatArray
    value_radius: float

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, (int, np.integer)) or self.count < 1:
            raise ValueError("count must be a positive integer")
        for field in ("key_min", "key_max", "value_center"):
            array = _array(getattr(self, field), 1, field).copy()
            array.flags.writeable = False
            object.__setattr__(self, field, array)
        if self.key_min.shape != self.key_max.shape or np.any(self.key_min > self.key_max):
            raise ValueError("key bounds must have matching shapes and min <= max")
        if not np.isfinite(self.value_radius) or self.value_radius < 0:
            raise ValueError("value_radius must be finite and nonnegative")


@dataclass(frozen=True)
class OmissionBound:
    error_upper_bound: float
    retained_output: FloatArray
    omitted_mass_upper_bound: float
    retained_tokens: int
    omitted_tokens: int


def summarize_page(keys: ArrayLike, values: ArrayLike) -> PageSummary:
    """Build exact-state summaries. Do not include future/masked positions."""
    k, v = _array(keys, 2, "keys"), _array(values, 2, "values")
    if len(k) != len(v):
        raise ValueError("keys and values must have matching token counts")
    center = v.mean(axis=0)
    radius = float(np.linalg.norm(v - center, axis=1).max())
    return PageSummary(len(k), k.min(axis=0), k.max(axis=0), center, radius)


def _attention(query: ArrayLike, keys: ArrayLike, values: ArrayLike) -> tuple[FloatArray, float]:
    q, k, v = _array(query, 1, "query"), _array(keys, 2, "keys"), _array(values, 2, "values")
    if k.shape[1] != len(q) or len(k) != len(v):
        raise ValueError("query/key widths and key/value token counts must match")
    with np.errstate(over="ignore", invalid="ignore"):
        logits = k @ q / np.sqrt(len(q))
    if not np.isfinite(logits).all():
        raise ValueError("attention logits overflowed float64")
    log_z = float(np.logaddexp.reduce(logits))
    output = np.exp(logits - log_z) @ v
    if not np.isfinite(output).all():
        raise ValueError("attention output overflowed float64")
    return output, log_z


def attention_output(query: ArrayLike, keys: ArrayLike, values: ArrayLike) -> FloatArray:
    """Stable dense reference for one query and one attention head."""
    return _attention(query, keys, values)[0]


def _fractional_box_bound(log_z: float, log_u: FloatArray, distances: FloatArray) -> float:
    """Maximize sum(m_p*D_p)/(Z+sum(m_p)), with 0 <= m_p <= u_p.

    Sorting D descending reduces the box maximum to its prefix vertices.
    Log-domain cumulative sums avoid exponentiating large raw attention logits.
    """
    if len(log_u) == 0:
        return 0.0
    order = np.argsort(-distances, kind="stable")
    log_mass = -np.inf
    log_numerator = -np.inf
    best = 0.0
    for idx in order:
        log_mass = np.logaddexp(log_mass, log_u[idx])
        if distances[idx] > 0:
            log_numerator = np.logaddexp(log_numerator, log_u[idx] + np.log(distances[idx]))
        candidate = float(np.exp(log_numerator - np.logaddexp(log_z, log_mass)))
        best = max(best, candidate)
    return best


def bound_omitted_pages(
    query: ArrayLike,
    retained_keys: ArrayLike,
    retained_values: ArrayLike,
    omitted_pages: Sequence[PageSummary],
) -> OmissionBound:
    """Bound ||dense_output - retained_output||_2 for a fixed KV state.

    Assumes disjoint retained/omitted partitions, complete page coverage, no
    quantization error, and standard scaled dot-product softmax without added
    position biases. RoPE keys may be supplied after their original rotation.
    Causality, cache lineage, and those partition assumptions are caller duties.
    This function cannot repair an already-diverged model trajectory.
    """
    q = _array(query, 1, "query")
    y, log_z = _attention(q, retained_keys, retained_values)
    if not omitted_pages:
        return OmissionBound(0.0, y, 0.0, len(np.asarray(retained_keys)), 0)

    log_u, distances = [], []
    for page in omitted_pages:
        if page.key_min.shape != q.shape or page.value_center.shape != y.shape:
            raise ValueError("page dimensions do not match query/value dimensions")
        # Coordinatewise bound on the largest possible q dot k in this page.
        bound_key = np.where(q >= 0, page.key_max, page.key_min)
        with np.errstate(over="ignore", invalid="ignore"):
            upper = float(q @ bound_key / np.sqrt(len(q)))
            distance = float(np.linalg.norm(page.value_center - y) + page.value_radius)
        if not np.isfinite(upper) or not np.isfinite(distance):
            raise ValueError("page bound overflowed float64")
        log_u.append(np.log(page.count) + upper)
        distances.append(distance)

    logs, dists = np.asarray(log_u), np.asarray(distances)
    bound = _fractional_box_bound(log_z, logs, dists)
    log_total = float(np.logaddexp.reduce(logs))
    mass_bound = float(np.exp(log_total - np.logaddexp(log_z, log_total)))
    return OmissionBound(
        bound, y, mass_bound, len(np.asarray(retained_keys)), sum(p.count for p in omitted_pages)
    )
