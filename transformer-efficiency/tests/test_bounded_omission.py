"""Numerical checks for the fixed-state local bound; not LLM benchmarks."""
from itertools import product

import numpy as np
import pytest

from htis.bounded_omission import (
    PageSummary,
    _fractional_box_bound,
    attention_output,
    bound_omitted_pages,
    summarize_page,
)


@pytest.mark.parametrize("seed", range(24))
def test_bound_covers_actual_omission_error(seed):
    rng = np.random.default_rng(seed)
    q, k, v = rng.normal(size=7), rng.normal(size=(41, 7)), rng.normal(size=(41, 5))
    retained = rng.permutation(41)[:9]
    omitted = np.setdiff1d(np.arange(41), retained)
    pages = [summarize_page(k[idx], v[idx]) for idx in np.array_split(omitted, 5)]
    bound = bound_omitted_pages(q, k[retained], v[retained], pages)
    dense = attention_output(q, k, v)
    error = np.linalg.norm(dense - bound.retained_output)
    assert error <= bound.error_upper_bound + 1e-11
    assert 0 <= bound.omitted_mass_upper_bound <= 1 + 1e-12
    assert bound.retained_tokens + bound.omitted_tokens == len(k)


@pytest.mark.parametrize("seed", range(8))
def test_fractional_box_solution_matches_all_vertices(seed):
    rng = np.random.default_rng(seed)
    z = float(rng.uniform(0.1, 4))
    u, d = rng.uniform(0.1, 5, 6), rng.uniform(0, 6, 6)
    brute = max(float((u * bits) @ d / (z + (u * bits).sum())) for bits in product([0, 1], repeat=6))
    actual = _fractional_box_bound(np.log(z), np.log(u), d)
    assert actual == pytest.approx(brute, abs=1e-12)


def test_no_omission_is_exact():
    result = bound_omitted_pages([1.0], [[0.0], [1.0]], [[2.0], [4.0]], [])
    assert result.error_upper_bound == 0
    assert result.omitted_mass_upper_bound == 0


def test_identical_values_need_no_mass_to_error_assumption():
    page = summarize_page([[30.0], [-30.0]], [[7.0], [7.0]])
    result = bound_omitted_pages([1.0], [[0.0]], [[7.0]], [page])
    assert result.error_upper_bound == pytest.approx(0, abs=1e-12)
    assert result.omitted_mass_upper_bound > 0.99


def test_small_attention_mass_can_hide_large_value_effect():
    page = summarize_page([[-8.0]], [[100_000.0]])
    result = bound_omitted_pages([1.0], [[0.0]], [[0.0]], [page])
    actual = attention_output([1.0], [[0.0], [-8.0]], [[0.0], [100_000.0]])[0]
    assert result.omitted_mass_upper_bound < 0.001
    assert actual > 30
    assert result.error_upper_bound == pytest.approx(actual)


def test_log_domain_handles_extreme_finite_logits():
    page = summarize_page([[1001.0]], [[3.0]])
    result = bound_omitted_pages([1.0], [[1000.0]], [[1.0]], [page])
    assert np.isfinite(result.error_upper_bound)
    assert result.error_upper_bound == pytest.approx(2 / (1 + np.exp(-1)))


def test_page_owns_immutable_summaries():
    minimum = np.array([0.0])
    page = PageSummary(1, minimum, np.array([0.0]), np.array([1.0]), 0.0)
    minimum[0] = -9
    assert page.key_min[0] == 0
    with pytest.raises(ValueError):
        page.key_min[0] = -1


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_page_counts_rejected(count):
    with pytest.raises(ValueError):
        PageSummary(count, np.array([0.0]), np.array([0.0]), np.array([1.0]), 0.0)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        summarize_page([[np.nan]], [[0]])
    with pytest.raises(ValueError):
        summarize_page([[0], [1]], [[0]])
    with pytest.raises(ValueError):
        bound_omitted_pages([1], [], [], [])
    with pytest.raises(ValueError):
        bound_omitted_pages([1, 2], [[1]], [[1]], [])
    with pytest.raises(ValueError):
        PageSummary(1, np.array([1.0]), np.array([0.0]), np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        PageSummary(1, np.array([0.0]), np.array([0.0]), np.array([1.0]), -1.0)
