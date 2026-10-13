"""Tests for causality, hard budgets, stable attention and scoring mechanics."""
import numpy as np
import pytest

from htis.cache import CachePolicy, StreamingKVCache, attention


def test_attention_is_normalized_and_stable():
    q = np.array([1000., 0., 0., 0.])
    k = np.array([[1000., 0., 0., 0.], [-1000., 0., 0., 0.]])
    v = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    out, weights = attention(q, k, v)
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
    np.testing.assert_allclose(out, v[0], atol=1e-12)


def test_all_policies_enforce_budget_and_retained_index_order():
    for policy in CachePolicy:
        cache = StreamingKVCache(6, policy=policy, sink_tokens=2, recent_tokens=2)
        for t in range(30):
            out, weights, positions = cache.step(np.ones(4), np.full(4, t), np.full(4, t))
            assert len(positions) <= 6
            assert positions == tuple(sorted(positions))
            assert t in positions
            assert len(weights) == len(positions)
            assert out.shape == (4,)
            assert cache.bytes_used() == 2 * len(positions) * 4 * 2
        assert cache.stored_tokens == 6


def test_hybrid_protects_early_sinks_and_recent_tokens():
    cache = StreamingKVCache(8, policy="hybrid", sink_tokens=2, recent_tokens=3)
    for t in range(35):
        _, _, positions = cache.step(np.ones(4), np.ones(4), np.ones(4))
        if t >= 8:
            assert {0, 1}.issubset(positions)
            assert {t, t - 1, t - 2}.issubset(positions)


def test_recent_policy_does_not_use_future_queries():
    a = StreamingKVCache(4, policy="recent", sink_tokens=0, recent_tokens=1)
    b = StreamingKVCache(4, policy="recent", sink_tokens=0, recent_tokens=1)
    for t in range(12):
        q = np.array([t + 1., 0., 0., 0.])
        args = (q, np.ones(4), np.ones(4))
        out_a, _, pos_a = a.step(*args)
        out_b, _, pos_b = b.step(*args)
        assert pos_a == pos_b
        np.testing.assert_array_equal(out_a, out_b)
    assert a.positions == (8, 9, 10, 11)


def test_weight_changes_hybrid_selection_based_on_observed_attention():
    # Hot first non-sink key is repeatedly referenced before eviction.
    high = StreamingKVCache(5, policy="hybrid", sink_tokens=0, recent_tokens=1,
                            attention_weight=1.0, decay=1.0)
    low = StreamingKVCache(5, policy="hybrid", sink_tokens=0, recent_tokens=1,
                           attention_weight=0.0, decay=1.0)
    keys = np.eye(8)
    for t in range(12):
        q = keys[1] * 50
        k = keys[t % 8]
        for cache in (high, low):
            cache.step(q, k, k)
    assert 1 in high.positions
    assert 1 not in low.positions


def test_invalid_inputs_and_dimensions_rejected():
    with pytest.raises(ValueError):
        StreamingKVCache(0)
    with pytest.raises(ValueError):
        StreamingKVCache(4, sink_tokens=2, recent_tokens=3)
    with pytest.raises(ValueError):
        StreamingKVCache(4, attention_weight=1.1)
    with pytest.raises(ValueError):
        attention(np.ones(4), np.ones((2, 5)), np.ones((2, 5)))
    cache = StreamingKVCache(4, sink_tokens=0, recent_tokens=1)
    cache.step(np.ones(4), np.ones(4), np.ones(4))
    with pytest.raises(ValueError):
        cache.step(np.ones(5), np.ones(5), np.ones(5))
    with pytest.raises(ValueError):
        cache.step(np.ones(4), np.array([np.nan, 1., 1., 1.]), np.ones(4))


def test_sink_recent_when_budget_only_holds_sinks_and_current():
    cache = StreamingKVCache(3, policy="sink_recent", sink_tokens=2, recent_tokens=1)
    for t in range(10):
        _, _, positions = cache.step(np.ones(4), np.ones(4), np.ones(4))
    assert positions == (0, 1, 9)
