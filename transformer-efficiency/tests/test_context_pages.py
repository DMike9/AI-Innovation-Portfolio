"""Physical page selection tests on labeled random unit fixtures."""
from dataclasses import replace

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from htis_context.adapter import LlamaReferenceAdapter
from htis_context.pages import BaselinePolicy, MemoryBudget
from htis_context.prefix_cache import ExactPrefixCache
from htis_context.provenance import SourceRecipe, SemanticRoute
from htis_context.state import PrefixSnapshot
from test_context_adapter import random_adapter


POLICIES = ["recent", "sinks_recent", "prompt_random", "attention_history", "query_summary"]


@pytest.mark.parametrize("kwargs", [{"page_size": 0}, {"page_size": 1.5}, {"max_pages": True},
                                    {"name": "trained-htis"}, {"recent_tokens": 0}])
def test_invalid_page_policies_rejected(kwargs):
    with pytest.raises(ValueError):
        BaselinePolicy(**kwargs)


@pytest.mark.parametrize("policy", POLICIES)
def test_real_attention_inputs_are_physically_selected(random_adapter, policy):
    a = random_adapter
    prefix = a.forward(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])).state
    sparse = LlamaReferenceAdapter(a.model, a.identity,
                                  policy=BaselinePolicy(policy, page_size=8, max_pages=2, prompt_tokens=8))
    result = sparse.forward(torch.tensor([[17]]), state=prefix)
    assert result.state.lineage.startswith("approximate")
    for layer, event in zip(result.state.layers, result.metrics["attention_inputs"]):
        assert event["before_selection_tokens"] == 17
        assert event["key_tokens"] == layer.keys.shape[2] == 9
        assert layer.keys.shape[1] == 2  # physical KV groups, not 4 query heads
        assert layer.slots.tolist() == list(range(8)) + [16]
        assert event["page_ids"] == [0, 2]
        assert layer.features.ids.tolist() == [0, 2]
        assert result.state.recipe.positions[0][16] == 16
    assert prefix.layers[0].keys.shape[2] == 16
    with pytest.raises(ValueError, match="Approximate"):
        ExactPrefixCache(100000).put(result.state)


@pytest.mark.parametrize("policy", POLICIES)
def test_full_capacity_is_dense_equivalent(random_adapter, policy):
    a = random_adapter
    state = a.forward(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])).state
    sparse = LlamaReferenceAdapter(a.model, a.identity,
                                  policy=BaselinePolicy(policy, max_pages=32, prompt_tokens=8))
    for token in (9, 10, 11):
        expected = a.forward(torch.tensor([[token]]), state=state)
        actual = sparse.forward(torch.tensor([[token]]), state=state)
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
        assert actual.state.lineage == "dense"
        state = actual.state


def test_retained_keys_keep_original_RoPE_rotation(random_adapter):
    a = random_adapter
    ids = torch.arange(1, 18)[None]
    pos = torch.arange(37, 54)[None]
    prefix = a.forward(ids[:, :16], position_ids=pos[:, :16]).state
    dense = a.forward(ids[:, 16:], state=prefix, position_ids=pos[:, 16:])
    sparse = LlamaReferenceAdapter(a.model, a.identity, policy=BaselinePolicy("recent", max_pages=2, prompt_tokens=8))
    actual = sparse.forward(ids[:, 16:], state=prefix, position_ids=pos[:, 16:])
    # Layer 0 projected keys cannot have changed before its intervention.
    selected = actual.state.layers[0].slots
    torch.testing.assert_close(actual.state.layers[0].keys, dense.state.layers[0].keys[:, :, selected], atol=0, rtol=0)
    assert actual.state.recipe.positions[0][-1] == 53


def test_insertion_features_reused_and_partial_page_extended(random_adapter):
    a = random_adapter
    prefix = a.forward(torch.arange(1, 10)[None]).state
    actual = a.forward(torch.tensor([[10]]), state=prefix)
    old, new = prefix.layers[0].features, actual.state.layers[0].features
    torch.testing.assert_close(old.key_min[:, 0], new.key_min[:, 0], atol=0, rtol=0)
    k = actual.state.layers[0].keys[0, :, 8:]
    torch.testing.assert_close(new.key_min[:, 1], k.amin(dim=1), atol=0, rtol=0)


def test_protected_region_and_byte_limits_enforced(random_adapter, monkeypatch):
    a = random_adapter
    prefix = a.forward(torch.arange(1, 18)[None]).state
    bad = LlamaReferenceAdapter(a.model, a.identity, policy=BaselinePolicy("recent", max_pages=1, prompt_tokens=8))
    with pytest.raises(ValueError, match="Protected"):
        bad.forward(torch.tensor([[18]]), state=prefix)
    bounded = LlamaReferenceAdapter(a.model, a.identity, memory_budget=MemoryBudget(1, 100000))
    original = torch.matmul
    calls = []
    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "matmul", tracked)
    with pytest.raises(MemoryError, match="KV"):
        bounded.forward(torch.tensor([[1, 2]]))
    assert not calls  # budget failure occurs before attention matmul
    changed = LlamaReferenceAdapter(a.model, a.identity, policy=BaselinePolicy("full", page_size=4))
    with pytest.raises(ValueError, match="Page size"):
        changed.forward(torch.tensor([[18]]), state=prefix)


def test_sparse_batch_and_chunk_rejected(random_adapter):
    a = random_adapter
    sparse = LlamaReferenceAdapter(a.model, a.identity, policy=BaselinePolicy("recent", max_pages=2))
    with pytest.raises(ValueError, match="Sparse"):
        sparse.forward(torch.tensor([[1, 2]]))
    with pytest.raises(ValueError, match="Sparse"):
        sparse.forward(torch.tensor([[1], [2]]))


def test_matmul_batch_dimensions_do_not_repeat_KV_groups(random_adapter, monkeypatch):
    observed = []
    original = torch.matmul
    def tracked(left, right, *args, **kwargs):
        if left.ndim == 4 and right.ndim == 4:
            observed.append((left.shape, right.shape))
        return original(left, right, *args, **kwargs)
    monkeypatch.setattr(torch, "matmul", tracked)
    random_adapter.forward(torch.tensor([[1, 2, 3]]))
    assert observed
    assert all(a[1] == b[1] == 2 for a, b in observed)


def test_prefix_reuse_is_copy_isolated_and_equivalent(random_adapter):
    a = random_adapter
    prefix = a.forward(torch.tensor([[1, 2, 3]])).state
    cache = ExactPrefixCache(100000)
    assert cache.put(prefix)
    first = cache.get(a.identity, prefix.recipe)
    second = cache.get(a.identity, prefix.recipe)
    torch.testing.assert_close(a.forward(torch.tensor([[4, 5]]), state=first).logits,
                               a.forward(torch.tensor([[1, 2, 3, 4, 5]])).logits[:, 3:], atol=1e-6, rtol=1e-5)
    first.layers[0].keys.zero_()
    assert torch.equal(second.layers[0].keys, prefix.layers[0].keys)
    assert cache.bytes_used > prefix.kv_bytes  # features + provenance charged
    assert cache.events[-1]["event"] == "hit"


@pytest.mark.parametrize("field,value", [
    ("model_id", "different-model"), ("revision", "1" * 40), ("config_hash", "changed"),
    ("tokenizer_revision", "2" * 40), ("template_hash", "changed"), ("rope_hash", "changed"),
    ("dtype", "torch.float16"), ("adapters", ("adapter-x",)), ("access_scope", "tenant-b"),
    ("mask_semantics", "noncausal"), ("cache_format", "quantized"), ("implementation", "changed-lineage")
])
def test_identity_invalidation(random_adapter, field, value):
    state = random_adapter.forward(torch.tensor([[1, 2, 3]])).state
    cache = ExactPrefixCache(100000)
    cache.put(state)
    assert cache.get(replace(state.identity, **{field: value}), state.recipe) is None


@pytest.mark.parametrize("recipe", [
    SourceRecipe(((9, 2, 3),), ((0, 1, 2),), ((1, 1, 1),)),  # changed system/preceding context
    SourceRecipe(((2, 3),), ((1, 2),), ((1, 1),)),  # same chunk, different ancestry
    SourceRecipe(((1, 2, 3),), ((1, 2, 3),), ((1, 1, 1),)),  # shifted positions
    SourceRecipe(((1, 2, 3),), ((0, 1, 2),), ((0, 1, 1),)),  # padding variant
    SourceRecipe(((1, 3, 2),), ((0, 1, 2),), ((1, 1, 1),)),  # reordered source
])
def test_ancestry_positions_padding_invalidation(random_adapter, recipe):
    state = random_adapter.forward(torch.tensor([[1, 2, 3]])).state
    cache = ExactPrefixCache(100000)
    cache.put(state)
    assert cache.get(state.identity, recipe) is None
    with pytest.raises(TypeError):
        cache.get(state.identity, SemanticRoute("source", "embed", (1., 2.)))


@pytest.mark.parametrize("eviction", ["lru", "lfu"])
def test_deterministic_eviction_and_oversized_admission(random_adapter, eviction):
    a = random_adapter
    states = [a.forward(torch.tensor([[i, 2, 3]])).state for i in (1, 4, 5)]
    measure = ExactPrefixCache(100000)
    measure.put(states[0])
    size = measure.bytes_used
    cache = ExactPrefixCache(2 * size + 100, eviction=eviction)
    cache.put(states[0]); cache.put(states[1])
    cache.get(states[0].identity, states[0].recipe)
    cache.put(states[2])
    assert cache.get(states[1].identity, states[1].recipe) is None
    assert cache.get(states[0].identity, states[0].recipe) is not None
    assert cache.bytes_used <= cache.max_bytes
    assert not ExactPrefixCache(1).put(states[0])
