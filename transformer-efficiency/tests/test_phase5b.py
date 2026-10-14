"""Experimental contract, leakage and real execution semantics (small random fixtures)."""
from copy import deepcopy
import pytest
import torch

from test_context_adapter import random_adapter
from htis_context.phase5b import budget_bytes, validate_selection, make_adapter, generate, process_memory
from htis_context.phase5b_diagnostics import ordinal_correlation
from htis_context.prefix_cache import ExactPrefixCache
from htis_context.provenance import SourceRecipe


@pytest.mark.parametrize('pages,expected',[(4,1474560),(8,2949120),(16,5898240)])
def test_physical_gqa_budget(pages,expected):
    assert budget_bytes(dict(layers=30,kv_heads=3,head_dim=64),pages)==expected


def trial(candidate,case='a',split='validation',f1=.2,nll=2.,latency=5.):
    return dict(candidate=candidate,case=case,split=split,policy='query',seed=17,f1=f1,mean_nll=nll,mean_decode_ms=latency)


def test_validation_threshold_selection_uses_quality_before_speed():
    rows=[trial(0,f1=.3,latency=10),trial(1,f1=.2,latency=1)]
    assert validate_selection(rows,[{},{}])['selected_candidate']==0


def test_validation_threshold_selection_rejects_heldout_and_mismatched_pairs():
    with pytest.raises(ValueError,match='validation'):
        validate_selection([trial(0,split='test')],[{}])
    with pytest.raises(ValueError,match='matched'):
        validate_selection([trial(0),trial(1,case='b')],[{},{}])


def config():
    return dict(page_size=8,prompt_tokens=8,sink_tokens=8,recent_tokens=8,working_bytes=64*1024**2)


def test_full_recovery_identity_does_not_accidentally_bound_dense_cache(random_adapter):
    dense=random_adapter;prefix=dense.forward(torch.arange(1,41)[None]).state
    a=make_adapter(dense.model,dense,'full',4,17,config(),{'not_a_recovery_field':True})
    result=a.forward(torch.tensor([[41]]),state=prefix)
    expected=dense.forward(torch.tensor([[41]]),state=prefix)
    assert result.state.kv_bytes==expected.state.kv_bytes
    torch.testing.assert_close(result.logits,expected.logits)


@pytest.mark.parametrize('name',['recent','query','random'])
def test_budget_and_exact_prefix_numerical_equivalence(random_adapter,name):
    dense=random_adapter;prefix=dense.forward((torch.arange(1,65)%dense.model.config.vocab_size)[None]).state
    cache=ExactPrefixCache(4*1024**2);assert cache.put(prefix)
    restored=cache.get(dense.identity,prefix.recipe)
    a=make_adapter(dense.model,dense,name,4,17,config())
    b=make_adapter(dense.model,dense,name,4,17,config())
    cold=a.forward(torch.tensor([[3]]),state=prefix);warm=b.forward(torch.tensor([[3]]),state=restored)
    torch.testing.assert_close(cold.logits,warm.logits)
    assert all(len(x.features.ids)<=4 for x in cold.state.layers)
    changed=deepcopy(prefix.recipe.tokens);changed=tuple((2,)+x[1:] for x in changed)
    assert cache.get(dense.identity,SourceRecipe(changed,prefix.recipe.positions,prefix.recipe.mask)) is None


def test_rank_diagnostic_detects_task_change_and_is_identical_for_fixed_scores():
    assert ordinal_correlation([1,2,3],[1,2,3])==pytest.approx(1)
    assert ordinal_correlation([1,2,3],[3,2,1])==pytest.approx(-1)


def test_process_memory_is_explicitly_process_scoped():
    memory=process_memory()
    assert 'scope' in memory
    if 'working_set_bytes' in memory:
        assert 0<memory['working_set_bytes']<=memory['process_lifetime_peak_working_set_bytes']
