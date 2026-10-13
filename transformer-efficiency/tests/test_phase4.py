"""Random models ONLY unit fixtures; separate opt-in genuine pretrained checks."""
from dataclasses import replace
import os
from pathlib import Path
import json

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from test_context_adapter import random_adapter
from htis_context.adapter import LlamaReferenceAdapter
from htis_context.pages import BaselinePolicy, MemoryBudget
from htis_context.recovery import RecoveryPolicy, RecoveryConfig, split_pages
from htis_context.state import PrefixSnapshot, _fingerprint


def setup(a, config=None, budget=3, mask=None, offset=0):
    prefix=a.forward(torch.arange(1,33)[None],position_ids=torch.arange(offset,offset+32)[None],
                     attention_mask=mask).state
    config=config or RecoveryConfig(interval=1,cooldown=0,deterministic=True)
    p=RecoveryPolicy(BaselinePolicy("recent",max_pages=budget,prompt_tokens=8,recent_tokens=1),config)
    return prefix,p,LlamaReferenceAdapter(a.model,a.identity,policy=p)


def test_offload_reload_exact_positions_RoPE_GQA_layers(random_adapter):
    prefix,p,a=setup(random_adapter,offset=37)
    first=a.forward(torch.tensor([[33]]),state=prefix,position_ids=torch.tensor([[69]]))
    old={key:r.kv.clone() for key,r in first.state.archive._pages.items()}
    second=a.forward(torch.tensor([[34]]),state=first.state,position_ids=torch.tensor([[70]]))
    fetched=[e for e in second.metrics['recovery']['events'] if e['action']=='reload']
    assert fetched
    for e in fetched:
        current=second.state.layers[e['layer']]; source=old[e['layer'],e['page']]
        idx=torch.nonzero(torch.isin(current.slots,source.slots)).flatten()
        torch.testing.assert_close(current.keys[:,:,idx],source.keys,atol=0,rtol=0)
        torch.testing.assert_close(current.values[:,:,idx],source.values,atol=0,rtol=0)
        assert current.keys.shape[1]==2 and current.keys.shape[-1]==8
        assert e['original_slots']==source.slots.tolist()
    assert second.state.recipe.positions[0][-1]==70
    assert all(len(x.features.ids)<=3 for x in second.state.layers)
    with pytest.raises(ValueError,match='Approximate'):
        PrefixSnapshot(second.state)


@pytest.mark.parametrize('capacity',[0,100,5000])
def test_archive_capacity_missing_eviction(random_adapter,capacity):
    prefix,p,a=setup(random_adapter,RecoveryConfig(max_archive_bytes=capacity,interval=1,cooldown=0,deterministic=True))
    run=a.forward(torch.tensor([[33]]),state=prefix)
    assert run.state.archive.bytes<=capacity
    if capacity<=100:
        assert run.metrics['recovery']['discarded_pages']>0
        with pytest.raises(KeyError,match='unavailable'):
            run.state.archive.get(0,1,a.identity)
    else:
        assert any(x['action']=='archive-evict' for x in run.metrics['recovery']['events'])


@pytest.mark.parametrize('mutation',['lineage','memory-chain','mask','positions','tokens','values','scope'])
def test_stale_archive_rejected(random_adapter,mutation):
    prefix,p,a=setup(random_adapter)
    state=a.forward(torch.tensor([[33]]),state=prefix).state.fork()
    if mutation=='lineage': state.lineage='approximate:other-path'
    elif mutation=='memory-chain': state.memory_chain='other-trajectory'
    elif mutation=='scope': state.identity=replace(state.identity,access_scope='other')
    elif mutation=='values': state.layers[0].values.zero_()
    else:
        name={'mask':'mask','positions':'positions','tokens':'tokens'}[mutation]
        v=[list(x) for x in getattr(state.recipe,name)];v[0][1]+=1 if mutation!='mask' else -1
        state.recipe=replace(state.recipe,**{name:tuple(tuple(x) for x in v)})
    with pytest.raises(ValueError):
        a.forward(torch.tensor([[34]]),state=state)


def test_request_forks_and_private_integrity(random_adapter):
    prefix,p,a=setup(random_adapter)
    state=a.forward(torch.tensor([[33]]),state=prefix).state
    before=_fingerprint(state); count=len(state.archive._pages)
    fork=state.fork()
    a.forward(torch.tensor([[34]]),state=fork)
    assert _fingerprint(state)==before and len(state.archive._pages)==count
    k=next(iter(state.archive._pages));r=state.archive._pages[k]
    one=state.archive.get(*k,a.identity);one.values.zero_()
    assert not torch.equal(one.values,state.archive.get(*k,a.identity).values)
    with pytest.raises(ValueError,match='scope'):
        state.archive.get(*k,replace(a.identity,access_scope='other'))
    with torch.inference_mode(): r.kv.values.zero_()
    with pytest.raises(ValueError,match='integrity'):
        state.archive.get(*k,a.identity)


def test_task_change_repeated_recovery_and_hysteresis(random_adapter):
    prefix,p,a=setup(random_adapter,RecoveryConfig(interval=1,cooldown=3,deterministic=True))
    state=prefix; fetches=[]
    for i in range(33,40):
        p.recovery.task='new-task' if i>=35 else 'old-task'
        run=a.forward(torch.tensor([[i]]),state=state);state=run.state
        fetches.extend((i,e['layer'],e['page']) for e in run.metrics['recovery']['events'] if e['action']=='reload')
        if i==35: assert all('task-change' in d['triggers'] for d in run.metrics['recovery']['decisions'])
        assert all(len(x.features.ids)<=3 for x in state.layers)
    assert fetches
    for i,l,pid in fetches:
        later=[j for j,ll,pp in fetches if ll==l and pp==pid and j>i]
        assert not later or min(later)-i>=3


def test_expensive_route_skips_scorer(random_adapter):
    prefix,p,a=setup(random_adapter,RecoveryConfig(score_ms_prior=100,max_route_ms=1))
    def fail(*args,**kwargs): raise AssertionError('expensive scorer executed')
    p.base=type('NeverScore',(),{'name':'never','select':fail})()
    run=a.forward(torch.tensor([[33]]),state=prefix)
    assert all(d['route']=='cheap-recency' for d in run.metrics['recovery']['decisions'])
    assert run.metrics['recovery']['fetch_count']==0


@pytest.mark.parametrize('masked',[False,True])
def test_bound_complete_coverage_and_blockwise_current_state(random_adapter,masked):
    mask=torch.ones((1,32),dtype=torch.long)
    if masked: mask[:,9:13]=0
    prefix,p,a=setup(random_adapter,RecoveryConfig(tolerance=0,fallback=True,max_adaptive_fetches=0),mask=mask,offset=7)
    dense_state=prefix
    for token in [33,34,35]:
        actual=a.forward(torch.tensor([[token]]),state=prefix)
        exact=random_adapter.forward(torch.tensor([[token]]),state=dense_state)
        torch.testing.assert_close(actual.logits,exact.logits,atol=1e-6,rtol=1e-5)
        assert all(b['coverage'] and b['exact_current_state'] and b['tolerance_met'] for b in actual.metrics['recovery']['bounds'])
        assert actual.state.lineage.startswith('approximate')
        prefix=actual.state;dense_state=exact.state


def test_bound_adaptive_fetch_and_unavailable_fallback(random_adapter):
    prefix,p,a=setup(random_adapter,RecoveryConfig(tolerance=0,fallback=True,max_adaptive_fetches=1,interval=1,cooldown=0))
    run=a.forward(torch.tensor([[33]]),state=prefix)
    assert sum(b['adaptive_fetches'] for b in run.metrics['recovery']['bounds'])>0
    assert all(b['coverage'] for b in run.metrics['recovery']['bounds'])
    prefix,p,a=setup(random_adapter,RecoveryConfig(max_archive_bytes=0,tolerance=0,fallback=True,max_adaptive_fetches=0))
    run=a.forward(torch.tensor([[33]]),state=prefix)
    assert all(b['coverage'] and not b['tolerance_met'] and b['fallback']=='unavailable' for b in run.metrics['recovery']['bounds'])


def test_missing_initial_summaries_no_bound_claim(random_adapter):
    prefix,p,a=setup(random_adapter)
    sparse=LlamaReferenceAdapter(a.model,a.identity,policy=BaselinePolicy('recent',max_pages=3,prompt_tokens=8,recent_tokens=8))
    state=sparse.forward(torch.tensor([[33]]),state=prefix).state
    p=RecoveryPolicy(p.base,RecoveryConfig(tolerance=0,fallback=True))
    run=LlamaReferenceAdapter(a.model,a.identity,policy=p).forward(torch.tensor([[34]]),state=state)
    assert all(not b['coverage'] and not b['tolerance_met'] for b in run.metrics['recovery']['bounds'])


@pytest.mark.parametrize('mode',['active','archive-index','working','protected'])
def test_all_budgets_enforced(random_adapter,mode):
    config=RecoveryConfig(max_index_bytes=1) if mode=='archive-index' else RecoveryConfig(max_working_bytes=1) if mode=='working' else RecoveryConfig()
    prefix,p,a=setup(random_adapter,config,budget=1 if mode=='protected' else 3)
    if mode=='active': a.memory_budget=MemoryBudget(1,1000000)
    with pytest.raises((MemoryError,ValueError)):
        a.forward(torch.tensor([[33]]),state=prefix)


def test_full_capacity_identical_and_archive_adapter_required(random_adapter):
    prefix,p,a=setup(random_adapter,budget=32)
    run=a.forward(torch.tensor([[33]]),state=prefix)
    expected=random_adapter.forward(torch.tensor([[33]]),state=prefix)
    torch.testing.assert_close(run.logits,expected.logits,atol=0,rtol=0)
    assert run.state.lineage=='dense' and run.state.archive.bytes==0
    restored=PrefixSnapshot(run.state).restore(run.state.identity,run.state.recipe)
    assert restored.archive is None and restored.memory_chain is None
    random_adapter.forward(torch.tensor([[34]]),state=restored)
    with pytest.raises(ValueError,match='controller'):
        random_adapter.forward(torch.tensor([[34]]),state=run.state)


def test_bound_latency_budget_and_no_dense_history_repair(random_adapter):
    prefix,p,a=setup(random_adapter,RecoveryConfig(tolerance=0,fallback=True,max_fallback_ms=0,max_adaptive_fetches=0))
    run=a.forward(torch.tensor([[33]]),state=prefix)
    assert all(b['fallback']=='latency-budget' and not b['tolerance_met'] for b in run.metrics['recovery']['bounds'])
    # A supplied approximate state can be integrated exactly without restoring
    # what an earlier dense generation would have computed.
    prefix,p,a=setup(random_adapter,RecoveryConfig(tolerance=0,fallback=True,max_adaptive_fetches=0))
    with torch.inference_mode(): prefix.layers[1].values[:,:,8:16]*=100
    prefix.lineage='approximate:controlled-prior-state-drift'
    expected=random_adapter.forward(torch.tensor([[33]]),state=prefix)
    run=a.forward(torch.tensor([[33]]),state=prefix)
    torch.testing.assert_close(run.logits,expected.logits,atol=1e-6,rtol=1e-5)
    assert run.state.lineage.startswith('approximate')
    assert all(not b['historical_dense_repair'] for b in run.metrics['recovery']['bounds'])


def test_archive_transplant_and_config_change_rejected(random_adapter):
    prefix,p,a=setup(random_adapter)
    state=a.forward(torch.tensor([[33]]),state=prefix).state
    other=random_adapter.forward(torch.arange(2,35)[None]).state
    other.archive=state.archive
    other.lineage=state.lineage
    with pytest.raises(ValueError,match='Stale'):
        a.forward(torch.tensor([[34]]),state=other)
    changed=RecoveryPolicy(p.base,replace(p.recovery.config,interval=2))
    with pytest.raises(ValueError,match='configuration'):
        LlamaReferenceAdapter(a.model,a.identity,policy=changed).forward(torch.tensor([[34]]),state=state)


def test_runtime_recovery_does_not_import_offline_targets():
    import ast
    source=Path(__file__).resolve().parents[1]/'src/htis_context/recovery.py'
    imports=[node.module for node in ast.walk(ast.parse(source.read_text())) if isinstance(node,ast.ImportFrom)]
    assert not any(n and any(x in n for x in ['offline_labels','counterfactual','training','phase4']) for n in imports)


def test_value_sensitive_fetch_prioritizes_low_mass_large_value(random_adapter):
    import numpy as np
    from htis.bounded_omission import summarize_page
    prefix,p,a=setup(random_adapter)
    run=a.forward(torch.tensor([[33]]),state=prefix)
    controller=p.recovery
    # Explicit synthetic single-head mathematical fixture for ranking, never
    # called pretrained validation. Page 1 has smaller logits but huge values.
    controller.archive.summaries[0,1]=(summarize_page(np.full((8,8),-1.),np.full((8,8),10000.)),)
    controller.archive.summaries[0,2]=(summarize_page(np.ones((8,8)),np.ones((8,8))),)
    q=torch.ones(1,1,1,8);out=torch.zeros(1,1,1,8)
    assert controller.fetch_priority(0,1,q,out)>controller.fetch_priority(0,2,q,out)


def test_controlled_perturbation_and_validation_uncertainty_trigger(random_adapter):
    from htis_context.phase4 import perturb_value_page
    prefix,p,a=setup(random_adapter)
    before=prefix.layers[1].values.clone(); signature=a._weights_signature()
    perturb_value_page(prefix,1,1,100,8)
    torch.testing.assert_close(prefix.layers[1].values[:,:,8:16],before[:,:,8:16]*100,atol=0,rtol=0)
    assert a._weights_signature()==signature and prefix.lineage.startswith('approximate')
    p.recovery.calibration={'validation_residual_log_target_radius':0.8}
    run=a.forward(torch.tensor([[33]]),state=prefix)
    assert all('uncertainty' in d['triggers'] for d in run.metrics['recovery']['decisions'])


@pytest.mark.skipif(os.environ.get('HTIS_RUN_PRETRAINED')!='1',reason='Opt-in real pretrained weights')
def test_real_pretrained_recovery_and_full_fallback():
    from htis_context.checkpoint import load
    from htis_context.learned import load_estimator
    root=Path(__file__).resolve().parents[2]
    c=json.loads((root/'transformer-efficiency/configs/smollm2-135m.json').read_text())
    model,tok,dense,_=load(c,Path('.model-cache/smollm2-135m'))
    load_estimator(root/'transformer-efficiency/results/phase3/trained-v1/htis-seed-17',dense.identity)
    prefix=dense.forward(tok('A library keeps old books. The river flows past the house. '*5,return_tensors='pt').input_ids[:,:64]).state
    p=RecoveryPolicy(BaselinePolicy('recent',max_pages=4,prompt_tokens=8,recent_tokens=8),
                     RecoveryConfig(tolerance=0,fallback=True,max_adaptive_fetches=1,max_route_ms=20))
    a=LlamaReferenceAdapter(model,dense.identity,policy=p)
    state=prefix;exact=prefix
    for token in tok(' Remember the library.',add_special_tokens=False).input_ids[:3]:
        run=a.forward(torch.tensor([[token]]),state=state)
        teacher=dense.forward(torch.tensor([[token]]),state=exact)
        torch.testing.assert_close(run.logits,teacher.logits,atol=c['atol'],rtol=c['rtol'])
        assert all(b.get('exact_current_state',False) for b in run.metrics['recovery']['bounds'])
        state=run.state;exact=teacher.state
