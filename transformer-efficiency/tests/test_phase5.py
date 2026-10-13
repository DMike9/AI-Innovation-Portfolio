"""Synthetic unit fixtures are separate from opt-in real checkpoint validation."""
import json
import os
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('transformers')
from test_context_adapter import random_adapter
from htis_context.phase5_metrics import answer_metrics,normalize,paired_bootstrap,dominated
from htis_context.phase5_policy import ConditionalHTIS,choose_validation,BenefitCostController,PeriodicBoundController
from htis_context.learned import RiskEstimator,TrainedPagePolicy
from htis_context.adapter import LlamaReferenceAdapter
from htis_context.pages import BaselinePolicy
from htis_context.recovery import RecoveryPolicy,RecoveryConfig
from htis_context.phase5_gpu import availability,timed,run as gpu_run
from htis_context.provenance import SourceRecipe
from htis_context.prefix_cache import ExactPrefixCache
from htis_context.state import _fingerprint


@pytest.mark.parametrize('pred,answers,em,f1',[
    ('The Blue!',['blue'],1,1),('blue box',['box'],0,2/3),('',['blue'],0,0),
    ('Blue',['red','blue'],1,1),('a',['the'],1,1)])
def test_official_style_answer_normalization(pred,answers,em,f1):
    out=answer_metrics(pred,answers);assert out['em']==em;assert out['f1']==pytest.approx(f1)


def test_bootstrap_clusters_seeds_without_token_replication():
    a=paired_bootstrap([('doc1',1),('doc1',3),('doc2',4)],repetitions=100)
    assert a['groups']==2 and a['mean_delta']==3
    assert a==paired_bootstrap([('doc1',1),('doc1',3),('doc2',4)],repetitions=100)


def test_pareto_reports_dominated_and_ties():
    rows=[{'arm':'simple','mean_f1':.5,'mean_decode_ms':1,'max_total_storage_bytes':10},
          {'arm':'advanced','mean_f1':.4,'mean_decode_ms':2,'max_total_storage_bytes':20},
          {'arm':'tie','mean_f1':.5,'mean_decode_ms':1,'max_total_storage_bytes':10}]
    assert dominated(rows)=={'simple':[],'advanced':['simple','tie'],'tie':[]}


def val(arm,case,kl,time,f1=0):return dict(split='validation',arm=arm,case=case,seed=17,mean_kl=kl,mean_decode_ms=time,f1=f1)


def test_validation_selection_rejects_fast_quality_loss_and_test_labels():
    rows=[val('recent','x',.1,10),val('htis','x',.2,1),val('conditional','x',.11,5)]
    assert choose_validation(rows)['selected_arm']=='conditional'
    rows[0]['split']='test'
    with pytest.raises(ValueError,match='validation'):choose_validation(rows)


def test_validation_selection_requires_paired_cases():
    with pytest.raises(ValueError,match='identical'):choose_validation([val('recent','x',.1,1),val('htis','y',.1,1)])


def learned_fixture(a):
    # Clearly synthetic random estimator, ONLY tests routing/control semantics.
    return TrainedPagePolicy(RiskEstimator('htis'),torch.zeros(39),torch.ones(39),page_size=8,max_pages=3,
        layer_count=len(a.model.model.layers),prompt_tokens=8,sink_tokens=8,recent_tokens=1)


def test_conditional_interval_one_matches_original_without_mutating_prefix(random_adapter):
    torch.manual_seed(9);base=learned_fixture(random_adapter)
    import copy
    conditional=ConditionalHTIS(copy.deepcopy(base),interval=1,threshold=0)
    prefix=random_adapter.forward(torch.arange(1,41)[None]).state;before=_fingerprint(prefix)
    a=LlamaReferenceAdapter(random_adapter.model,random_adapter.identity,policy=base)
    b=LlamaReferenceAdapter(random_adapter.model,random_adapter.identity,policy=conditional)
    for token in (4,5,6):
        x=a.forward(torch.tensor([[token]]),state=prefix);y=b.forward(torch.tensor([[token]]),state=prefix)
        torch.testing.assert_close(x.logits,y.logits,atol=0,rtol=0)
    assert _fingerprint(prefix)==before and conditional.scoring_calls==3*len(prefix.layers)


def test_conditional_skip_is_actual_recency_selection(random_adapter):
    base=learned_fixture(random_adapter);p=ConditionalHTIS(base,interval=1000,threshold=2)
    prefix=random_adapter.forward(torch.arange(1,41)[None]).state
    x=prefix.layers[0];q=torch.ones(1,random_adapter.model.config.num_attention_heads,1,x.keys.shape[-1])
    p.select(x.slots,x.history,x.features,q,now=40)
    p.cursor=0
    chosen=p.select(x.slots,x.history,x.features,q,now=41)
    expected=p.cheap.select(x.slots,x.history,x.features,q,now=41)
    assert torch.equal(chosen[0],expected[0]);assert p.scoring_calls==1
    assert p.query_storage_bytes==q.numel()*q.element_size()


def test_recomputed_features_match_and_hook_restored_even_on_error(random_adapter):
    from htis_context.phase5_feature_ablation import RecomputedFeatureAdapter,recompute_features
    from htis_context import adapter as module
    original=module.update_features
    prefix=random_adapter.forward(torch.arange(1,41)[None]).state
    p=BaselinePolicy('recent',max_pages=3,prompt_tokens=8,recent_tokens=1)
    a=LlamaReferenceAdapter(random_adapter.model,random_adapter.identity,policy=p)
    b=RecomputedFeatureAdapter(random_adapter.model,random_adapter.identity,policy=p)
    x=a.forward(torch.tensor([[4]]),state=prefix);y=b.forward(torch.tensor([[4]]),state=prefix)
    torch.testing.assert_close(x.logits,y.logits,atol=0,rtol=0)
    for token in (5,6):
        x=a.forward(torch.tensor([[token]]),state=x.state);y=b.forward(torch.tensor([[token]]),state=y.state)
        torch.testing.assert_close(x.logits,y.logits,atol=0,rtol=0)
    assert module.update_features is original
    with pytest.raises(RuntimeError):
        with recompute_features():raise RuntimeError('forced failure')
    assert module.update_features is original


def test_benefit_gate_zero_is_cheap_but_preserves_archive_provenance(random_adapter):
    base=BaselinePolicy('recent',max_pages=3,prompt_tokens=8,recent_tokens=1)
    cfg=RecoveryConfig(interval=1,cooldown=0)
    p=RecoveryPolicy(base,cfg);p.recovery=BenefitCostController(p,cfg,{},benefit_ms_per_margin=0,processing_ms_prior=1)
    a=LlamaReferenceAdapter(random_adapter.model,random_adapter.identity,policy=p)
    prefix=random_adapter.forward(torch.arange(1,41)[None]).state
    x=a.forward(torch.tensor([[41]]),state=prefix);y=a.forward(torch.tensor([[42]]),state=x.state)
    assert y.state.archive.config==cfg and p.recovery.config==cfg
    y.state.archive.validate(y.state)
    assert all(not d['benefit_cost_gate'] and d['route']=='cheap-recency' for d in y.metrics['recovery']['decisions'])
    assert not any(e['action']=='reload' for e in y.metrics['recovery']['events'])


def test_periodic_bound_skip_never_claims_coverage(random_adapter):
    cfg=RecoveryConfig(tolerance=.1)
    p=RecoveryPolicy(BaselinePolicy('recent',max_pages=3,prompt_tokens=8,recent_tokens=1),cfg)
    p.recovery=PeriodicBoundController(p,cfg,{},every=1000)
    a=LlamaReferenceAdapter(random_adapter.model,random_adapter.identity,policy=p)
    prefix=random_adapter.forward(torch.arange(1,41)[None]).state
    x=a.forward(torch.tensor([[41]]),state=prefix)
    assert all(not b['coverage'] and not b['bound_evaluated'] for b in x.metrics['recovery']['bounds'])


def test_gpu_dryrun_does_not_load_checkpoint_or_allocate_gpu(tmp_path,monkeypatch):
    import htis_context.phase5_gpu as module
    monkeypatch.setattr(module,'load',lambda *args,**kwargs:pytest.fail('dry-run must not load'))
    result=gpu_run(tmp_path/'dry.json',dry_run=True)
    assert result['executed'] is False and not result['cpu_dry_run_is_gpu_execution']
    assert json.loads((tmp_path/'dry.json').read_text())==result


def test_synchronized_timer_calls_before_and_after():
    calls=[]
    result,elapsed=timed(lambda:calls.append('compute') or 7,lambda:calls.append('sync'))
    assert result==7 and elapsed>=0 and calls==['sync','compute','sync']


def test_phase5_data_frozen_groups_and_no_official_truncation():
    path=Path('transformer-efficiency/data/phase5-short-v1/manifest.json')
    if not path.exists():pytest.skip('Frozen Phase 5 data not installed')
    data=json.loads(path.read_text());rows=data['rows']
    groups={r['group'] for r in rows if r['split']=='validation'}
    assert not groups.intersection(r['group'] for r in rows if r['split']=='test')
    assert all(r['input_tokens']+6<=256 for r in rows)
    assert all(r['official_paragraph_untruncated'] for r in rows if r['kind']=='official-short-qa')


@pytest.mark.pretrained
@pytest.mark.skipif(os.environ.get('HTIS_RUN_PRETRAINED')!='1',reason='Opt-in genuine pretrained checkpoint')
def test_real_pretrained_sdpa_equivalence_and_conditional_full_activation():
    from htis_context.checkpoint import load
    from htis_context.phase5 import upstream_sdpa,policy,definition
    cfg=json.loads(Path('transformer-efficiency/configs/smollm2-135m.json').read_text())
    model,tok,dense,_=load(cfg,Path('.model-cache/smollm2-135m'))
    ids=tok('The green book is on the shelf. '*9,add_special_tokens=False).input_ids
    report=upstream_sdpa(model,ids,[4,5,6,7,8,9],atol=cfg['atol'],rtol=cfg['rtol'],dense=dense)
    assert all(x['passed'] for x in report['checks'])
    c=definition()['configuration'];a=policy('htis',c,dense.identity,30)
    b=ConditionalHTIS(policy('htis',c,dense.identity,30),interval=1,threshold=0)
    prefix=dense.forward(torch.tensor([ids])).state
    x=LlamaReferenceAdapter(model,dense.identity,policy=a).forward(torch.tensor([[4]]),state=prefix)
    y=LlamaReferenceAdapter(model,dense.identity,policy=b).forward(torch.tensor([[4]]),state=prefix)
    torch.testing.assert_close(x.logits,y.logits,atol=0,rtol=0)
