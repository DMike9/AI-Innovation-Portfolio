"""Phase 3 unit fixtures are synthetic; real checkpoint test is explicitly opt-in."""
import ast
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest
torch=pytest.importorskip("torch")
pytest.importorskip("transformers")

from htis_context.adapter import LlamaReferenceAdapter
from htis_context.counterfactual_data import validate_splits
from htis_context.learned import RiskEstimator, TrainedPagePolicy, load_estimator
from htis_context.offline_labels import CapturePolicy, InterventionPolicy, kl, local_damage
from htis_context.pages import BaselinePolicy
from htis_context.runtime_features import WIDTH, SCHEMA, extract
from htis_context.training import read_dataset
from htis_context.provenance import ExecutionIdentity
from test_context_adapter import random_adapter


def captured(adapter, ids, prefix=None):
    capture=CapturePolicy(8,len(adapter.model.model.layers))
    result=LlamaReferenceAdapter(adapter.model,adapter.identity,policy=capture).forward(ids,state=prefix)
    return capture,result


def test_capture_keeps_existing_dense_output_and_observes_prequery_history(random_adapter):
    a=random_adapter
    prefix=a.forward(torch.arange(1,18)[None]).state
    capture,result=captured(a,torch.tensor([[18]]),prefix)
    expected=a.forward(torch.tensor([[18]]),state=prefix)
    torch.testing.assert_close(result.logits,expected.logits,atol=0,rtol=0)
    assert capture.calls[0]['history'][:,-1].eq(0).all()
    assert result.state.layers[0].history[:,-1].gt(0).all()
    assert capture.calls[0]['features'].values.shape==(3,WIDTH)


def test_single_and_multi_page_interventions_are_physical(random_adapter):
    a=random_adapter
    prefix=a.forward(torch.arange(1,26)[None]).state
    capture,full=captured(a,torch.tensor([[26]]),prefix)
    for pages in [(1,),(1,2)]:
        policy=InterventionPolicy(8,0,pages,4)
        changed=LlamaReferenceAdapter(a.model,a.identity,policy=policy).forward(torch.tensor([[26]]),state=prefix)
        assert not set(pages)&set(changed.state.layers[0].features.ids.tolist())
        assert changed.state.lineage!='dense'
        assert local_damage(capture.calls[0]['query'],full.state.layers[0],pages,8)>=0
        assert kl(full.logits,changed.logits)>=0
    torch.testing.assert_close(prefix.layers[0].keys,a.forward(torch.arange(1,26)[None]).state.layers[0].keys)


def test_runtime_features_have_no_future_dependency(random_adapter):
    a=random_adapter
    prefix=a.forward(torch.arange(1,18)[None]).state
    before,result=captured(a,torch.tensor([[18]]),prefix)
    # Offline future text and logits may change; same runtime prefix/query cannot.
    a.forward(torch.tensor([[19,20]]),state=result.state)
    after,_=captured(a,torch.tensor([[18]]),prefix)
    for first,second in zip(before.calls,after.calls):
        torch.testing.assert_close(first['features'].values,second['features'].values,atol=0,rtol=0)


def test_locality_respects_original_gaps(random_adapter):
    a=random_adapter
    prefix=a.forward(torch.arange(1,26)[None]).state
    capture,full=captured(a,torch.tensor([[26]]),prefix)
    layer=full.state.layers[0]
    feature_indices=torch.tensor([0,2,3])
    indices=torch.nonzero(torch.isin(layer.slots//8,layer.features.ids[feature_indices])).flatten()
    batch=extract(layer.slots[indices],layer.history[:,indices],layer.features.subset(feature_indices),
                  capture.calls[0]['query'],layer=0,layer_count=2,now=25,page_size=8)
    windows=batch.values[:,24:39].reshape(-1,3,5)
    assert batch.page_ids==(0,2,3)
    assert windows[0,2,4]==0  # page 2 is not page 0's immediate neighbor
    assert windows[1,0,4]==0  # missing original page 1
    assert windows[1,2,4]==1  # original page 3 really is adjacent


def test_label_modules_do_not_enter_runtime_import_graph():
    import htis_context.runtime_features as feature_module
    import htis_context.learned as learned_module
    for module in [feature_module,learned_module]:
        tree=ast.parse(Path(module.__file__).read_text())
        imports=[node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        assert not any('offline_labels' in name or 'counterfactual_data' in name or 'training' in name for name in imports)


@pytest.mark.parametrize('kind',['linear','mlp','htis','no_attention','no_locality','no_content','no_query','no_value'])
def test_trainable_models_have_real_gradient_updates(kind):
    torch.manual_seed(4)
    model=RiskEstimator(kind,12)
    inputs=torch.randn(12,WIDTH)
    target=torch.rand(12)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    loss=(model(inputs)-target).square().mean();loss.backward();optimizer.step()
    assert any(not torch.equal(initial[n],p) for n,p in model.named_parameters())


def test_no_query_ablation_does_not_read_current_query():
    model=RiskEstimator('no_query',12).eval()
    first=torch.randn(5,WIDTH); second=first.clone();second[:,17:24]=1000
    torch.testing.assert_close(model(first),model(second),atol=0,rtol=0)


def test_no_value_ablation_removes_value_projections_and_locality_values():
    model=RiskEstimator('no_value',12).eval()
    first=torch.randn(5,WIDTH); second=first.clone()
    second[:,6:10]=1000;second[:,12]=1000;second[:,14:17]=1000
    second[:,24:39].reshape(-1,3,5)[:,:,:4]=1000
    torch.testing.assert_close(model(first),model(second),atol=0,rtol=0)


def test_learned_policy_respects_page_protection_budget_and_exploration(random_adapter):
    a=random_adapter
    prefix=a.forward(torch.arange(1,34)[None]).state
    model=RiskEstimator('linear')
    policy=TrainedPagePolicy(model,torch.zeros(WIDTH),torch.ones(WIDTH),page_size=8,max_pages=3,
          layer_count=2,prompt_tokens=8,sink_tokens=8,recent_tokens=1,exploration_seed=17)
    run=LlamaReferenceAdapter(a.model,a.identity,policy=policy).forward(torch.tensor([[34]]),state=prefix)
    for layer in run.state.layers:
        assert len(layer.features.ids)<=3
        assert {0,4}<=set(layer.features.ids.tolist())
    assert policy.exploration_events
    assert policy.feature_bytes>0


def test_document_and_task_holdout_leakage_rejected():
    sessions=[{'document_id':'a','split':'train','task_family':'continue'},
              {'document_id':'b','split':'validation','task_family':'continue'},
              {'document_id':'c','split':'test','task_family':'compare'}]
    validate_splits(sessions,'compare')
    with pytest.raises(ValueError,match='Document'):
        validate_splits(sessions+[{'document_id':'a','split':'test','task_family':'continue'}],'compare')
    with pytest.raises(ValueError,match='Held-out'):
        validate_splits(sessions+[{'document_id':'d','split':'train','task_family':'compare'}],'compare')


@pytest.mark.pretrained
def test_pretrained_capture_and_physical_intervention():
    if os.environ.get('HTIS_RUN_PRETRAINED')!='1':
        pytest.skip('Opt-in real pinned checkpoint validation; synthetic fixtures do not replace it')
    from htis_context.checkpoint import load
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/smollm2-135m.json').read_text())
    model_dir=root.parent/'.model-cache/smollm2-135m'
    if not (model_dir/'htis-checkpoint.json').exists():
        pytest.skip('Pinned pretrained checkpoint missing')
    model,tokenizer,adapter,_=load(config,model_dir)
    ids=tokenizer('A library stores books and documents. Its reading room is open to visitors. Readers return books at the desk. The library closes at six.',return_tensors='pt').input_ids
    prefix=adapter.forward(ids[:,:-1]).state
    capture,full=captured(adapter,ids[:,-1:],prefix)
    with torch.no_grad():
        expected=model(ids,use_cache=False).logits[:,-1:]
    torch.testing.assert_close(full.logits,expected,atol=config['atol'],rtol=config['rtol'])
    intervention=InterventionPolicy(8,14,(1,),len(full.state.layers[14].features.ids))
    changed=LlamaReferenceAdapter(model,adapter.identity,policy=intervention).forward(ids[:,-1:],state=prefix)
    assert 1 not in changed.state.layers[14].features.ids
    assert kl(full.logits,changed.logits)>0


@pytest.fixture
def real_phase3_artifacts():
    if os.environ.get('HTIS_RUN_PHASE3_ARTIFACTS')!='1':
        pytest.skip('Set HTIS_RUN_PHASE3_ARTIFACTS=1 for recorded real trace/trained-weight validation')
    root=Path(__file__).resolve().parents[1]/'results/phase3'
    if not (root/'trained-v1/report.json').exists():
        pytest.skip('Actual Phase 3 dataset and trained artifacts are unavailable')
    manifest,singles,labels=read_dataset(root/'dataset-v1')
    training=json.loads((root/'trained-v1/report.json').read_text())
    return root,manifest,singles,labels,training


def test_actual_trained_artifacts_reload_and_reproduce_validation_metrics(real_phase3_artifacts):
    root,manifest,singles,labels,training=real_phase3_artifacts
    train=[r for r in singles if r['split']=='train']
    validation=[r for r in singles if r['split']=='validation']
    raw=torch.tensor([r['values'] for r in train])
    expected_mean=raw.mean(0)
    expected_std=raw.std(0,unbiased=False).clamp(min=1e-4)
    identity=ExecutionIdentity(**manifest['execution_identity'])
    for entry in training['models']:
        model,mean,std,record=load_estimator(root/'trained-v1'/entry['directory'],identity)
        torch.testing.assert_close(mean,expected_mean,atol=0,rtol=0)
        torch.testing.assert_close(std,expected_std,atol=0,rtol=0)
        inputs=(torch.tensor([r['values'] for r in validation])-mean)/std
        target=torch.tensor([__import__('math').log1p(1000*r['label']['horizon_kl']) for r in validation])
        with torch.no_grad():
            mse=float((model(inputs)-target).square().mean())
        assert mse==pytest.approx(record['validation']['log_target_mse'],abs=1e-6)


def test_actual_learned_artifacts_reject_wrong_model_scope_and_weight_corruption(real_phase3_artifacts,tmp_path):
    import shutil
    root,manifest,_,_,training=real_phase3_artifacts
    identity=ExecutionIdentity(**manifest['execution_identity'])
    source=root/'trained-v1'/training['models'][0]['directory']
    with pytest.raises(ValueError,match='provenance'):
        load_estimator(source,replace(identity,access_scope='other-tenant'))
    with pytest.raises(ValueError,match='provenance'):
        load_estimator(source,replace(identity,revision='0'*40))
    copied=tmp_path/'model'
    shutil.copytree(source,copied)
    path=copied/'weights.safetensors'
    path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='checksum'):
        load_estimator(copied,identity)


def test_actual_dataset_checksum_and_splits(real_phase3_artifacts,tmp_path):
    import shutil
    root,manifest,singles,labels,training=real_phase3_artifacts
    sets={name:set(training['split_documents'][name]) for name in ['train','validation','test']}
    assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
    assert all(r['split']=='test' for r in singles if r['task_family']=='compare')
    assert sum(len(r['omitted_pages'])==2 for r in labels)==93
    copied=tmp_path/'dataset'
    shutil.copytree(root/'dataset-v1',copied)
    path=copied/manifest['features_file']
    path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='checksum'):
        read_dataset(copied)
