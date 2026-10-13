"""Phase 4 CPU pretrained comparisons using the existing adapter and checkpoints."""
from __future__ import annotations
import argparse
from dataclasses import replace, asdict
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import time

import torch
from .adapter import LlamaReferenceAdapter
from .checkpoint import load
from .learned import load_estimator, TrainedPagePolicy
from .offline_labels import kl
from .pages import BaselinePolicy, MemoryBudget, update_features
from .prefix_cache import ExactPrefixCache
from .recovery import RecoveryPolicy, RecoveryConfig
from .records import write_report
from .state import _fingerprint


ARMS = ['recent/no-recovery','recent/recovery','attention-history/recovery','htis/no-recovery',
        'htis/recovery','deterministic/recovery','recent/bound-adaptive','recent/bound-fallback',
        'recent/expensive-access','recent/unavailable-archive','htis/always-refresh']


def perturb_value_page(prefix,layer,page,scale,page_size):
    """Explicit artificial supplied-KV stress; never changes pretrained weights."""
    with torch.inference_mode():
        x=prefix.layers[layer];idx=x.slots//page_size==page
        x.values[:,:,idx]*=scale
        x.features=update_features(x.keys,x.values,x.slots,page_size,None,x.slots)
    prefix.lineage='approximate:controlled-pretrained-value-outlier'


def base_policy(name,config,identity,layers,training):
    common=dict(page_size=config['page_size'],max_pages=config['active_pages'],prompt_tokens=config['prompt_tokens'],
                sink_tokens=config['sink_tokens'],recent_tokens=config['recent_tokens'])
    if name.startswith('htis'):
        estimator,mean,std,record=load_estimator(training/config['trained_directory'],identity)
        return TrainedPagePolicy(estimator,mean,std,layer_count=layers,**common),record
    return BaselinePolicy('attention_history' if name.startswith('attention-history') else 'recent',**common),{}


def make_policy(name,config,identity,layers,training):
    base,record=base_policy(name,config,identity,layers,training)
    if name.endswith('no-recovery'):
        return base
    cfg=RecoveryConfig(**config['recovery'])
    if name.startswith('deterministic'):cfg=replace(cfg,deterministic=True)
    if 'bound' in name:cfg=replace(cfg,tolerance=config['bound_tolerance'],fallback=name.endswith('fallback'))
    if name.endswith('expensive-access'):cfg=replace(cfg,transfer_ms_per_mb=config['expensive_access_ms_per_mb'])
    if name.endswith('unavailable-archive'):cfg=replace(cfg,max_archive_bytes=0,tolerance=config['bound_tolerance'],fallback=True)
    if name.endswith('always-refresh'):cfg=replace(cfg,interval=1,max_route_ms=1e6,cooldown=0)
    return RecoveryPolicy(base,cfg,calibration=record)


@torch.no_grad()
def evaluate(config,model_config,model_dir,dataset,training,output):
    if output.exists(): raise FileExistsError('Never overwrite a Phase 4 evaluation')
    output.mkdir(parents=True)
    began=time.perf_counter()
    model,tokenizer,dense,manifest=load(model_config,model_dir)
    data=json.loads((dataset/'manifest.json').read_text())
    traces={x['id']:x for x in data['traces'] if x['split']=='test'}
    trials=[];checks=[];workloads=[]
    prefix_cache=ExactPrefixCache(32*1024*1024)
    kv_limit=config['active_pages']*config['page_size']*model_config['layers']*model_config['kv_heads']*model_config['head_dim']*8
    for number,workload in enumerate(config['workloads']):
        trace=traces[workload['trace']]
        ids=torch.tensor(trace['prefix_recipe']['tokens'])
        pos=torch.tensor(trace['prefix_recipe']['positions'])
        mask=torch.tensor(trace['prefix_recipe']['mask'])
        prefix=dense.forward(ids,position_ids=pos,attention_mask=mask).state
        prefix_reuse=None
        if workload['kind']=='repeated-exact-prefix':
            prefix_cache.put(prefix)
            start=time.perf_counter();restored=prefix_cache.get(dense.identity,prefix.recipe)
            prefix_reuse={'hit':restored is not None,'restore_ms':(time.perf_counter()-start)*1000,
                         'kv_bytes':restored.kv_bytes,'identical_state':_fingerprint(restored)==_fingerprint(prefix)}
            prefix=restored
        perturbation=None
        if workload.get('value_scale'):
            layer=14;page=1
            # Reapply the actual supplied-state perturbation from Phase 3.
            perturb_value_page(prefix,layer,page,workload['value_scale'],config['page_size'])
            perturbation={'layer':layer,'page':page,'scale':workload['value_scale'],
                          'natural_model_state':False,'reference':'same supplied perturbed KV snapshot'}
        tokens=[trace['current_token_id'],trace['future_token_ids'][0]]+tokenizer(workload['suffix'],add_special_tokens=False).input_ids
        tokens=tokens[:config['steps']]
        workloads.append({**workload,'document_id':trace['document_id'],'source_sha256':trace['source_sha256'],
            'tokens':tokens,'prefix_recipe':trace['prefix_recipe'],'prefix_state_sha256':_fingerprint(prefix),
            'prefix_reuse':prefix_reuse,'perturbation':perturbation})
        teacher=[];teacher_state=prefix
        for token in tokens:
            run=dense.forward(torch.tensor([[token]]),state=teacher_state)
            teacher.append(run.logits[:,-1]);teacher_state=run.state
        for repeat in range(config['timing_repeats']):
            for name in ARMS:
                policy=make_policy(name,config,dense.identity,model_config['layers'],training)
                adapter=LlamaReferenceAdapter(model,dense.identity,policy=policy,
                    memory_budget=MemoryBudget(kv_limit,config['recovery']['max_working_bytes']))
                state=prefix;observations=[];begin=time.perf_counter()
                for step,token in enumerate(tokens):
                    if hasattr(policy,'recovery'):
                        policy.recovery.task=trace['task_family'] if step<2 else workload['kind']
                    run=adapter.forward(torch.tensor([[token]]),state=state);state=run.state
                    recovery=run.metrics.get('recovery',{})
                    persistent=state.kv_bytes+run.metrics['metadata_tensor_bytes']+recovery.get('archive_storage_bytes',0)+recovery.get('index_estimate_bytes',0)
                    scorer=policy.base if isinstance(policy,RecoveryPolicy) else policy
                    scorer_bytes=(sum(x.numel()*x.element_size() for x in scorer.estimator.parameters())
                        +scorer.mean.numel()*scorer.mean.element_size()+scorer.std.numel()*scorer.std.element_size()) if hasattr(scorer,'estimator') else 0
                    observation={'step':step,'token':token,'kl_to_dense':kl(teacher[step],run.logits[:,-1]),
                        'forward_ms':run.metrics['elapsed_ms'],'selection_ms':run.metrics['selection_ms'],
                        'active_kv_bytes':state.kv_bytes,'active_metadata_bytes':run.metrics['metadata_tensor_bytes'],
                        'total_cache_index_scorer_storage_estimate_bytes':persistent+scorer_bytes,'scorer_persistent_bytes':scorer_bytes,
                        'base_attention_working_estimate_bytes':run.metrics['estimated_cache_and_attention_working_tensor_bytes'],
                        'active_pages_per_layer':[x.features.ids.tolist() for x in state.layers],
                        'recovery':recovery,'lineage':state.lineage}
                    observations.append(observation)
                    budget_ok=state.kv_bytes<=kv_limit and recovery.get('archive_storage_bytes',0)<=config['recovery']['max_archive_bytes']
                    coverage_ok=all(b.get('bound_covers_measured_error',True) for b in recovery.get('bounds',[]))
                    checks.append({'name':f'{number}/{name}/repeat-{repeat}/step-{step}','passed':budget_ok and coverage_ok})
                trials.append({'workload':workload['kind'],'trace':trace['id'],'policy':name,'repeat':repeat,
                    'steps':observations,'elapsed_ms':(time.perf_counter()-begin)*1000,
                    'mean_KL_to_dense':sum(x['kl_to_dense'] for x in observations)/len(observations)})
                # Durable completed-trial checkpoint, even if a later case fails.
                with gzip.open(output/'partial-trials.jsonl.gz','at',encoding='utf-8') as stream:
                    stream.write(json.dumps(trials[-1],separators=(',',':'),allow_nan=False)+'\n')
                (output/'progress.json').write_text(json.dumps({'completed_trials':len(trials),'completed_steps':len(checks),
                    'workload_index':number,'workload':workload['kind'],'policy':name,'repeat':repeat,
                    'elapsed_seconds':time.perf_counter()-began,'configuration':config},indent=2)+'\n')
        (output/'partial-workloads.json').write_text(json.dumps(workloads,indent=2)+'\n')
        print(f'Phase 4 workload {number+1}/{len(config["workloads"])} complete: {workload["kind"]}',flush=True)
    aggregate=[]
    for name in ARMS:
        rows=[r for r in trials if r['policy']==name];steps=[s for r in rows for s in r['steps']]
        recoveries=[s['recovery'] for s in steps];events=[e for x in recoveries for e in x.get('events',[])]
        bounds=[b for x in recoveries for b in x.get('bounds',[])]
        aggregate.append({'policy':name,'trials':len(rows),'steps':len(steps),
            'mean_KL_to_dense':sum(x['kl_to_dense'] for x in steps)/len(steps),
            'mean_forward_ms':sum(x['forward_ms'] for x in steps)/len(steps),
            'mean_controller_ms':sum(x.get('controller_ms',0) for x in recoveries)/len(steps),
            'mean_scorer_ms':sum(x.get('scorer_ms',0) for x in recoveries)/len(steps),
            'mean_integrity_ms':sum(x.get('integrity_ms',0) for x in recoveries)/len(steps),
            'mean_bound_ms':sum(x.get('bound_ms',0) for x in recoveries)/len(steps),
            'mean_fallback_ms':sum(x.get('fallback_ms',0) for x in recoveries)/len(steps),
            'max_active_kv_bytes':max(x['active_kv_bytes'] for x in steps),
            'max_archive_storage_bytes':max([x.get('archive_storage_bytes',0) for x in recoveries],default=0),
            'max_index_estimate_bytes':max([x.get('index_estimate_bytes',0) for x in recoveries],default=0),
            'max_total_storage_estimate_bytes':max(x['total_cache_index_scorer_storage_estimate_bytes'] for x in steps),
            'max_combined_known_working_estimate_bytes':max([x.get('combined_known_working_estimate_bytes',0) for x in recoveries],default=0),
            'reloads':sum(e['action']=='reload' for e in events),'offloads':sum(e['action']=='offload' for e in events),
            'transferred_copy_bytes':sum(e.get('copy_bytes',0) for e in events),
            'reload_ms':sum(e.get('ms',0) for e in events if e['action']=='reload'),
            'offload_ms':sum(e.get('ms',0) for e in events if e['action']=='offload'),
            'fallback_count':sum(b.get('exact_current_state',False) for b in bounds),
            'tolerance_unmet':sum(not b['tolerance_met'] for b in bounds),
            'coverage_failures':sum(not b['coverage'] for b in bounds),
            'bound_validation_failures':sum(not b.get('bound_covers_measured_error',True) for b in bounds)})
    comparisons=[]
    for kind in [w['kind'] for w in config['workloads']]:
        means={name:sum(r['mean_KL_to_dense'] for r in trials if r['workload']==kind and r['policy']==name)/config['timing_repeats'] for name in ARMS}
        comparisons.append({'workload':kind,'mean_kl':means,
            'recency_recovery_delta':means['recent/recovery']-means['recent/no-recovery'],
            'htis_recovery_delta':means['htis/recovery']-means['htis/no-recovery'],
            'htis_vs_recency_recovery_delta':means['htis/recovery']-means['recent/recovery']})
    report={'experiment':config['experiment'],'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'configuration':config,'execution_identity':asdict(dense.identity),'checkpoint_files_sha256':manifest['files_sha256'],
        'phase3_dataset_manifest_sha256':hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
        'workloads':workloads,'aggregate':aggregate,'paired_comparisons':comparisons,'checks':checks,'passed':all(x['passed'] for x in checks),
        'trials':trials,'wall_seconds':time.perf_counter()-began,'prefix_cache_events':prefix_cache.events,
        'limitations':['CPU FP32 eager reference; GPU transfers/savings, optimized kernels, RSS/allocator peaks unmeasured.',
            'Small held-out teacher-forced study, not validated answers, benchmark accuracy, or significance.',
            'Archive FIFO; shared subset per KV group/layer; complete cold prefill remains dense.',
            'Local bound applies only to complete supplied-state coverage; fallback is not historical dense repair.',
            'Expensive-access arm uses a declared conservative cost prior, not artificially reported measured copy latency.',
            'No recomputation support; discarded pages retain summaries but cannot be restored.',
            'Frozen Phase 3 weights/normalization/calibration; no training or held-out threshold tuning.']}
    write_report(report,output/'report.json')
    print(json.dumps({'passed':report['passed'],'checks':len(checks),'trials':len(trials),'wall_seconds':report['wall_seconds'],'aggregate':aggregate},indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('transformer-efficiency/configs/phase4-small.json'))
    p.add_argument('--model-config',type=Path,default=Path('transformer-efficiency/configs/smollm2-135m.json'))
    p.add_argument('--model-dir',type=Path,default=Path('.model-cache/smollm2-135m'))
    p.add_argument('--dataset',type=Path,default=Path('transformer-efficiency/results/phase3/dataset-v1'))
    p.add_argument('--training',type=Path,default=Path('transformer-efficiency/results/phase3/trained-v1'))
    p.add_argument('--output',type=Path,default=Path('transformer-efficiency/results/phase4/evaluation-v1'))
    a=p.parse_args()
    r=evaluate(json.loads(a.config.read_text()),json.loads(a.model_config.read_text()),a.model_dir,a.dataset,a.training,a.output)
    return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
