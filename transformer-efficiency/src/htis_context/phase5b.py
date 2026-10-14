"""Preregistered extension using unchanged adapter, prefix cache and frozen policies."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import torch

from .adapter import LlamaReferenceAdapter
from .checkpoint import load
from .learned import load_estimator, TrainedPagePolicy
from .offline_labels import kl
from .pages import BaselinePolicy, MemoryBudget
from .phase5 import observation, append, upstream_sdpa
from .phase5_data import sha, write
from .phase5_metrics import answer_metrics
from .phase5b_data import DATA, ROOT, prepare
from .prefix_cache import ExactPrefixCache
from .provenance import SourceRecipe
from .recovery import RecoveryConfig, RecoveryPolicy
from .state import _fingerprint

RESULT = ROOT/'results/phase5b'
MANIFEST = ROOT/'configs/phase5b-cpu-v1.json'
MODEL = ROOT/'configs/smollm2-135m.json'
TRAIN = ROOT/'results/phase3/trained-v1'
POLICIES = ['recent', 'query', 'random', 'htis']


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def runtime_hashes():
    paths = [*ROOT.glob('src/htis_context/*.py'), *ROOT.glob('src/htis/*.py'),
             *ROOT.glob('tests/*.py'), MODEL, Path(__file__)]
    return {p.as_posix(): sha(p) for p in sorted(set(paths))}


def verify_frozen(manifest):
    for name, digest in manifest['runtime_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Frozen runtime changed: '+name)
    if sha(DATA) != manifest['data_sha256']:
        raise ValueError('Frozen data changed')
    for name, digest in manifest['checkpoints_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Frozen checkpoint changed: '+name)


def budget_bytes(model_config, pages, page_size=8):
    return pages*page_size*model_config['layers']*model_config['kv_heads']*model_config['head_dim']*2*4


def freeze():
    if MANIFEST.exists():
        raise FileExistsError('Protocol/configuration is immutable')
    data, mc = read(DATA), read(MODEL)
    cfg = dict(page_size=8, budgets=[4,8,16], prompt_tokens=8, sink_tokens=8,
        recent_tokens=8, seeds=[17,29], timing_repeats=2, warmup_steps=2,
        batch=1, concurrency=1, precision='float32', greedy=True,
        eos_stopping=True, answer_stopping=False, fixed_generation_by_case=True,
        archive_bytes=128*1024**2, index_bytes=64*1024**2, working_bytes=512*1024**2,
        prefix_bytes=192*1024**2, bootstrap_repetitions=10000, bootstrap_seed=1729,
        recovery_candidates=[dict(interval=4,cooldown=2,query_change=.35,risk_trigger=.5,
            uncertainty_trigger=.5,ranking_trigger=.5,max_route_ms=100000.,score_ms_prior=.1),
            dict(interval=8,cooldown=2,query_change=.35,risk_trigger=.5,
            uncertainty_trigger=.5,ranking_trigger=.5,max_route_ms=100000.,score_ms_prior=.1)])
    cases = [r for r in data['rows'] if r['split']=='test']
    systems = [cases[0]['id'], 'session5b/revisit']
    feasible = []
    for pages in cfg['budgets']:
        # Eight protected opening slots plus the last eight can span three pages.
        minimum = 1 + (cfg['recent_tokens']+cfg['page_size']-2)//cfg['page_size']+1
        feasible.append(dict(pages=pages, required_protected_pages_worst_case=minimum,
            feasible=pages>=minimum, physical_kv_ceiling_bytes=budget_bytes(mc,pages)))
    checkpoints = [*TRAIN.glob('htis-seed-*/*'), Path('.model-cache/smollm2-135m/htis-checkpoint.json'),
                   Path('.model-cache/smollm2-135m/model.safetensors')]
    checkpoints = [p for p in checkpoints if p.is_file()]
    manifest = dict(experiment='phase5b-extended-transformer-efficiency-v1',
        frozen_at_utc=stamp(), configuration=cfg, policies=POLICIES,
        data_sha256=sha(DATA), model_configuration_sha256=sha(MODEL),
        runtime_sha256=runtime_hashes(), checkpoints_sha256={p.as_posix():sha(p) for p in checkpoints},
        protocol_sha256=sha(RESULT/'PREREGISTRATION.md'),
        quality_cases=[r['id'] for r in cases], systems_cases=systems,
        validation_cases=[r['id'] for r in data['rows'] if r['split']=='validation'],
        feasibility=feasible, frozen_before_validation_and_final_evaluation=True,
        quality_matrix='Ten test cases x 4/8/16 pages x four bounded policies x recovery OFF/ON x seeds17/29; full dense once per case (490 trials)',
        systems_matrix='Two preselected cases x all bounded quality cells x prefix OFF/ON x two repeats. Repeat0 seed17, repeat1 seed29. Full reference x recovery no-op OFF/ON x prefix OFF/ON x two repeats; budget N/A.',
        analysis='Equal case means; seeds averaged per case; paired cluster bootstrap. Report workload/case and budget, means/medians/seed variability and every regression. Three simple comparisons times three budgets: Bonferroni 99.444% intervals for primary A; nine clusters cannot establish broad generalization.',
        memory='Measured physical tensors plus separately labeled metadata/index/recipe byte estimates, prefix bytes, scorer tensors, known transient buffers, and process-wide OS working set. Dense prefill is NOT budget bounded. No cache-system allocator peak claim.',
        gpu='Run existing availability gate; do not provision. CPU FP32 recovery is not GPU validated.')
    write(MANIFEST,manifest)
    write(RESULT/'frozen-manifest.json',dict(manifest_sha256=sha(MANIFEST),frozen_at_utc=manifest['frozen_at_utc'],manifest=manifest))
    return {'manifest_sha256':sha(MANIFEST),'feasibility':feasible,'quality_cases':len(cases)}


def make_adapter(model,dense,name,pages,seed,config,recovery_config=None):
    common=dict(page_size=config['page_size'],max_pages=pages,prompt_tokens=config['prompt_tokens'],
        sink_tokens=config['sink_tokens'],recent_tokens=config['recent_tokens'])
    record={}
    if name=='htis':
        estimator,mean,std,record=load_estimator(TRAIN/f'htis-seed-{seed}',dense.identity)
        p=TrainedPagePolicy(estimator,mean,std,layer_count=len(model.model.layers),**common)
    else:
        p=BaselinePolicy({'random':'prompt_random','query':'query_summary'}.get(name,name),seed=seed,**common)
    # Full cache never has evicted pages; recovery ON is a preregistered identity control.
    if recovery_config is not None and name!='full':
        p=RecoveryPolicy(p,RecoveryConfig(**recovery_config),calibration=record)
    maximum=pages*config['page_size']*len(model.model.layers)*model.config.num_key_value_heads*model.model.layers[0].self_attn.head_dim*8
    return LlamaReferenceAdapter(model,dense.identity,policy=p,
        memory_budget=None if name=='full' else MemoryBudget(maximum,config['working_bytes']))


def recovery_candidate(config,candidate):
    return dict(candidate,max_archive_bytes=config['archive_bytes'],max_index_bytes=config['index_bytes'],
        max_working_bytes=config['working_bytes'])


def process_memory():
    """Windows OS counters; cumulative process peak includes research fixtures."""
    if os.name!='nt':
        return {'scope':'OS process counters unavailable on this host'}
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
            ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    get=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
    get.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not get(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(working_set_bytes=counters.WorkingSetSize,process_lifetime_peak_working_set_bytes=counters.PeakWorkingSetSize,
        scope='OS process working set, includes model/fixtures/Python/allocator; lifetime peak is not per-request cache memory')


def observe(run,policy):
    row=observation(run,policy)
    row['source_recipe_estimate_bytes']=run.metrics['source_recipe_json_bytes']
    row['archive_kv_bytes']=row['recovery'].get('archive_kv_bytes',0)
    row['known_cache_system_storage_bytes']=row['total_storage_bytes']+row['source_recipe_estimate_bytes']
    row['input_state_kv_bytes']=run.metrics['input_state_kv_bytes']
    row['cache_transition_estimate_bytes']=run.metrics['peak_cache_transition_tensor_bytes']
    row['attention_workspace_estimate_bytes']=run.metrics['peak_attention_workspace_tensor_bytes']
    row['copy_event_ms']=sum(e.get('ms',0) for e in row['recovery'].get('events',[]))
    return row


@torch.no_grad()
def generate(row,adapter,prepared,tokenizer,*,capture_logits=False):
    state=prepared; token=row['suffix_ids'][-1]; outputs=[]; steps=[]; logits=[]; first_ms=None
    began=time.perf_counter()
    observation_ms=0.
    for i in range(row['generation_tokens']):
        run=adapter.forward(torch.tensor([[token]]),state=state); state=run.state
        token=int(run.logits[0,-1].argmax());outputs.append(token)
        if capture_logits:logits.append(run.logits[0,-1].detach().clone())
        if i==0:first_ms=(time.perf_counter()-began)*1000
        start=time.perf_counter();steps.append(observe(run,adapter.policy));observation_ms+=(time.perf_counter()-start)*1000
        if token==tokenizer.eos_token_id:break
    elapsed=(time.perf_counter()-began)*1000
    text=tokenizer.decode(outputs,skip_special_tokens=True)
    return dict(output_ids=outputs,raw_output=text,**answer_metrics(text,row['answers']),
        output_tokens=len(outputs),generation_limit=row['generation_tokens'],
        completion='eos' if outputs[-1]==tokenizer.eos_token_id else 'limit',
        free_steps=steps,decode_wall_ms=elapsed,first_decode_wall_ms=first_ms,
        observation_overhead_ms=observation_ms,logit_capture_bytes=sum(t.numel()*t.element_size() for t in logits)),logits


@torch.no_grad()
def quality(row,name,pages,seed,prepared,teacher,model,dense,tokenizer,config,rec=None):
    begin=time.perf_counter();adapter=make_adapter(model,dense,name,pages,seed,config,rec)
    setup=(time.perf_counter()-begin)*1000
    state=prepared; nll=[]; divergence=[]; observations=[]
    for i,token in enumerate([row['suffix_ids'][-1]]+row['answer_ids'][:-1]):
        run=adapter.forward(torch.tensor([[token]]),state=state);state=run.state
        logits=run.logits[:,-1];divergence.append(kl(teacher[i],logits))
        nll.append(float(-torch.log_softmax(logits.float(),-1)[0,row['answer_ids'][i]]))
        observations.append(observe(run,adapter.policy))
    adapter=make_adapter(model,dense,name,pages,seed,config,rec)
    result,_=generate(row,adapter,prepared,tokenizer)
    result.update(case=row['id'],group=row['group'],kind=row['kind'],split=row['split'],
        policy=name,budget=None if name=='full' else pages,seed=seed,recovery=rec is not None,
        input_tokens=row['input_tokens'],answers=row['answers'],mean_nll=statistics.mean(nll),mean_kl=statistics.mean(divergence),
        nll_per_teacher_token=nll,kl_per_teacher_token=divergence,teacher_steps=observations,setup_ms=setup,
        mean_decode_ms=statistics.mean(x['forward_ms'] for x in result['free_steps']))
    return result


def cell_key(row):
    return (row['case'],row['policy'],row['budget'],row['seed'],row['recovery'])


def validate_selection(trials,candidates):
    if not trials or any(r['split']!='validation' for r in trials):
        raise ValueError('Only new validation sessions may select thresholds')
    table=[]
    for candidate in range(len(candidates)):
        rows=[r for r in trials if r.get('candidate')==candidate]
        pairs={(r['case'],r['policy'],r['seed']) for r in rows}
        if any(pairs!={(r['case'],r['policy'],r['seed']) for r in trials if r.get('candidate')==i} for i in range(len(candidates))):
            raise ValueError('Recovery candidates must have matched validation coverage')
        table.append(dict(candidate=candidate,mean_f1=statistics.mean(r['f1'] for r in rows),
            mean_nll=statistics.mean(r['mean_nll'] for r in rows),mean_decode_ms=statistics.mean(r['mean_decode_ms'] for r in rows)))
    best=min(table,key=lambda r:(-r['mean_f1'],r['mean_nll'],r['mean_decode_ms'],r['candidate']))
    return dict(selected_candidate=best['candidate'],table=table,
        rule='Highest validation answer F1, then lowest NLL, then lowest decode latency, then candidate index. No held-out labels.',
        validation_case_ids=sorted({r['case'] for r in trials}))


def load_run():
    manifest=read(MANIFEST);verify_frozen(manifest)
    model,tokenizer,dense,checkpoint=load(read(MODEL),Path('.model-cache/smollm2-135m'))
    for _ in range(manifest['configuration']['warmup_steps']):
        dense.forward(torch.tensor([[1,2,3,4,5,6,7,8]]))
    return manifest,model,tokenizer,dense,checkpoint


@torch.no_grad()
def evaluate(mode,output):
    output.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter();manifest,model,tok,dense,checkpoint=load_run();cfg=manifest['configuration']
    data=read(DATA);rows=[r for r in data['rows'] if r['split']==('validation' if mode=='validation' else 'test')]
    calibration=None if mode=='validation' else read(RESULT/'validation-v1/calibration.json')
    if calibration is not None:
        if calibration['manifest_sha256']!=sha(MANIFEST):raise ValueError('Calibration mismatch')
        rec=calibration['recovery_configuration']
    trials=[];checks=[];prefills=[];sdpa=[]
    for number,row in enumerate(rows):
        print('Preparing '+row['id'],flush=True)
        start=time.perf_counter();prefix=dense.forward(torch.tensor([row['prefix_ids']]))
        prefix_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter();prepared=dense.forward(torch.tensor([row['suffix_ids'][:-1]]),state=prefix.state)
        suffix_ms=(time.perf_counter()-start)*1000
        prefills.append(dict(case=row['id'],prefix_ms=prefix_ms,suffix_ms=suffix_ms,
            dense_prefill_kv_bytes=prepared.state.kv_bytes,process_memory=process_memory(),
            dense_prefill_tensor_working_estimate=prefix.metrics['estimated_cache_and_attention_working_tensor_bytes']))
        teacher=[];state=prepared.state
        for token in [row['suffix_ids'][-1]]+row['answer_ids'][:-1]:
            run=dense.forward(torch.tensor([[token]]),state=state);state=run.state;teacher.append(run.logits[:,-1].clone())
        before=_fingerprint(prepared.state)
        full=quality(row,'full',4,17,prepared.state,teacher,model,dense,tok,cfg)
        trials.append(full);append(output/'trials.jsonl.gz',full)
        if mode=='test':
            # Complete untruncated prompt and common-token numerical comparison.
            upstream=upstream_sdpa(model,row['prefix_ids']+row['suffix_ids'],full['output_ids'],
                atol=read(MODEL)['atol'],rtol=read(MODEL)['rtol'],dense=dense)
            upstream['case']=row['id'];sdpa.append(upstream)
        matrix=[]
        if mode=='validation':
            for name in POLICIES:
                matrix.append((name,8,17,None,None))
                for i,candidate in enumerate(cfg['recovery_candidates']):
                    matrix.append((name,8,17,recovery_candidate(cfg,candidate),i))
        else:
            matrix=[(name,budget,seed,rec if recovery else None,None) for budget in cfg['budgets']
                    for name in POLICIES for recovery in [False,True] for seed in cfg['seeds']]
        random.Random(1729+number).shuffle(matrix)
        for name,budget,seed,recovery_config,candidate in matrix:
            trial=quality(row,name,budget,seed,prepared.state,teacher,model,dense,tok,cfg,recovery_config)
            if candidate is not None:trial['candidate']=candidate
            trials.append(trial);append(output/'trials.jsonl.gz',trial)
            ceiling=budget_bytes(read(MODEL),budget)
            observations=trial['teacher_steps']+trial['free_steps']
            passed=all(x['active_kv_bytes']<=ceiling and all(len(ids)<=budget for ids in x['active_pages']) for x in observations)
            checks.append(dict(case=row['id'],policy=name,budget=budget,seed=seed,recovery=recovery_config is not None,passed=passed))
            write(output/'progress.json',dict(completed_trials=len(trials),case=row['id'],last_cell=checks[-1],at_utc=stamp()))
            if not passed:raise AssertionError('Physical budget failed')
            print(f'{mode} {len(trials)} {row["id"]} {name} {budget} {seed} recovery={recovery_config is not None}',flush=True)
        checks.append(dict(case=row['id'],kind='dense_fixture_unchanged',passed=before==_fingerprint(prepared.state)))
        if not checks[-1]['passed']:raise AssertionError('Shared dense prefill mutated')
    if mode=='validation':
        selection=validate_selection(trials,cfg['recovery_candidates'])
        selection.update(manifest_sha256=sha(MANIFEST),completed_at_utc=stamp(),
            recovery_configuration=recovery_candidate(cfg,cfg['recovery_candidates'][selection['selected_candidate']]))
        write(output/'calibration.json',selection)
    verify_frozen(manifest)
    report=dict(mode=mode,trials=len(trials),checks=checks,prefills=prefills,upstream_sdpa=sdpa,
        manifest_sha256=sha(MANIFEST),raw_sha256=sha(output/'trials.jsonl.gz'),
        execution_identity=asdict(dense.identity),checkpoint=checkpoint,wall_seconds=time.perf_counter()-began,
        completed_at_utc=stamp(),passed=all(c['passed'] for c in checks))
    write(output/'report.json',report)
    return {'trials':len(trials),'checks':len(checks),'passed':report['passed'],'wall_seconds':report['wall_seconds']}


@torch.no_grad()
def request(row,name,budget,seed,recovery_enabled,reuse,repeat,model,dense,tok,cfg,rec,cache):
    began=time.perf_counter();setup_start=began
    adapter=make_adapter(model,dense,name,budget,seed,cfg,rec if recovery_enabled else None)
    setup_ms=(time.perf_counter()-setup_start)*1000
    lookup_ms=prefill_ms=admission_ms=0.;hit=False
    recipe=SourceRecipe(**row['prefix_recipe']);state=None
    if reuse:
        start=time.perf_counter();state=cache.get(dense.identity,recipe);lookup_ms=(time.perf_counter()-start)*1000
        hit=state is not None
    if state is None:
        start=time.perf_counter();state=dense.forward(torch.tensor([row['prefix_ids']])).state
        prefill_ms=(time.perf_counter()-start)*1000
        if reuse:
            start=time.perf_counter();admitted=cache.put(state);admission_ms=(time.perf_counter()-start)*1000
            if not admitted:raise AssertionError('Prefix admission unexpectedly infeasible')
    prefill_kv=state.kv_bytes
    start=time.perf_counter();prepared=dense.forward(torch.tensor([row['suffix_ids'][:-1]]),state=state).state
    suffix_ms=(time.perf_counter()-start)*1000
    generated,logits=generate(row,adapter,prepared,tok,capture_logits=True)
    elapsed=(time.perf_counter()-began)*1000
    generated.update(case=row['id'],group=row['group'],kind=row['kind'],policy=name,budget=None if name=='full' else budget,
        seed=seed,recovery=recovery_enabled,recovery_identity_control=name=='full' and recovery_enabled,
        reuse=reuse,repeat=repeat,hit=hit,lookup_ms=lookup_ms,prefill_ms=prefill_ms,
        admission_ms=admission_ms,setup_ms=setup_ms,suffix_ms=suffix_ms,
        ttft_ms=setup_ms+lookup_ms+prefill_ms+admission_ms+suffix_ms+generated['first_decode_wall_ms'],
        end_to_end_ms=elapsed,tokens_per_second=generated['output_tokens']/(elapsed/1000),
        decode_tokens_per_second=generated['output_tokens']/(generated['decode_wall_ms']/1000),
        prefix_storage_bytes=cache.bytes_used if reuse else 0,dense_prefill_kv_bytes=prefill_kv,
        saved_prefix_tokens=len(row['prefix_ids']) if hit else 0,context_tokens=row['input_tokens'],
        process_memory=process_memory())
    return generated,logits


@torch.no_grad()
def systems(output):
    output.mkdir(parents=True,exist_ok=False);began=time.perf_counter()
    manifest,model,tok,dense,_=load_run();cfg=manifest['configuration']
    calibration=read(RESULT/'validation-v1/calibration.json')
    if calibration['manifest_sha256']!=sha(MANIFEST):raise ValueError('Calibration mismatch')
    rec=calibration['recovery_configuration'];results=[];equivalence=[];lifecycle=[]
    rows=[r for r in read(DATA)['rows'] if r['id'] in manifest['systems_cases']]
    for row in rows:
        cells=[(name,budget,recovery) for name in POLICIES for budget in cfg['budgets'] for recovery in [False,True]]
        cells += [('full',4,recovery) for recovery in [False,True]]
        random.Random(901+row['input_tokens']).shuffle(cells)
        for name,budget,recovery_enabled in cells:
            for repeat in range(cfg['timing_repeats']):
                seed=cfg['seeds'][repeat%len(cfg['seeds'])];cache=ExactPrefixCache(cfg['prefix_bytes'])
                # A real reuse-ON miss populates this cell's independent cache.
                # Stored logits are temporary diagnostics, charged and discarded.
                populate,pop_logits=request(row,name,budget,seed,recovery_enabled,True,repeat,model,dense,tok,cfg,rec,cache)
                populate['traffic']='cold-admission';results.append(populate);append(output/'systems.jsonl.gz',populate)
                pair=[False,True]
                if repeat%2:pair.reverse()
                current={}
                for reuse in pair:
                    trial,logits=request(row,name,budget,seed,recovery_enabled,reuse,repeat,model,dense,tok,cfg,rec,cache)
                    trial['traffic']='warm-repeat' if reuse else 'cold-no-cache'
                    current[reuse]=(trial,logits);results.append(trial);append(output/'systems.jsonl.gz',trial)
                cold,cold_logits=current[False];warm,warm_logits=current[True]
                passed=cold['output_ids']==warm['output_ids']==populate['output_ids'] and len(cold_logits)==len(warm_logits)==len(pop_logits)
                errors=[]
                for a,b,c in zip(cold_logits,warm_logits,pop_logits):
                    errors.append(max(float((a-b).abs().max()),float((a-c).abs().max())))
                    passed=passed and torch.allclose(a,b,atol=read(MODEL)['atol'],rtol=read(MODEL)['rtol']) and torch.allclose(a,c,atol=read(MODEL)['atol'],rtol=read(MODEL)['rtol'])
                check=dict(case=row['id'],policy=name,budget=budget if name!='full' else None,seed=seed,
                    recovery=recovery_enabled,repeat=repeat,passed=bool(passed),max_abs_logit_error=max(errors),
                    compared_steps=len(errors),saved_prefill_ms=cold['prefill_ms'],hit=warm['hit'])
                equivalence.append(check);append(output/'equivalence.jsonl.gz',check)
                if not passed or not warm['hit']:raise AssertionError('Prefix numerical equivalence/hit failure')
                # Same shape/positions but changed first token MUST miss; run an
                # actual modified-prefix request for full identity controls.
                modified=dict(row);modified['prefix_ids']=list(row['prefix_ids']);modified['prefix_ids'][0]=(modified['prefix_ids'][0]+1)%model.config.vocab_size
                modified['prefix_recipe']=dict(tokens=[modified['prefix_ids']],positions=[list(range(len(modified['prefix_ids'])))],mask=[[1]*len(modified['prefix_ids'])])
                start=time.perf_counter();miss=cache.get(dense.identity,SourceRecipe(**modified['prefix_recipe']))
                invalidate_ms=(time.perf_counter()-start)*1000
                if miss is not None:raise AssertionError('Modified prefix reused incompatible KV')
                lifecycle.append(dict(case=row['id'],policy=name,budget=budget,seed=seed,repeat=repeat,
                    modified_prefix_missed=True,invalidation_ms=invalidate_ms,events=cache.events))
                if name=='full' and not recovery_enabled:
                    changed,_=request(modified,name,budget,seed,False,True,repeat,model,dense,tok,cfg,rec,cache)
                    changed['traffic']='modified-prefix';changed['original_case']=row['id']
                    if changed['hit']:raise AssertionError('Modified prefix hit')
                    results.append(changed);append(output/'systems.jsonl.gz',changed)
                write(output/'progress.json',dict(requests=len(results),paired_cells=len(equivalence),at_utc=stamp()))
                print(f'systems requests={len(results)} pairs={len(equivalence)} {name} {budget} recovery={recovery_enabled}',flush=True)
    # Different current questions with an identical shared prefix; test actual
    # reuse between requests rather than only repeating one complete prompt.
    shared=[r for r in read(DATA)['rows'] if r['group']=='session5b/switch']
    cache=ExactPrefixCache(cfg['prefix_bytes']);shared_checks=[]
    for i,row in enumerate(shared):
        trial,logits=request(row,'query',8,17,False,True,0,model,dense,tok,cfg,rec,cache)
        trial['traffic']='shared-prefix-task-change';results.append(trial);append(output/'systems.jsonl.gz',trial)
        cold,cold_logits=request(row,'query',8,17,False,False,0,model,dense,tok,cfg,rec,cache)
        cold['traffic']='shared-prefix-cold-control';results.append(cold);append(output/'systems.jsonl.gz',cold)
        passed=trial['hit']==(i>0) and trial['output_ids']==cold['output_ids'] and all(torch.allclose(a,b,atol=read(MODEL)['atol'],rtol=read(MODEL)['rtol']) for a,b in zip(logits,cold_logits))
        shared_checks.append(dict(case=row['id'],hit=trial['hit'],passed=bool(passed)))
        if not passed:raise AssertionError('Shared-prefix task-change failed')
    verify_frozen(manifest)
    report=dict(requests=len(results),paired_cells=len(equivalence),equivalence=equivalence,
        lifecycle=lifecycle,shared_prefix_checks=shared_checks,manifest_sha256=sha(MANIFEST),
        raw_sha256=sha(output/'systems.jsonl.gz'),wall_seconds=time.perf_counter()-began,
        completed_at_utc=stamp(),passed=all(x['passed'] for x in equivalence+shared_checks),
        timing_notes='Wall timings include standard adapter instrumentation, controller construction and explicit logit/observation capture. Request wall time includes answer decoding/metric scoring; decode wall time ends before these. JSON serialization outside request timing. Diagnostic overhead measured, not subtracted. Admission requests separately charged.')
    write(output/'report.json',report)
    return {'requests':len(results),'paired_cells':len(equivalence),'passed':report['passed'],'wall_seconds':report['wall_seconds']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','freeze','validation','test','systems'])
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.mode=='prepare':
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained('.model-cache/smollm2-135m',local_files_only=True,trust_remote_code=False)
        result=prepare(args.output or DATA.parent,tok)
        result={'rows':len(result['rows']),'inputs':[r['input_tokens'] for r in result['rows']]}
    elif args.mode=='freeze':result=freeze()
    elif args.mode=='systems':result=systems(args.output or RESULT/'systems-v1')
    else:result=evaluate(args.mode,args.output or RESULT/('validation-v1' if args.mode=='validation' else 'evaluation-v1'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
