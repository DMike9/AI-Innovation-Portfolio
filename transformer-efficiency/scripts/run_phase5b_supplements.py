"""Separately frozen exploratory mechanism controls; primary protocol unchanged."""
from argparse import ArgumentParser
from pathlib import Path
import json
import random
import time

import torch

from htis_context.phase5b import (read,load_run,verify_frozen,stamp,quality,request,
    DATA,RESULT,MANIFEST,POLICIES,budget_bytes)
from htis_context.phase5 import append
from htis_context.phase5_data import sha,write
from htis_context.prefix_cache import ExactPrefixCache
from htis_context.state import _fingerprint

CONFIG=RESULT/'exploratory-controls-config-v1.json'


def freeze():
    if CONFIG.exists():raise FileExistsError('Never overwrite supplement config')
    primary=read(MANIFEST)
    config=dict(frozen_at_utc=stamp(),experiment_label='Separately labeled exploratory controls; no primary protocol/decision changes',
        primary_manifest_sha256=sha(MANIFEST),calibration_sha256=sha(RESULT/'validation-v1/calibration.json'),
        source_sha256=sha(Path(__file__)),budget=8,policies=POLICIES,seeds=[17,29],
        quality_cases=primary['quality_cases'],systems_cases=primary['systems_cases'],
        archive_control='Identical existing RecoveryPolicy/configuration, except archive capacity zero; index and actual discard copies still charged',
        quality_trials=80,archive_control_requests=48,
        high_reuse_seed=17,high_reuse_warm_repetitions=8,high_reuse_requests=80,
        rationale='Middle budget chosen for a mechanical availability control, not selected by held-out quality. All four baselines and all primary quality cases included. Eight warm repeats are actual requests, not derived traffic mixes.',
        timing='External argmax timestamp records actual first token decision availability; numerical outputs must match. No serving/network streaming endpoint.',
        inference='Exploratory same-case controls do not increase independent source/session count or retroactively alter primary A/B/C criteria.')
    write(CONFIG,config);return config


def loaded():
    extra=read(CONFIG)
    if extra['source_sha256']!=sha(Path(__file__)) or extra['primary_manifest_sha256']!=sha(MANIFEST):raise ValueError('Supplement freeze changed')
    if extra['calibration_sha256']!=sha(RESULT/'validation-v1/calibration.json'):raise ValueError('Calibration changed')
    return extra,load_run()


def measured_request(*args):
    """Read-only timestamp hook around existing token decision; restored always."""
    start=time.perf_counter();first=[];old=torch.Tensor.argmax
    def argmax(tensor,*positional,**keywords):
        result=old(tensor,*positional,**keywords)
        if not first:first.append(time.perf_counter())
        return result
    try:
        torch.Tensor.argmax=argmax
        row,logits=request(*args)
    finally:
        torch.Tensor.argmax=old
    if not first:raise AssertionError('No actual token decision timestamp')
    row['direct_first_argmax_available_ms']=(first[0]-start)*1000
    row['direct_minus_phase_sum_ttft_ms']=row['direct_first_argmax_available_ms']-row['ttft_ms']
    row['ttft_definition']='First greedy argmax result available, including all request phase gaps; before Python scalar conversion; no network streaming'
    return row,logits


@torch.no_grad()
def archive_control():
    output=RESULT/'exploratory-no-archive-v1';output.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter();extra,(manifest,model,tok,dense,_)=loaded();cfg=manifest['configuration']
    rec=dict(read(RESULT/'validation-v1/calibration.json')['recovery_configuration'],max_archive_bytes=0)
    count=0;checks=[]
    for n,row in enumerate(r for r in read(DATA)['rows'] if r['split']=='test'):
        prefix=dense.forward(torch.tensor([row['prefix_ids']])).state
        prepared=dense.forward(torch.tensor([row['suffix_ids'][:-1]]),state=prefix).state
        before=_fingerprint(prepared);teacher=[];state=prepared
        for token in [row['suffix_ids'][-1]]+row['answer_ids'][:-1]:
            run=dense.forward(torch.tensor([[token]]),state=state);state=run.state;teacher.append(run.logits[:,-1].clone())
        cells=[(name,seed) for name in POLICIES for seed in extra['seeds']]
        random.Random(3311+n).shuffle(cells)
        for name,seed in cells:
            result=quality(row,name,8,seed,prepared,teacher,model,dense,tok,cfg,rec)
            result['experiment_label']=extra['experiment_label'];result['archive_capacity_bytes']=0
            observations=result['teacher_steps']+result['free_steps']
            passed=all(s['active_kv_bytes']<=budget_bytes(read('transformer-efficiency/configs/smollm2-135m.json'),8)
                and s['archive_kv_bytes']==0 and s['archive_bytes']==0 and s['recovery'].get('fetch_count',0)==0 for s in observations)
            if not passed:raise AssertionError('Zero-archive/physical budget control failed')
            append(output/'trials.jsonl.gz',result);count+=1
            checks.append(dict(case=row['id'],policy=name,seed=seed,passed=passed))
            write(output/'progress.json',dict(trials=count,planned=80,case=row['id'],at_utc=stamp()))
            print('zero archive quality '+str(count),flush=True)
        if _fingerprint(prepared)!=before:raise AssertionError('Dense fixture mutated')
    system_count=0;system_checks=[]
    for row in [r for r in read(DATA)['rows'] if r['id'] in extra['systems_cases']]:
        for name in POLICIES:
            for repeat,seed in enumerate(extra['seeds']):
                cache=ExactPrefixCache(cfg['prefix_bytes']);current=[]
                for reuse,traffic in [(True,'cold-admission'),(False,'cold-no-cache'),(True,'warm-repeat')]:
                    trial,logits=measured_request(row,name,8,seed,True,reuse,repeat,model,dense,tok,cfg,rec,cache)
                    trial.update(traffic=traffic,experiment_label=extra['experiment_label'],archive_capacity_bytes=0)
                    append(output/'systems.jsonl.gz',trial);system_count+=1;current.append((trial,logits))
                reference=current[0]
                passed=all(r['output_ids']==reference[0]['output_ids'] and all(torch.allclose(a,b,atol=5e-5,rtol=1e-5)
                    for a,b in zip(reference[1],logits)) for r,logits in current)
                system_checks.append(dict(case=row['id'],policy=name,seed=seed,passed=bool(passed)))
                if not passed:raise AssertionError('Supplement prefix equivalence failed')
                write(output/'progress.json',dict(trials=count,requests=system_count,planned_requests=48,at_utc=stamp()))
                print('zero archive requests '+str(system_count),flush=True)
    verify_frozen(manifest)
    report=dict(trials=count,requests=system_count,checks=checks,system_checks=system_checks,passed=all(c['passed'] for c in checks+system_checks),
        configuration_sha256=sha(CONFIG),quality_sha256=sha(output/'trials.jsonl.gz'),systems_sha256=sha(output/'systems.jsonl.gz'),
        wall_seconds=time.perf_counter()-began,completed_at_utc=stamp(),experiment_label=extra['experiment_label'])
    write(output/'report.json',report);return report


@torch.no_grad()
def high_reuse():
    output=RESULT/'exploratory-high-reuse-v1';output.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter();extra,(manifest,model,tok,dense,_)=loaded();cfg=manifest['configuration'];count=0;checks=[]
    for row in [r for r in read(DATA)['rows'] if r['id'] in extra['systems_cases']]:
        for name in POLICIES:
            cache=ExactPrefixCache(cfg['prefix_bytes']);seed=extra['high_reuse_seed']
            reference,reference_logits=measured_request(row,name,8,seed,False,True,0,model,dense,tok,cfg,None,cache)
            reference.update(traffic='cold-admission',experiment_label=extra['experiment_label'])
            append(output/'systems.jsonl.gz',reference);count+=1
            for repeat in range(extra['high_reuse_warm_repetitions']):
                warm,logits=measured_request(row,name,8,seed,False,True,repeat,model,dense,tok,cfg,None,cache)
                warm.update(traffic='eight-repeat-warm',experiment_label=extra['experiment_label'])
                passed=warm['hit'] and warm['output_ids']==reference['output_ids'] and all(torch.allclose(a,b,atol=5e-5,rtol=1e-5) for a,b in zip(reference_logits,logits))
                checks.append(dict(case=row['id'],policy=name,repeat=repeat,passed=bool(passed)))
                if not passed:raise AssertionError('High-reuse output/lookup failed')
                append(output/'systems.jsonl.gz',warm);count+=1
                write(output/'progress.json',dict(requests=count,planned=80,at_utc=stamp()))
                print('eight repeat requests '+str(count),flush=True)
            cold,logits=measured_request(row,name,8,seed,False,False,0,model,dense,tok,cfg,None,cache)
            cold.update(traffic='cold-no-cache-control',experiment_label=extra['experiment_label'])
            if cold['output_ids']!=reference['output_ids']:raise AssertionError('Cold high-reuse control changed outputs')
            append(output/'systems.jsonl.gz',cold);count+=1
    verify_frozen(manifest)
    report=dict(requests=count,checks=checks,passed=all(c['passed'] for c in checks),configuration_sha256=sha(CONFIG),
        raw_sha256=sha(output/'systems.jsonl.gz'),wall_seconds=time.perf_counter()-began,completed_at_utc=stamp(),
        actual_warm_repeats_per_admission=8,scope='Exploratory real high-reuse traffic, all four policies at eight pages, seed17, no recovery, two sources; does not alter primary contribution criteria')
    write(output/'report.json',report);return report


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('mode',choices=['freeze','archive-control','high-reuse']);args=parser.parse_args()
    result={'freeze':freeze,'archive-control':archive_control,'high-reuse':high_reuse}[args.mode]()
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','system_checks']},indent=2))
