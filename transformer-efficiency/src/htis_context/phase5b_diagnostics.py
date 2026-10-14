"""Separate offline importance/profiling diagnostics; never used to tune test runs."""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .adapter import LlamaReferenceAdapter
from .learned import load_estimator
from .offline_labels import CapturePolicy, InterventionPolicy, kl, local_damage
from .phase5_data import write, sha
from .phase5_profile import Spans, instrument
from .phase5b import (DATA, RESULT, MANIFEST, TRAIN, POLICIES, read, load_run,
    make_adapter, process_memory, verify_frozen, stamp)


def ordinal_correlation(a,b):
    if len(a)<2:return 1.
    # Unique deterministic ordinal ranks; tie-breaks by original page order.
    ra=np.argsort(np.argsort(-np.asarray(a),kind='stable'),kind='stable')
    rb=np.argsort(np.argsort(-np.asarray(b),kind='stable'),kind='stable')
    return float(np.corrcoef(ra,rb)[0,1])


@torch.no_grad()
def importance(output):
    output.mkdir(parents=True,exist_ok=False)
    manifest,model,tok,dense,_=load_run();cfg=manifest['configuration'];records=[]
    for row in [r for r in read(DATA)['rows'] if r['split']=='test']:
        prepared=dense.forward(torch.tensor([row['prefix_ids']+row['suffix_ids'][:-1]])).state
        capture=CapturePolicy(8,len(model.model.layers))
        captured=LlamaReferenceAdapter(model,dense.identity,policy=capture).forward(torch.tensor([[row['suffix_ids'][-1]]]),state=prepared)
        for layer in [0,14,29]:
            call=capture.calls[layer];batch=call['features'];query=call['query'];kv=captured.state.layers[layer]
            learned=[];times=[]
            for seed in cfg['seeds']:
                estimator,mean,std,_=load_estimator(TRAIN/f'htis-seed-{seed}',dense.identity)
                start=time.perf_counter();scores=estimator((batch.values-mean)/std).tolist()
                times.append((time.perf_counter()-start)*1000);learned.append(scores)
            summary=batch.values[:,17].tolist()
            # Natural unscaled current-state value norms; no synthetic values.
            value=kv.features.value_center.norm(dim=-1).mean(0).tolist()
            n=len(batch.page_ids);ordered=sorted(range(n),key=lambda i:(learned[0][i],-batch.page_ids[i]))
            low_value_candidates=ordered[:max(1,n//2)]
            high_value_low_score=max(low_value_candidates,key=lambda i:value[i])
            tested=list(dict.fromkeys([ordered[0],high_value_low_score,max(range(n),key=lambda i:summary[i])]))
            damage=[]
            for index in tested:
                page=int(batch.page_ids[index]);p=InterventionPolicy(8,layer,(page,),n)
                intervention=LlamaReferenceAdapter(model,dense.identity,policy=p).forward(torch.tensor([[row['suffix_ids'][-1]]]),state=prepared)
                damage.append(dict(page=page,learned_score=learned[0][index],query_score=summary[index],value_center_norm=value[index],
                    local_l2=local_damage(query,kv,(page,),8),next_token_kl=kl(captured.logits[:,-1],intervention.logits[:,-1])))
            records.append(dict(case=row['id'],group=row['group'],kind=row['kind'],layer=layer,
                page_ids=list(batch.page_ids),htis_seed17=learned[0],htis_seed29=learned[1],query_summary=summary,
                value_center_norm=value,seed_rank_correlation=ordinal_correlation(*learned),
                query_rank_correlation=ordinal_correlation(learned[0],summary),score_network_ms=times,
                first_snapshot_rank_budget_invariant=True,
                reason='Frozen scorer has no budget input; subsequent rankings can change because retained features/queries differ.',
                omission_diagnostics=damage))
        print('importance '+row['id'],flush=True)
    task_changes=[]
    for layer in [0,14,29]:
        rows=[r for r in records if r['group']=='session5b/switch' and r['layer']==layer]
        a,b=rows
        # Common-prefix pages only; different questions append different suffix pages.
        common=sorted(set(a['page_ids'])&set(b['page_ids']))
        pairs=lambda r,key:[r[key][r['page_ids'].index(i)] for i in common]
        task_changes.append(dict(layer=layer,cases=[a['case'],b['case']],common_pages=len(common),
            htis_rank_correlation=ordinal_correlation(pairs(a,'htis_seed17'),pairs(b,'htis_seed17')),
            query_rank_correlation=ordinal_correlation(pairs(a,'query_summary'),pairs(b,'query_summary'))))
    verify_frozen(manifest)
    report=dict(records=records,task_conditioning=task_changes,manifest_sha256=sha(MANIFEST),
        completed_at_utc=stamp(),passed=True,
        scope='Offline full-state current-query diagnostics at layers0/14/29; natural tensors, no retraining or routing feedback. Counterfactual local L2/next-token KL do not measure answer correctness.',
        limitation='Three preselected omission candidates per layer are diagnostic; no full counterfactual ranking labels and no claim that score quality is calibrated at longer contexts.')
    write(output/'report.json',report)
    return {'cases':len({r['case'] for r in records}),'layer_snapshots':len(records),'counterfactuals':sum(len(r['omission_diagnostics']) for r in records)}


@torch.no_grad()
def profile(output):
    output.mkdir(parents=True,exist_ok=False)
    manifest,model,tok,dense,_=load_run();cfg=manifest['configuration']
    row=next(r for r in read(DATA)['rows'] if r['split']=='validation')
    prepared=dense.forward(torch.tensor([row['prefix_ids']+row['suffix_ids'][:-1]])).state
    rec=read(RESULT/'validation-v1/calibration.json')['recovery_configuration'];results=[]
    token=torch.tensor([[row['suffix_ids'][-1]]])
    for name in POLICIES:
        for recovery in [False,True]:
            a=make_adapter(model,dense,name,8,17,cfg,rec if recovery else None)
            a.forward(token,state=prepared)
            plain=[]
            for _ in range(2):
                start=time.perf_counter();expected=a.forward(token,state=prepared)
                plain.append((time.perf_counter()-start)*1000)
            recorder=Spans();prof=cProfile.Profile();start=time.perf_counter()
            with instrument(model,recorder):
                prof.enable();actual=a.forward(token,state=prepared);prof.disable()
            profiled=(time.perf_counter()-start)*1000
            torch.testing.assert_close(actual.logits,expected.logits,atol=5e-5,rtol=1e-5)
            prof.dump_stats(str(output/f'{name}-recovery{int(recovery)}.pstats'))
            results.append(dict(policy=name,recovery=recovery,plain_ms=plain,instrumented_ms=profiled,
                instrumentation_plus_cprofile_overhead_ms=profiled-statistics.mean(plain),spans=recorder.rows,
                metrics=actual.metrics,process_memory=process_memory()))
    # Native operator allocation traffic, separate from comparative timings.
    a=make_adapter(model,dense,'query',8,17,cfg)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True) as native:
        a.forward(token,state=prepared)
    operators=[dict(operator=e.key,self_cpu_us=e.self_cpu_time_total,cpu_us=e.cpu_time_total,
        calls=e.count,self_cpu_memory_bytes=e.self_cpu_memory_usage) for e in native.key_averages()]
    operators.sort(key=lambda x:-x['self_cpu_us'])
    verify_frozen(manifest)
    write(output/'report.json',dict(validation_case=row['id'],input_tokens=row['input_tokens'],results=results,
        native_operators=operators,manifest_sha256=sha(MANIFEST),passed=True,completed_at_utc=stamp(),
        scope='Separate first-compaction profiles on validation source; two plain repeats plus separate cProfile/spans and native operator run. Native allocation traffic is not cache-system peak. Nested inclusive spans must not be summed.'))
    return {'profile_cells':len(results),'passed':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['importance','profile']);parser.add_argument('--output',type=Path)
    args=parser.parse_args();print(json.dumps((importance if args.mode=='importance' else profile)(args.output or RESULT/(args.mode+'-v1')),indent=2))
