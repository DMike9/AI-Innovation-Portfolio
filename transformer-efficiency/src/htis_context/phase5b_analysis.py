"""Artifact-only paired analysis: no model execution or outcome-dependent tuning."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics

import numpy as np

from .phase5_data import sha, write
from .phase5_metrics import paired_bootstrap
from .phase5b import RESULT, MANIFEST, DATA, read, POLICIES, cell_key


def raw(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def label(policy,budget,recovery):
    return policy+(' full' if budget is None else f' {budget}p')+(' + recovery' if recovery else '')


def stats(values):
    values=list(values)
    return dict(mean=float(np.mean(values)),median=float(np.median(values)),
        std=float(np.std(values,ddof=1)) if len(values)>1 else 0.,min=min(values),max=max(values),n=len(values))


def bootstrap(pairs,comparisons=1):
    result=paired_bootstrap(pairs,repetitions=10000,seed=1729)
    groups=defaultdict(list)
    for group,value in pairs:groups[group].append(value)
    v=np.array([np.mean(x) for x in groups.values()]);rng=np.random.default_rng(1729)
    samples=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)
    alpha=.05/comparisons
    result['adjusted_percentile_ci']=[float(x) for x in np.quantile(samples,[alpha/2,1-alpha/2])]
    result['comparison_count']=comparisons
    result['adjusted_confidence']=1-alpha
    return result


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['policy'],r['budget'],r['recovery'])].append(r)
    results=[]
    for (policy,budget,recovery),current in sorted(groups.items(),key=lambda x:str(x[0])):
        steps=[s for r in current for s in r.get('teacher_steps',[])+r['free_steps']]
        entry=dict(policy=policy,budget=budget,recovery=recovery,arm=label(policy,budget,recovery),trials=len(current),
            cases=len({r['case'] for r in current}),groups=len({r['group'] for r in current}),
            mean_f1=statistics.mean(r['f1'] for r in current),mean_em=statistics.mean(r['em'] for r in current),
            f1=stats(r['f1'] for r in current),output_tokens=stats(r['output_tokens'] for r in current),
            eos_completions=sum(r['completion']=='eos' for r in current),
            max_active_kv_bytes=max(s['active_kv_bytes'] for s in steps),
            max_archive_kv_bytes=max(s['archive_kv_bytes'] for s in steps),
            max_archive_storage_bytes=max(s['archive_bytes'] for s in steps),
            max_index_estimate_bytes=max(s['index_bytes'] for s in steps),
            max_metadata_tensor_bytes=max(s['metadata_bytes'] for s in steps),
            max_scorer_tensor_bytes=max(s['scorer_bytes'] for s in steps),
            max_total_storage_bytes=max(s['known_cache_system_storage_bytes']+r.get('prefix_storage_bytes',0) for r in current for s in r.get('teacher_steps',[])+r['free_steps']),
            max_known_working_estimate_bytes=max(s['working_tensor_estimate_bytes'] for s in steps),
            max_temporary_feature_bytes=max(s['feature_working_bytes'] for s in steps),
            recovery_fetches=sum(s['recovery'].get('fetch_count',0) for s in steps),
            transfer_copy_ms=sum(s['copy_event_ms'] for s in steps),
            copy_bytes=sum(s['recovery'].get('transferred_copy_bytes',0) for s in steps),
            mean_controller_ms=statistics.mean(s['recovery'].get('controller_ms',0) for s in steps),
            mean_recovery_scorer_ms=statistics.mean(s['recovery'].get('scorer_ms',0) for s in steps),
            mean_selection_ms=statistics.mean(s['selection_ms'] for s in steps),
            mean_insertion_feature_ms=statistics.mean(s['feature_update_ms'] for s in steps),
            seeds={str(seed):dict(mean_f1=statistics.mean(r['f1'] for r in current if r['seed']==seed),
                trials=sum(r['seed']==seed for r in current)) for seed in sorted({r['seed'] for r in current})})
        for key in ['mean_nll','mean_kl','mean_decode_ms','end_to_end_ms','prefill_ms','lookup_ms','admission_ms','suffix_ms','ttft_ms','decode_wall_ms','setup_ms','tokens_per_second','decode_tokens_per_second','observation_overhead_ms']:
            if key in current[0]:
                entry[key]=statistics.mean(r[key] for r in current)
                entry[key+'_stats']=stats(r[key] for r in current)
        results.append(entry)
    return results


def pareto(table,cost,memory):
    return {r['arm']:[b['arm'] for b in table if b['mean_f1']>=r['mean_f1'] and b[cost]<=r[cost] and b[memory]<=r[memory]
        and (b['mean_f1']>r['mean_f1'] or b[cost]<r[cost] or b[memory]<r[memory])] for r in table}


def paired_quality(rows,policy,budget,recovery,reference,ref_recovery=False,metric='f1',comparisons=1):
    index={cell_key(r):r for r in rows}
    pairs=[];regressions=[]
    for row in rows:
        if (row['policy'],row['budget'],row['recovery'])!=(policy,budget,recovery):continue
        other=index[(row['case'],reference,budget,row['seed'],ref_recovery)]
        delta=row[metric]-other[metric];pairs.append((row['group'],delta))
        regressions.append(dict(case=row['case'],group=row['group'],seed=row['seed'],delta=delta,
            policy_value=row[metric],reference_value=other[metric],output=row['raw_output'],reference_output=other['raw_output']))
    return dict(policy=policy,reference=reference,budget=budget,recovery=recovery,ref_recovery=ref_recovery,
        metric=metric,statistics=bootstrap(pairs,comparisons),case_pairs=regressions,
        mean_case_delta=statistics.mean(x['delta'] for x in regressions),
        improved=sum(x['delta']>0 for x in regressions),unchanged=sum(x['delta']==0 for x in regressions),worsened=sum(x['delta']<0 for x in regressions))


def analyze(output):
    output.mkdir(parents=True,exist_ok=False)
    manifest=read(MANIFEST);quality=raw(RESULT/'evaluation-v1/trials.jsonl.gz');system=raw(RESULT/'systems-v1/systems.jsonl.gz')
    qreport=read(RESULT/'evaluation-v1/report.json');sreport=read(RESULT/'systems-v1/report.json')
    if not qreport['passed'] or not sreport['passed']:raise ValueError('Incomplete execution')
    if sha(RESULT/'evaluation-v1/trials.jsonl.gz')!=qreport['raw_sha256'] or sha(RESULT/'systems-v1/systems.jsonl.gz')!=sreport['raw_sha256']:raise ValueError('Raw artifact checksum mismatch')
    table=summarize(quality);quality_pareto=pareto(table,'mean_decode_ms','max_total_storage_bytes')
    simple=[paired_quality(quality,'htis',budget,False,name,metric='f1',comparisons=9) for budget in [4,8,16] for name in ['recent','query','random']]
    nll=[paired_quality(quality,'htis',budget,False,name,metric='mean_nll',comparisons=9) for budget in [4,8,16] for name in ['recent','query','random']]
    recovery=[paired_quality(quality,name,budget,True,name,metric='f1',comparisons=12) for budget in [4,8,16] for name in POLICIES]
    recovery_kl=[paired_quality(quality,name,budget,True,name,metric='mean_kl',comparisons=12) for budget in [4,8,16] for name in POLICIES]
    systems={traffic:summarize([r for r in system if r['traffic']==traffic]) for traffic in ['cold-no-cache','warm-repeat','cold-admission','modified-prefix','shared-prefix-task-change','shared-prefix-cold-control']}
    warm_pareto=pareto(systems['warm-repeat'],'end_to_end_ms','max_total_storage_bytes')
    warm_index={(r['case'],r['policy'],r['budget'],r['recovery'],r['repeat']):r for r in system if r['traffic']=='warm-repeat'}
    reuse=[];amortization=[];recovery_performance=[]
    for cold in [r for r in system if r['traffic']=='cold-no-cache']:
        warm=warm_index[(cold['case'],cold['policy'],cold['budget'],cold['recovery'],cold['repeat'])]
        reuse.append(dict(case=cold['case'],group=cold['group'],policy=cold['policy'],budget=cold['budget'],recovery=cold['recovery'],repeat=cold['repeat'],
            latency_saving_ms=cold['end_to_end_ms']-warm['end_to_end_ms'],cold_ms=cold['end_to_end_ms'],warm_ms=warm['end_to_end_ms'],
            prefill_saved_ms=cold['prefill_ms'],lookup_ms=warm['lookup_ms'],saved_prefix_tokens=warm['saved_prefix_tokens'],
            prefix_bytes=warm['prefix_storage_bytes'],equivalent_output=cold['output_ids']==warm['output_ids']))
        admitted=next(r for r in system if r['traffic']=='cold-admission' and
            (r['case'],r['policy'],r['budget'],r['recovery'],r['repeat'])==(cold['case'],cold['policy'],cold['budget'],cold['recovery'],cold['repeat']))
        for hits in [0,1,8]:
            amortization.append(dict(case=cold['case'],policy=cold['policy'],budget=cold['budget'],recovery=cold['recovery'],repeat=cold['repeat'],warm_repeats=hits,
                measured_input_admission_ms=admitted['end_to_end_ms'],derived_cache_mean_ms=(admitted['end_to_end_ms']+hits*warm['end_to_end_ms'])/(hits+1),
                derived_no_cache_mean_ms=cold['end_to_end_ms'],derived_hit_rate=hits/(hits+1),
                scope='Derived traffic-mix amortization from measured requests; not new measured trials'))
    for recovered in [r for r in system if r['traffic']=='warm-repeat' and r['recovery'] and r['policy']!='full']:
        plain=warm_index[(recovered['case'],recovered['policy'],recovered['budget'],False,recovered['repeat'])]
        recovery_performance.append(dict(policy=recovered['policy'],budget=recovered['budget'],case=recovered['case'],group=recovered['group'],repeat=recovered['repeat'],
            latency_ratio=recovered['end_to_end_ms']/plain['end_to_end_ms'],latency_delta_ms=recovered['end_to_end_ms']-plain['end_to_end_ms'],
            f1_delta=recovered['f1']-plain['f1'],extra_storage_bytes=max(s['known_cache_system_storage_bytes'] for s in recovered['free_steps'])-max(s['known_cache_system_storage_bytes'] for s in plain['free_steps']),
            fetches=sum(s['recovery'].get('fetch_count',0) for s in recovered['free_steps'])))
    retrieval_quality=[]
    for entry in recovery:
        for pair in entry['case_pairs']:
            matched=next(r for r in quality if (r['case'],r['policy'],r['budget'],r['seed'],r['recovery'])==(pair['case'],entry['policy'],entry['budget'],pair['seed'],True))
            fetches=sum(s['recovery'].get('fetch_count',0) for s in matched['free_steps'])
            retrieval_quality.append(dict(policy=entry['policy'],budget=entry['budget'],**pair,fetches=fetches,
                outcome='improved' if pair['delta']>0 else 'worsened' if pair['delta']<0 else 'unchanged',
                unnecessary_by_answer_f1_proxy=fetches>0 and pair['delta']<=0,
                note='Answer F1 change is associated with full recovery route; not isolated causal utility of each reload.'))
    # Retention diagnostics use only actual generated trajectories.
    data={r['id']:r for r in read(DATA)['rows']};retention=[]
    for row in quality:
        source=data[row['case']];start=source['relevant_token_start']//8;end=(source['relevant_token_end']-1)//8
        target=set(range(start,end+1));steps=row['free_steps']
        retention.append(dict(case=row['case'],policy=row['policy'],budget=row['budget'],recovery=row['recovery'],seed=row['seed'],f1=row['f1'],
            initial_relevant_page_retention=float(np.mean([len(set(ids)&target)/len(target) for ids in steps[0]['active_pages']])),
            last_relevant_page_retention=float(np.mean([len(set(ids)&target)/len(target) for ids in steps[-1]['active_pages']])),
            note='Coarse target paragraph/opening-fact page overlap, not complete causal evidence or precise answer-span coverage.'))
    a_supported=any(all(x['statistics']['adjusted_percentile_ci'][0]>0 for x in simple if x['budget']==budget) for budget in [4,8,16])
    a_mean_better=any(all(x['mean_case_delta']>0 for x in simple if x['budget']==budget) for budget in [4,8,16])
    b_candidates=[]
    for comparison in recovery:
        perf=[r for r in recovery_performance if (r['policy'],r['budget'])==(comparison['policy'],comparison['budget'])]
        kl_comp=next(r for r in recovery_kl if (r['policy'],r['budget'])==(comparison['policy'],comparison['budget']))
        b_candidates.append(dict(policy=comparison['policy'],budget=comparison['budget'],mean_f1_delta=comparison['mean_case_delta'],
            adjusted_ci=comparison['statistics']['adjusted_percentile_ci'],mean_kl_delta=kl_comp['mean_case_delta'],
            mean_warm_latency_ratio=statistics.mean(r['latency_ratio'] for r in perf)))
    b_supported=any(r['adjusted_ci'][0]>0 and r['mean_kl_delta']<=0 and r['mean_warm_latency_ratio']<=2 for r in b_candidates)
    b_possible=any(r['mean_f1_delta']>0 and r['mean_warm_latency_ratio']<=2 for r in b_candidates)
    learned_warm=[r for r in systems['warm-repeat'] if r['policy']=='htis']
    nondominated=[r['arm'] for r in learned_warm if not warm_pareto[r['arm']]]
    # C's positive decision also requires paired F1 evidence against all simple
    # warm compositions, using the actual systems population (two groups).
    c_pairs=[]
    for budget in [4,8,16]:
        for recovered in [False,True]:
            for baseline in ['recent','query','random']:
                pairs=[]
                for row in [r for r in system if r['traffic']=='warm-repeat' and (r['policy'],r['budget'],r['recovery'])==('htis',budget,recovered)]:
                    ref=warm_index[(row['case'],baseline,budget,recovered,row['repeat'])]
                    pairs.append((row['group'],row['f1']-ref['f1']))
                c_pairs.append(dict(budget=budget,recovery=recovered,reference=baseline,statistics=bootstrap(pairs,18)))
    c_supported=any(label('htis',budget,recovered) in nondominated and all(x['statistics']['adjusted_percentile_ci'][0]>0 for x in c_pairs if (x['budget'],x['recovery'])==(budget,recovered)) for budget in [4,8,16] for recovered in [False,True])
    contributions=dict(A=dict(status='SUPPORTED' if a_supported else 'INCONCLUSIVE' if a_mean_better else 'NOT SUPPORTED',evidence='paired_htis_vs_simple; all three simple comparisons required at one budget'),
        B=dict(status='SUPPORTED' if b_supported else 'INCONCLUSIVE' if b_possible else 'NOT SUPPORTED',evidence='paired_recovery and recovery_cost_constraints',candidates=b_candidates),
        C=dict(status='SUPPORTED' if c_supported else 'INCONCLUSIVE' if nondominated else 'NOT SUPPORTED',evidence='warm_systems_pareto and paired_warm_compositions',nondominated_htis_cells=nondominated,
            boundary='Generic exact prefix reuse is separately measured; it is prior art, not evidence of novel learned coordination.'))
    report=dict(passed=True,quality_trials=len(quality),actual_system_requests=len(system),quality_groups=len({r['group'] for r in quality}),
        quality_table=table,systems_tables=systems,quality_pareto_dominated_by=quality_pareto,warm_systems_pareto_dominated_by=warm_pareto,
        paired_htis_vs_simple=simple,paired_htis_nll_vs_simple=nll,paired_recovery=recovery,paired_recovery_kl=recovery_kl,
        paired_warm_compositions=c_pairs,prefix_reuse_pairs=reuse,prefix_reuse_summary=stats(r['latency_saving_ms'] for r in reuse),
        prefix_reuse_ci=bootstrap([(r['group'],r['latency_saving_ms']) for r in reuse]),
        derived_amortization=amortization,recovery_performance_pairs=recovery_performance,recovery_quality_outcomes=retrieval_quality,
        retention_diagnostics=retention,contributions=contributions,
        workload_tables={kind:summarize([r for r in quality if r['kind']==kind]) for kind in sorted({r['kind'] for r in quality})},
        source_sha256=dict(quality=qreport['raw_sha256'],systems=sreport['raw_sha256'],manifest=sha(MANIFEST)),
        limitations=['Nine quality clusters and only two performance clusters; two checkpoint seeds, two repeat timings.',
            'Length/workload confounding; foundation-model public-data contamination unknown; small base model, 16/32-token limits.',
            'Dense prefill required; decode capacities are not end-to-end memory caps; structural/index bytes are estimates.',
            'OS memory counters are process-wide and include retained research fixtures, not isolated cache peaks.',
            'Reference CPU implementation, instrumentation overhead, one model, no CUDA; two complete LongBench-source cases do not establish a full benchmark score.'])
    write(output/'summary.json',report)
    lines=['# Phase 5B comparison','', '| Policy | Pages | Recovery | F1 | EM | NLL | KL | Decode ms | Active KV MiB | Known cache MiB |', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f'| {r["policy"]} | {r["budget"] or "full"} | {r["recovery"]} | {r["mean_f1"]:.4f} | {r["mean_em"]:.4f} | {r["mean_nll"]:.4f} | {r["mean_kl"]:.4f} | {r["mean_decode_ms"]:.2f} | {r["max_active_kv_bytes"]/2**20:.3f} | {r["max_total_storage_bytes"]/2**20:.3f} |')
    lines+=['','Known cache bytes include physical tensors plus labeled recipe/index estimates. Dense prefill and base model are separate. All policies/budgets included. Teacher metrics are not generated answer accuracy.','',f'Actual quality trials: {len(quality)}. Actual request executions: {len(system)}.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plots(output,table,systems['warm-repeat'])
    return {'quality_trials':len(quality),'requests':len(system),'contributions':contributions}


def plots(output,quality,warm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'full':'black','recent':'#4477aa','query':'#228833','random':'#ccbb44','htis':'#aa3377'}
    plan=[('quality-latency',quality,'mean_decode_ms','Mean measured decode ms/token',False),
          ('quality-active-memory',quality,'max_active_kv_bytes','Maximum physical decode KV MiB',True),
          ('quality-total-memory',quality,'max_total_storage_bytes','Maximum known cache-system MiB (estimates included)',True),
          ('warm-quality-latency',warm,'end_to_end_ms','Mean warm request ms (two source groups)',False),
          ('warm-quality-total-memory',warm,'max_total_storage_bytes','Maximum warm cache + prefix storage MiB',True)]
    for name,table,x,label_text,mb in plan:
        fig,ax=plt.subplots(figsize=(11,6.5),layout='constrained')
        for r in table:
            value=r[x]/2**20 if mb else r[x]
            marker={None:'*',4:'o',8:'s',16:'^'}[r['budget']]
            ax.scatter(value,r['mean_f1'],marker=marker,s=95,color=colors[r['policy']],
                facecolors='none' if r['recovery'] else colors[r['policy']],linewidths=1.7,
                label=r['arm'])
        ax.set_xlabel(label_text);ax.set_ylabel('Mean generated-answer token F1');ax.set_title('Phase 5B — actual CPU artifacts; all cells')
        ax.grid(alpha=.2);ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=7,ncol=1)
        if 'latency' in name:ax.set_xscale('log')
        fig.savefig(output/(name+'.png'),dpi=160);fig.savefig(output/(name+'.svg'));plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=RESULT/'analysis-v1')
    args=parser.parse_args();print(json.dumps(analyze(args.output),indent=2))
