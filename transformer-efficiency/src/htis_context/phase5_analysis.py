"""Analyze actual Phase 5 artifacts with paired session resampling and figures."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics
import numpy as np
from .phase5_data import write,sha
from .phase5_metrics import paired_bootstrap,dominated


def read_raw(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f]


def group(row):
    # Preregistered in source before held-out execution: the three positional
    # variants share an authored fact and filler, so resample them together.
    return 'session/position-family' if row['kind'] in {'beginning','middle','end'} else row['group']


def aggregate(rows):
    cases=defaultdict(list)
    for r in rows:cases[r['case']].append(r)
    def means(name):return [statistics.mean(r[name] for r in seeds) for seeds in cases.values()]
    steps=[s for r in rows for s in r['free_steps']]
    recoveries=[s['recovery'] for s in steps]
    events=[e for r in recoveries for e in r.get('events',[])]
    bounds=[b for r in recoveries for b in r.get('bounds',[])]
    decisions=[d for r in recoveries for d in r.get('decisions',[])]
    churn=[]
    for r in rows:
        for prev,nxt in zip(r['free_steps'],r['free_steps'][1:]):
            churn.extend(len(set(a)^set(b)) for a,b in zip(prev['active_pages'],nxt['active_pages']))
    return {'arm':rows[0]['arm'],'examples':len(cases),'trials':len(rows),'seeds':sorted({r['seed'] for r in rows}),
        'mean_f1':statistics.mean(means('f1')),'median_f1':statistics.median(means('f1')),'mean_em':statistics.mean(means('em')),
        'mean_kl':statistics.mean(means('mean_kl')),'median_kl':statistics.median(means('mean_kl')),
        'mean_nll':statistics.mean(means('mean_nll')),'mean_decode_ms':statistics.mean(means('mean_decode_ms')),
        'median_decode_ms':statistics.median(means('mean_decode_ms')),
        'p50_step_ms':float(np.quantile([s['forward_ms'] for s in steps],.5)),'p95_step_ms':float(np.quantile([s['forward_ms'] for s in steps],.95)),
        'max_active_kv_bytes':max(r['max_active_kv_bytes'] for r in rows),'max_total_storage_bytes':max(r['max_total_storage_bytes'] for r in rows),
        'max_archive_bytes':max(s['archive_bytes'] for s in steps),'max_index_bytes':max(s['index_bytes'] for s in steps),
        'max_working_tensor_estimate_bytes':max(s['working_tensor_estimate_bytes'] for s in steps),
        'max_feature_working_bytes':max(s['feature_working_bytes'] for s in steps),
        'max_query_storage_bytes':max(s.get('query_storage_bytes',0) for s in steps),
        'mean_feature_update_ms':statistics.mean(s.get('feature_update_ms',0) for s in steps),
        'compaction_copy_bytes':sum(s.get('compaction_copy_bytes',0) for s in steps),
        'mean_selection_ms':statistics.mean(s['selection_ms'] for s in steps),
        'mean_controller_ms':statistics.mean(r.get('controller_ms',0) for r in recoveries),
        'mean_scorer_ms':statistics.mean(r.get('scorer_ms',0) for r in recoveries),
        'mean_bound_ms':statistics.mean(r.get('bound_ms',0) for r in recoveries),
        'mean_fallback_ms':statistics.mean(r.get('fallback_ms',0) for r in recoveries),
        'reloads':sum(e['action']=='reload' for e in events),'offloads':sum(e['action']=='offload' for e in events),
        'copy_bytes':sum(e.get('copy_bytes',0) for e in events),'reload_ms':sum(e.get('ms',0) for e in events if e['action']=='reload'),
        'fallbacks':sum(b.get('exact_current_state',False) for b in bounds),'bound_records':len(bounds),
        'bounds_skipped':sum(b.get('reason')=='bound-frequency-skip' for b in bounds),
        'tolerance_unmet':sum(not b['tolerance_met'] for b in bounds),
        'bound_validation_failures':sum(not b.get('bound_covers_measured_error',True) for b in bounds),
        'refreshes':sum(d['refresh'] for d in decisions),'controller_decisions':len(decisions),
        'benefit_gates':sum(d.get('benefit_cost_gate',False) for d in decisions),
        'mean_page_symmetric_difference':statistics.mean(churn) if churn else 0,
        'conditional_scoring_calls':sum(r.get('conditional',{}).get('scoring_calls',0) for r in rows),
        'conditional_trigger_ms':sum(r.get('conditional',{}).get('trigger_ms',0) for r in rows),
        'seed_means':{str(seed):{key:statistics.mean(r[key] for r in rows if r['seed']==seed) for key in ['f1','mean_kl','mean_decode_ms']} for seed in sorted({r['seed'] for r in rows})}}


def paired(rows,arm,reference,metric):
    lhs=defaultdict(list);rhs=defaultdict(list);records={}
    for r in rows:
        records[r['case']]=r
        if r['arm']==arm:lhs[r['case']].append(r[metric])
        if r['arm']==reference:rhs[r['case']].append(r[metric])
    if set(lhs)!=set(rhs):raise ValueError('Pair analysis needs equal workload sets')
    return paired_bootstrap([(group(records[k]),statistics.mean(v)-statistics.mean(rhs[k])) for k,v in lhs.items()])


def analyze(directory,manifest_path,output):
    if output.exists():raise FileExistsError('Do not overwrite analysis')
    output.mkdir(parents=True)
    manifest=json.loads(manifest_path.read_text());report=json.loads((directory/'report.json').read_text())
    if report['manifest_sha256']!=sha(manifest_path) or report['raw_trials_sha256']!=sha(directory/'trials.jsonl.gz'):
        raise ValueError('Frozen run/manifest checksum mismatch')
    rows=read_raw(directory/'trials.jsonl.gz');systems=read_raw(directory/'systems.jsonl.gz')
    core=[r for r in rows if r['arm'] in manifest['core_arms']]
    table=[aggregate([r for r in core if r['arm']==arm]) for arm in manifest['core_arms']]
    # Pure window uses a different protection budget and is kept separate.
    pareto=dominated([r for r in table if r['arm']!='window'])
    refs=[r for r in core if r['arm']=='recent'];paired_results=[];workloads=[]
    for arm in manifest['core_arms']:
        comparisons={key:paired(core,arm,'recent',key) for key in ['f1','mean_kl','mean_nll','mean_decode_ms']}
        paired_results.append({'arm':arm,'reference':'recent','comparisons':comparisons})
    for case in sorted({r['case'] for r in rows}):
        subset=[r for r in core if r['case']==case];summary={}
        for arm in manifest['core_arms']:
            current=[r for r in subset if r['arm']==arm]
            summary[arm]={key:statistics.mean(r[key] for r in current) for key in ['f1','em','mean_kl','mean_nll','mean_decode_ms']}
            summary[arm]['outputs']=[{'seed':r['seed'],'text':r['raw_output']} for r in current]
        workloads.append({'case':case,'kind':subset[0]['kind'],'group':group(subset[0]),'metrics':summary})
    recovery=[]
    for name,reference in [('recent/recovery','recent'),('htis/recovery','htis')]:
        pairs=[(w['case'],w['metrics'][name],w['metrics'][reference]) for w in workloads]
        recovery.append({'arm':name,'reference':reference,'examples':len(pairs),
            'kl_improved':sum(a['mean_kl']<b['mean_kl'] for _,a,b in pairs),
            'f1_improved':sum(a['f1']>b['f1'] for _,a,b in pairs),
            'f1_worse':sum(a['f1']<b['f1'] for _,a,b in pairs)})
    ablations=[]
    subset=[r for r in rows if r['case'] in manifest['ablation_cases']]
    for arm in manifest['ablation_arms']:
        current=[r for r in subset if r['arm']==arm]
        reference='recent' if arm.startswith('recent') or arm.startswith('bound') else 'htis/recovery' if '/recovery' in arm else 'htis'
        ablations.append({'aggregate':aggregate(current),'reference':reference,
            'paired':{key:paired([r for r in subset if r['arm'] in {arm,reference}],arm,reference,key) for key in ['f1','mean_kl','mean_decode_ms']}})
    factorial=[]
    for arm in manifest['factorial_arms']:
        for reuse in (False,True):
            current=[s for s in systems if s['arm']==arm and s['reuse']==reuse]
            factorial.append({'arm':arm,'importance':arm.startswith('htis'),'recovery':'/recovery' in arm,'reuse':reuse,
                'requests':len(current),'mean_prefill_ms':statistics.mean(s['prefill_ms'] for s in current),
                'mean_lookup_ms':statistics.mean(s['lookup_ms'] for s in current),
                'mean_ttft_ms':statistics.mean(s['ttft_ms'] for s in current),'p50_ttft_ms':float(np.quantile([s['ttft_ms'] for s in current],.5)),
                'p95_ttft_ms':float(np.quantile([s['ttft_ms'] for s in current],.95)),
                'mean_end_to_end_ms':statistics.mean(s['end_to_end_ms'] for s in current),
                'mean_tokens_per_second':statistics.mean(s['tokens_per_second'] for s in current),
                'max_prefix_storage_bytes':max(s['prefix_storage_bytes'] for s in current),
                'max_total_including_prefix_bytes':max(s['prefix_storage_bytes']+max(x['total_storage_bytes'] for x in s['steps']) for s in current)})
    interactions=[]
    for case in manifest['systems_cases']:
        means={(arm,reuse):statistics.mean(s['end_to_end_ms'] for s in systems if s['case']==case and s['arm']==arm and s['reuse']==reuse)
               for arm in manifest['factorial_arms'] for reuse in (False,True)}
        interactions.append({'case':case,'importance_x_reuse_ms':(means['htis',True]-means['simple-selected',True])-(means['htis',False]-means['simple-selected',False]),
            'recovery_x_reuse_simple_ms':(means['simple-selected/recovery',True]-means['simple-selected',True])-(means['simple-selected/recovery',False]-means['simple-selected',False]),
            'interpretation':'Two sequential timing repeats per cell; noisy interactions, not causal systems-generalization evidence.'})
    results={'core':table,'pareto_dominated_by':pareto,'paired_bootstrap':paired_results,'per_workload':workloads,'ablations':ablations,
        'recovery_success':recovery,'factorial':factorial,'interactions':interactions,'trials':len(rows),'actual_system_requests':len(systems),
        'executed_checks':len(report['checks']),'passed':report['passed'],
        'upstream_sdpa':{'mean_f1':statistics.mean(s['f1'] for s in report['upstream_sdpa']),
            'mean_prefill_ms':statistics.mean(s['prefill_ms'] for s in report['upstream_sdpa']),
            'mean_decode_ms':statistics.mean(ms for s in report['upstream_sdpa'] for ms in s['decode_ms']),
            'checks':sum(len(s['checks']) for s in report['upstream_sdpa']),
            'max_equivalence_abs':max(c['max_abs'] for s in report['upstream_sdpa'] for c in s['checks'])},
        'stats_scope':'Eight cases, six document/session groups after clustering positional probes; only two cases for ablations and factorials. No statistical significance claim.',
        'memory_scope':'Measured tensors + index/header estimates; excludes weights, Python heap/allocator, source/tokenizer, offline teacher fixtures; prefix entries charged separately.',
        'latency_scope':'Batch/concurrency1 CPU FP32; per-step p95 descriptive correlated tokens, never used as independent statistical samples.',
        'input_report_sha256':sha(directory/'report.json'),'raw_sha256':sha(directory/'trials.jsonl.gz'),'system_raw_sha256':sha(directory/'systems.jsonl.gz')}
    write(output/'summary.json',results)
    lines=['# Phase 5 measured CPU pilot','',results['stats_scope'],'',
        '| Policy | EM | F1 | KL | NLL | Decode ms | Active KV bytes | Total cache/index/scorer bytes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['arm']} | {r['mean_em']:.3f} | {r['mean_f1']:.3f} | {r['mean_kl']:.6f} | {r['mean_nll']:.3f} | {r['mean_decode_ms']:.2f} | {r['max_active_kv_bytes']} | {r['max_total_storage_bytes']} |")
    lines+=['','Pure window has different prompt protection. Full has an unbounded KV budget. Sinks+recent aliases protected recency. Extra archive and prefix memory is not an equal-total-memory comparison.',
        '', '## Dominated configurations (F1, CPU decode, total storage)','']
    for arm,other in pareto.items():
        if other:lines.append(f'- `{arm}`: dominated by '+', '.join(f'`{x}`' for x in other))
    lines+=['','## Reuse × importance × recovery','',
        '| Policy | Reuse | TTFT ms | End-to-end ms | Tokens/s | Extra prefix bytes |', '|---|---|---:|---:|---:|---:|']
    for r in factorial:lines.append(f"| {r['arm']} | {r['reuse']} | {r['mean_ttft_ms']:.2f} | {r['mean_end_to_end_ms']:.2f} | {r['mean_tokens_per_second']:.2f} | {r['max_prefix_storage_bytes']} |")
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Standard exported scientific plots, each derived directly from run artifacts.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for cost,title,filename in [('mean_decode_ms','CPU decode latency (ms)','quality-latency'),
                               ('max_active_kv_bytes','Active KV tensor bytes','quality-active-memory'),
                               ('max_total_storage_bytes','Total cache/index/scorer storage estimate (bytes)','quality-total-memory')]:
        fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
        for r in table:
            ax.scatter(r[cost],r['mean_f1']);ax.annotate(r['arm'],(r[cost],r['mean_f1']),fontsize=7,xytext=(3,3),textcoords='offset points')
        ax.set(xlabel=title,ylabel='Mean free-generation token F1',title='HTIS-Context Phase 5: eight short CPU cases')
        ax.grid(alpha=.2);fig.savefig(output/(filename+'.svg'));fig.savefig(output/(filename+'.png'),dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for arm in ['full','recent','htis','htis/recovery']:
        current=[r for r in core if r['arm']==arm];ax.scatter([r['input_tokens'] for r in current],[r['mean_decode_ms'] for r in current],label=arm)
    ax.set(xlabel='Complete input tokens',ylabel='Mean CPU decode latency (ms)',title='Observed contexts; workload and length are confounded')
    ax.legend();ax.grid(alpha=.2);fig.savefig(output/'latency-context.svg');fig.savefig(output/'latency-context.png',dpi=150);plt.close(fig)
    print(json.dumps({'trials':len(rows),'systems':len(systems),'core':table,'recovery_success':recovery,'factorial':factorial},indent=2))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    analyze(a.input,a.manifest,a.output)
