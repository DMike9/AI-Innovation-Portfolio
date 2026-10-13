"""Final analysis: matched checkpoint seeds, clear figures, alternative Pareto metrics."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import statistics
from htis_context import phase5_analysis as base
from htis_context.phase5_data import write
from htis_context.phase5_metrics import paired_bootstrap,dominated,answer_metrics


DETERMINISTIC={'recent','history','query','full','window','sinks_recent','calibrated'}


def matched_paired(rows,arm,reference,metric):
    left=defaultdict(dict);right=defaultdict(dict);records={}
    for r in rows:
        records[r['case']]=r
        if r['arm']==arm:left[r['case']][r['seed']]=r[metric]
        if r['arm']==reference:right[r['case']][r['seed']]=r[metric]
    if set(left)!=set(right):raise ValueError('Pair analysis needs identical cases')
    pairs=[];used=[]
    for case in sorted(left):
        for seed,value in left[case].items():
            if seed in right[case]:target=right[case][seed]
            elif reference in DETERMINISTIC and len(set(right[case].values()))==1:target=next(iter(right[case].values()))
            else:continue
            pairs.append((base.group(records[case]),value-target));used.append((case,seed))
        if not any(k==case for k,_ in used):raise ValueError('No matching seed for case')
    result=paired_bootstrap(pairs)
    result.update(matched_case_seed_pairs=used,
        seed_method='Shared stochastic/trained checkpoint seeds only; deterministic simple reference can be held fixed across seeds.')
    return result


def figures(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=result['core'];colors=plt.get_cmap('tab20').colors
    def scatter(x,y,xlabel,ylabel,name,sdpa=False):
        fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
        for i,r in enumerate(rows):
            ax.scatter(r[x],r[y],color=colors[i],marker='x' if r['arm']=='window' else 'o',
                s=55,label=r['arm']+(' (different protection)' if r['arm']=='window' else ''))
        if sdpa:
            r=result['upstream_sdpa'];ax.scatter(r['mean_decode_ms'],r['mean_f1'],marker='*',s=150,color='black',label='upstream SDPA (full KV)')
        ax.set(xlabel=xlabel,ylabel=ylabel,title='Phase 5: eight short CPU cases; exploratory means')
        ax.grid(alpha=.2);ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8,frameon=False)
        fig.savefig(output/(name+'.svg'));fig.savefig(output/(name+'.png'),dpi=150);plt.close(fig)
    scatter('mean_decode_ms','mean_f1','Mean CPU decode latency (ms)','Mean free-generation token F1','quality-latency',True)
    scatter('max_active_kv_bytes','mean_f1','Maximum active KV tensor bytes','Mean free-generation token F1','quality-active-memory')
    scatter('max_total_storage_bytes','mean_f1','Maximum cache/index/scorer storage estimate (bytes)','Mean free-generation token F1','quality-total-memory')
    scatter('mean_decode_ms','mean_nll','Mean CPU decode latency (ms)','Mean teacher-forced answer NLL (lower is better)','nll-latency')


def analyze(directory,manifest,output):
    original=base.paired
    try:
        base.paired=matched_paired
        result=base.analyze(directory,manifest,output)
    finally:base.paired=original
    rows=base.read_raw(directory/'trials.jsonl.gz');systems=base.read_raw(directory/'systems.jsonl.gz')
    selection=json.loads(Path('transformer-efficiency/results/phase5/validation-v2/calibration.json').read_text())
    result['paired_vs_validation_selected_simple']={key:matched_paired(rows,'htis',selection['selected_simple_arm'],key)
        for key in ['f1','mean_kl','mean_nll','mean_decode_ms']}
    # Task F1, likelihood and divergence are distinct outcomes, with distinct frontiers.
    for name,key in [('nll','mean_nll'),('kl','mean_kl')]:
        table=[{**r,'quality':-r[key]} for r in result['core'] if r['arm']!='window']
        result['pareto_'+name+'_dominated_by']=dominated(table,quality='quality')
    data=json.loads(Path('transformer-efficiency/data/phase5-short-v1/manifest.json').read_text())
    answers={r['id']:r['answers'] for r in data['rows']}
    factorial_quality=[]
    for arm in json.loads(manifest.read_text())['factorial_arms']:
        for reuse in (False,True):
            current=[s for s in systems if s['arm']==arm and s['reuse']==reuse]
            metrics=[answer_metrics(s['raw_output'],answers[s['case']]) for s in current]
            factorial_quality.append({'arm':arm,'reuse':reuse,'mean_f1':statistics.mean(m['f1'] for m in metrics),
                'mean_em':statistics.mean(m['em'] for m in metrics),'requests':len(current)})
    result['factorial_free_quality']=factorial_quality
    result['analysis_notes']=['Supersedes initial analysis-v1 for matched-seed ablation comparisons and figure readability; inference artifacts unchanged.',
        'Core means weight eight cases equally. Bootstrap first averages within six source/session clusters, then weights clusters equally.',
        'Shared checkpoint seed comparisons prevent a single-seed controller/cache/budget ablation from being confounded with seed29.',
        'Conditional scoring_calls counts calls to trained-policy select, including its early-return path; it is not a count of network evaluations.',
        'Dominance is descriptive and metric-specific, not statistically established. F1, NLL and KL frontiers must be distinguished.',
        'Pure window excluded from matched-protection dominance. Aliases may differ in latency due to noise; no mechanism claim from alias timing.',
        'SDPA has measured quality/timing and passed common-token checks; total cache/index memory was not instrumented in that untouched upstream path.']
    write(output/'summary.json',result)
    figures(result,output)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    analyze(a.input,a.manifest,a.output)
