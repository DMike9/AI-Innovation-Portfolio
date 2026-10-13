"""Derive inspectable events, failures, bound diagnostics from recorded raw trials.

No fitting, threshold tuning or edits to the frozen comparison artifact.
"""
import gzip
import hashlib
import json
from pathlib import Path
import sys


def main():
    directory=Path(sys.argv[1])
    report=json.loads((directory/'report.json').read_text())
    raw=directory/report['trials_file'] if 'trials_file' in report else directory/'report.trials.jsonl.gz'
    trials=[json.loads(line) for line in gzip.open(raw,'rt',encoding='utf-8')]
    output=directory/'diagnostics'
    output.mkdir(exist_ok=False)
    churn={};bound_rows=[];misses=0;fetches=0;unmet=0;event_count=0
    controller_count=0;lookup_hits=0
    with gzip.open(output/'recovery-events.jsonl.gz','wt',encoding='utf-8') as f, gzip.open(output/'controller-decisions.jsonl.gz','wt',encoding='utf-8') as decisions:
        for trial in trials:
            last_loaded={};reloads=0;rapid=0;used=0
            for step in trial['steps']:
                for d in step['recovery'].get('decisions',[]):
                    row={k:trial[k] for k in ['workload','trace','policy','repeat']}
                    row.update(step=step['step'],**d)
                    decisions.write(json.dumps(row)+'\n');controller_count+=1
                for e in step['recovery'].get('events',[]):
                    row={k:trial[k] for k in ['workload','trace','policy','repeat']}
                    row.update(step=step['step'],**e)
                    f.write(json.dumps(row)+'\n');event_count+=1
                    key=e.get('layer'),e.get('page')
                    lookup_hits+=int(e['action'] in ['reload','fallback-read'])
                    if e['action']=='reload':
                        reloads+=1;fetches+=1
                        rapid+=int(key in last_loaded and step['step']-last_loaded[key]<=2)
                        last_loaded[key]=step['step']
                        used+=int(e['page'] in step['active_pages_per_layer'][e['layer']])
                for b in step['recovery'].get('bounds',[]):
                    unmet+=int(not b['tolerance_met'])
                    misses+=int(b.get('fallback')=='unavailable')
                    if 'measured_current_state_error_per_head' in b:
                        for h,(upper,error) in enumerate(zip(b['per_head'],b['measured_current_state_error_per_head'])):
                            bound_rows.append({'workload':trial['workload'],'policy':trial['policy'],'repeat':trial['repeat'],
                                'step':step['step'],'layer':b['layer'],'query_head':h,'bound':upper,'actual_local_l2':error,
                                'slack':upper-error,'ratio_if_positive':upper/error if error>1e-12 else None})
            churn.setdefault(trial['policy'],[]).append({'workload':trial['workload'],'repeat':trial['repeat'],
                'reloads':reloads,'reload_repeated_within_two_steps':rapid,'reload_still_retained_after_adaptation':used,
                'note':'The last measure is a residency diagnostic, not whether the page was semantically useful.'})
    with gzip.open(output/'bound-tightness.jsonl.gz','wt',encoding='utf-8') as f:
        for row in bound_rows:f.write(json.dumps(row)+'\n')
    ratios=sorted(r['ratio_if_positive'] for r in bound_rows if r['ratio_if_positive'] is not None)
    failures=[]
    for comparison in report['paired_comparisons']:
        reasons=[]
        if comparison['recency_recovery_delta']>0:reasons.append('recency recovery increased measured KL')
        if comparison['htis_recovery_delta']>0:reasons.append('HTIS recovery increased measured KL')
        if comparison['htis_vs_recency_recovery_delta']>0:reasons.append('HTIS recovery worse than simple recovery')
        if reasons:failures.append({**comparison,'reasons':reasons})
    diagnostic={'source_report_sha256':hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
        'raw_trials':len(trials),'event_rows':event_count,'bound_head_comparisons':len(bound_rows),
        'controller_decision_rows':controller_count,'archive_indexed_lookup_hits':lookup_hits,
        'archive_lookup_misses':0,'lookup_scope':'Runtime lookups request only pages present in the archive index; unavailable fallback is counted separately.',
        'bound_coverage_violations':sum(r['actual_local_l2']>r['bound']+1e-6 for r in bound_rows),
        'bound_ratio_median':ratios[len(ratios)//2] if ratios else None,'bound_ratio_max':max(ratios) if ratios else None,
        'fallback_unavailable_layer_events':misses,'tolerance_unmet_layer_events':unmet,
        'reloads':fetches,'churn':churn,'negative_quality_examples':failures,
        'interpretation':'Fixed-state bound diagnostics are not final-answer correctness or historical dense repair.',
        'files':[{ 'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
                 for p in sorted(output.glob('*.gz'))]}
    (output/'summary.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    print(json.dumps({k:v for k,v in diagnostic.items() if k not in ['churn','negative_quality_examples','files']},indent=2))


if __name__=='__main__':main()
