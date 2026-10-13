"""Verify Phase 4 artifacts and the preserved baseline; update the existing ledger.

Execute only after the recorded comparison and final correctness suite pass.
"""
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path.cwd(); directory=root/'transformer-efficiency/results/phase4'
    ledger=json.loads((root/'CLAIM_EVIDENCE.json').read_text())
    prior=list(ledger['artifacts'])
    for a in prior:assert sha(root/a['path'])==a['sha256'],a['path']
    report=json.loads((directory/'evaluation-v3/report.json').read_text())
    tests=json.loads((directory/'tests-final-current.json').read_text())
    command=json.loads((directory/'evaluate-command-v3.json').read_text())
    assert report['passed'] and all(c['passed'] for c in report['checks'])
    assert tests['exit_code']==0 and command['exit_code']==0
    assert not command['source_modified_during_run']
    trials=list(gzip.open(directory/'evaluation-v3'/report['raw_trials_artifact'],'rt',encoding='utf-8'))
    assert len(trials)==report['raw_trial_count']
    assert sha(directory/'evaluation-v3'/report['raw_trials_artifact'])==report['raw_trials_sha256']
    partial=list(gzip.open(directory/'evaluation-v3/partial-trials.jsonl.gz','rt',encoding='utf-8'))
    assert [json.loads(x) for x in partial]==[json.loads(x) for x in trials]
    assert not tests['source_modified_during_run']
    for name,h in command['source_files_sha256'].items():
        # Analysis/audit utilities may be repaired after a measured run. Keep
        # inference, tests, configuration, packaging and recorder checks strict.
        if name.startswith(('src/','tests/','configs/')) or name in {'pyproject.toml','scripts/run_recorded.py'}:
            assert sha(root/'transformer-efficiency'/name)==h,name
    changed=subprocess.check_output(['git','diff','--name-only','HEAD'],text=True).splitlines()
    changed+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines()
    protected=['signalrank/','src/','transformer-efficiency/src/htis/','transformer-efficiency/results/phase0/',
               'transformer-efficiency/results/phase1/','transformer-efficiency/results/phase2/','transformer-efficiency/results/phase3/',
               'transformer-efficiency/data/phase3-public-v1/']
    assert not any(p.startswith(tuple(protected)) for p in changed)
    diagnostics=json.loads((directory/'evaluation-v3/diagnostics/summary.json').read_text())
    assert diagnostics['bound_coverage_violations']==0
    if any(c.get('selected_phase')==4 and c.get('status')=='VERIFIED COMPLETE' for c in ledger['continuations']):
        # Repeated continuation audits are read-only; do not duplicate artifacts,
        # rewrite original timestamps/baselines, or change historical evidence.
        assert ledger['phases']['4']=='VERIFIED COMPLETE'
        print(json.dumps({'verified_artifacts':len(prior),'runtime_checks':len(report['checks']),
            'raw_trials':len(trials),'status':'VERIFIED COMPLETE - read-only repeat audit'},indent=2))
        return
    checkpoint={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'selected_phase':4,'status':'VERIFIED COMPLETE - small CPU reference scope','prior_artifact_hashes_verified':len(prior),
        'runtime_checks':len(report['checks']),'raw_trials':len(trials),'test_stdout':tests['stdout'],
        'changed_files':sorted(set(changed)),'protected_sources_and_evidence_unchanged':True,
        'dependencies':{n:version(n) for n in ['torch','transformers','numpy','pytest','safetensors']},
        'benchmark_source_unchanged_during_run':True,'gpu_measurements':False,'paid_compute':False,
        'push_merge_deploy_commit':False,'next_unfinished_phase':5,
        'limitations':report['limitations']}
    (directory/'checkpoint-verification.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
    for p in sorted(directory.rglob('*')):
        if not p.is_file() or any(x.startswith('pytest-temp') for x in p.parts):continue
        if p.suffix not in {'.json','.gz'}:continue
        path=p.relative_to(root).as_posix()
        ledger['artifacts'].append({'path':path,'sha256':sha(p),'bytes':p.stat().st_size})
    config=root/'transformer-efficiency/configs/phase4-small.json'
    ledger['artifacts'].append({'path':config.relative_to(root).as_posix(),'sha256':sha(config),'bytes':config.stat().st_size})
    compat=root/'transformer-efficiency/src/htis_context/runtime_compatibility.json'
    ledger['artifacts'].append({'path':compat.relative_to(root).as_posix(),'sha256':sha(compat),'bytes':compat.stat().st_size})
    ledger['phases']['4']='VERIFIED COMPLETE'
    ledger['phase4_execution_identity']=report['execution_identity']
    c=next(c for c in ledger['claims'] if c['id']=='C010')
    c.update(status='implemented-and-verified-small-cpu-study',scope='Bounded CPU request archive; current-query routing; optional complete-coverage local bound and exact-current-state blockwise fallback, never historical dense repair',
        evidence=['transformer-efficiency/results/phase4/tests-final-current.json','transformer-efficiency/results/phase4/evaluation-v3/report.json',
                  'transformer-efficiency/results/phase4/evaluation-v3/report.trials.jsonl.gz','transformer-efficiency/results/phase4/checkpoint-verification.json'])
    ledger['claims'].append({'id':'C017','claim':'Phase 4 recovery and HTIS costs and quality were measured against strong simple baselines',
        'status':'measured-small-study-with-negative-results','scope':'Nine authored/public-source held-out workloads, two repeats, equal active budget, CPU FP32 eager; no trained-weight updates or official benchmark/answer/GPU claim',
        'evidence':['transformer-efficiency/results/phase4/evaluation-v3/report.json','transformer-efficiency/results/phase4/evaluation-v3/diagnostics/summary.json'],
        'metrics':report['aggregate']})
    ledger['continuations'].append({'baseline_commit':checkpoint['baseline_commit'],'selected_phase':4,'status':'VERIFIED COMPLETE',
        'prior_artifacts_reused':len(prior),'next_unfinished_phase':5,'paid_compute':False,'deployment_or_push':False,
        'scope':'Small CPU recoverable architecture validation; not a proof of learned-policy superiority or hardware efficiency'})
    ledger['recorded_at_utc']=checkpoint['recorded_at_utc']
    (root/'CLAIM_EVIDENCE.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps({'verified_prior':len(prior),'total_hashed_artifacts':len(ledger['artifacts']),
                      'runtime_checks':len(report['checks']),'raw_trials':len(trials),'status':checkpoint['status']},indent=2))


if __name__=='__main__':main()
