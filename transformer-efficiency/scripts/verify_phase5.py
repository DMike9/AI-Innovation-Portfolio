"""Verify executed Phase 5 CPU scope; append evidence without altering Phases 0–4."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


ROOT=Path.cwd()
BASE='727c91c1a9fa0371d689802c0bdb5e1f7827bfde'
P5=ROOT/'transformer-efficiency/results/phase5'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def raw(p):return [json.loads(line) for line in gzip.open(p,'rt',encoding='utf-8')]


def main():
    ledger=read(ROOT/'CLAIM_EVIDENCE.json')
    baseline=json.loads(subprocess.check_output(['git','show',BASE+':CLAIM_EVIDENCE.json'],text=True))
    for artifact in ledger['artifacts']:
        assert sha(ROOT/artifact['path'])==artifact['sha256'],artifact['path']
    assert ledger['claims'][:len(baseline['claims'])]==baseline['claims'],'Earlier claims changed'
    assert ledger['artifacts'][:len(baseline['artifacts'])]==baseline['artifacts'],'Earlier artifact ledger changed'
    report=read(P5/'evaluation-v1/report.json');analysis=read(P5/'analysis-v2/summary.json')
    tests=read(P5/'tests-final-v2.json');command=read(P5/'evaluation-command-v1.json')
    validation=read(P5/'validation-v2/report.json');selection=read(P5/'validation-v2/calibration.json')
    manifest=ROOT/'transformer-efficiency/configs/phase5-cpu-v2.json'
    config=read(manifest);data=read(ROOT/'transformer-efficiency/data/phase5-short-v1/manifest.json')
    assert report['passed'] and validation['passed'] and analysis['passed']
    assert tests['exit_code']==0 and '186 passed' in tests['stdout']
    assert command['exit_code']==0 and not command['source_modified_during_run']
    assert not tests['source_modified_during_run']
    assert selection['experiment_manifest_sha256']==sha(manifest)==report['manifest_sha256']
    started=datetime.fromisoformat(command['started_at_utc'])
    val_command=read(P5/'validation-command-v2.json')
    val_started=datetime.fromisoformat(val_command['started_at_utc'])
    assert val_started.timestamp()+val_command['wall_seconds']<started.timestamp(),'Calibration not completed before held-out run'
    assert val_command['exit_code']==0 and not val_command['source_modified_during_run']
    # Earlier val execution differed only in prepared GPU-runner source, which
    # was not imported by the CPU experiment. Critical numerical code matches.
    for name,h in command['source_files_sha256'].items():
        if name.startswith(('src/','tests/','configs/')) or name in {'pyproject.toml','scripts/run_recorded.py'}:
            assert sha(ROOT/'transformer-efficiency'/name)==h,name
    for name in ['src/htis_context/phase5.py','src/htis_context/phase5_policy.py',
                 'src/htis_context/adapter.py','src/htis_context/learned.py','src/htis_context/recovery.py',
                 'src/htis_context/pages.py','src/htis_context/runtime_features.py']:
        assert val_command['source_files_sha256'][name]==command['source_files_sha256'][name],name
    rows=raw(P5/'evaluation-v1/trials.jsonl.gz');systems=raw(P5/'evaluation-v1/systems.jsonl.gz')
    assert len(rows)==report['trials']==analysis['trials']
    assert len(systems)==32==analysis['actual_system_requests']
    assert sha(P5/'evaluation-v1/trials.jsonl.gz')==report['raw_trials_sha256']==analysis['raw_sha256']
    assert all(len(r['output_ids'])==6 for r in rows)
    test_ids={r['id'] for r in data['rows'] if r['split']=='test'}
    val_ids={r['id'] for r in data['rows'] if r['split']=='validation'}
    assert test_ids.isdisjoint(val_ids) and set(selection['validation_case_ids'])==val_ids
    assert len(test_ids)==8 and len(val_ids)==4
    for arm in config['core_arms']:
        assert {r['case'] for r in rows if r['arm']==arm}==test_ids,arm
    for arm in config['ablation_arms']:
        assert {r['case'] for r in rows if r['arm']==arm}==set(config['ablation_cases']),arm
    assert all(c['passed'] for c in report['checks'])
    assert all(c['passed'] for r in report['upstream_sdpa'] for c in r['checks'])
    for case in config['systems_cases']:
        for arm in config['factorial_arms']:
            current=[s for s in systems if s['case']==case and s['arm']==arm]
            assert len(current)==4 and all(s['output_ids']==current[0]['output_ids'] for s in current)
    gpu=read(P5/'gpu-availability-v1.json')
    assert gpu['executed'] is False and not gpu['cuda_available']
    changed=subprocess.check_output(['git','diff','--name-only','HEAD'],text=True).splitlines()
    changed+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines()
    protected=['signalrank/','src/','transformer-efficiency/src/htis/',
               *[f'transformer-efficiency/results/phase{i}/' for i in range(5)],
               'transformer-efficiency/data/phase3-public-v1/']
    assert not any(p.startswith(tuple(protected)) for p in changed),'Earlier work changed'
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
    if any(c['selected_phase']==5 for c in ledger['continuations']):
        print(json.dumps({'status':ledger['phases']['5'],'verified_artifacts':len(ledger['artifacts']),
            'trials':len(rows),'system_requests':len(systems),'prior_claims_unchanged':True,'repeat_audit_read_only':True},indent=2))
        return
    checkpoint={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'baseline_commit':BASE,'selected_phase':5,
        'status':'VERIFIED COMPLETE - short CPU pilot only','prior_artifact_hashes_verified':len(baseline['artifacts']),
        'test_count':186,'validation_trials':validation['trials'],'heldout_quality_trials':len(rows),
        'system_requests':len(systems),'runtime_checks':len(report['checks']),'sdpa_equivalence_checks':sum(len(r['checks']) for r in report['upstream_sdpa']),
        'changed_files':sorted(set(changed+['transformer-efficiency/results/phase5/checkpoint-verification.json'])),
        'verification_command':'.venv/Scripts/python.exe transformer-efficiency/scripts/verify_phase5.py',
        'prior_claims_sources_weights_evidence_unchanged':True,
        'benchmark_source_unchanged_during_run':True,'execution_identity':report['execution_identity'],
        'gpu_scope':'BLOCKED / NOT EXECUTED; dry-run and CPU synchronization/gating unit tests only',
        'long_context_official_benchmarks':'NOT EXECUTED','second_model_family':'NOT EXECUTED','phase6':'NOT STARTED',
        'paid_compute_commit_push_merge_deploy':False,'next_unfinished_phase':'5 extension before Phase 6',
        'limitations':report['limitations']}
    write(P5/'checkpoint-verification.json',checkpoint)
    paths=[p for p in P5.rglob('*') if p.is_file()]
    paths+=list((ROOT/'transformer-efficiency/data/phase5-short-v1').glob('*'))
    paths+=list((ROOT/'transformer-efficiency/configs').glob('phase5-*.json'))
    paths.append(ROOT/'transformer-efficiency/requirements-phase5.lock')
    for p in sorted(paths):
        if p.suffix not in {'.json','.gz','.pstats','.png','.svg','.md','.lock'}:continue
        ledger['artifacts'].append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
    ledger['phases']['5']='VERIFIED COMPLETE (short CPU pilot only)'
    ledger['phase5_scope']={'cpu_pilot':'VERIFIED COMPLETE','gpu_study':'BLOCKED / NOT EXECUTED',
        'official_long_context_study':'NOT EXECUTED','cross_family_generalization':'NOT EXECUTED','phase6':'NOT STARTED'}
    ledger['phase5_execution_identity']=report['execution_identity']
    ledger['claims'] += [
        {'id':'C018','claim':'Existing pretrained reference bottlenecks were profiled before Phase 5 controller experiments',
         'status':'measured-short-cpu-profile','scope':'Five existing policies on one validation prefix; profiling overhead separated from plain timings, no GPU or production speed claim',
         'evidence':['transformer-efficiency/results/phase5/profile-v1/report.json','transformer-efficiency/results/phase5/profile-v1/PROFILING.md']},
        {'id':'C019','claim':'Pretrained free generation, divergence, ablations and uncertainty were measured in a controlled short CPU pilot',
         'status':'measured-small-scope','scope':analysis['stats_scope'],
         'evidence':['transformer-efficiency/results/phase5/evaluation-v1/report.json','transformer-efficiency/results/phase5/evaluation-v1/trials.jsonl.gz',
                     'transformer-efficiency/results/phase5/analysis-v2/summary.json','transformer-efficiency/results/phase5/tests-final-v2.json'],
         'metrics':analysis['core'],'dominated_by':analysis['pareto_dominated_by']},
        {'id':'C020','claim':'Exact reuse crossed with learned retention and recovery was measured on actual cold/warm CPU requests',
         'status':'measured-tiny-factorial','scope':'Two predeclared cases, two repeats per cell, batch/concurrency1, fixed validation-selected simple baseline; extra prefix/archive storage charged separately',
         'evidence':['transformer-efficiency/results/phase5/evaluation-v1/systems.jsonl.gz','transformer-efficiency/results/phase5/analysis-v2/summary.json'],
         'metrics':analysis['factorial']},
        {'id':'C021','claim':'CUDA runner prepared with synchronization, equivalence gates, repeat timing and memory counters',
         'status':'prepared-not-gpu-validated','scope':'Dry-run/gating/synchronization CPU tests passed. Actual CUDA correctness, timing, transfer and allocation paths NOT EXECUTED; CPU archive recovery not supported on GPU',
         'evidence':['transformer-efficiency/results/phase5/gpu-availability-v1.json','transformer-efficiency/results/phase5/tests-final-v2.json']}
    ]
    ledger['continuations'].append({'baseline_commit':BASE,'selected_phase':5,'status':'VERIFIED COMPLETE - short CPU pilot only',
        'prior_artifacts_reused':len(baseline['artifacts']),'next_unfinished_phase':'5 extension','paid_compute':False,'deployment_or_push':False,
        'scope':'Short FP32 CPU model-quality/systems pilot; GPU/long-context/generalization not executed; Phase 6 not started'})
    ledger['recorded_at_utc']=checkpoint['recorded_at_utc']
    write(ROOT/'CLAIM_EVIDENCE.json',ledger)
    print(json.dumps({'status':checkpoint['status'],'prior_hashes_verified':len(baseline['artifacts']),
        'total_artifacts':len(ledger['artifacts']),'quality_trials':len(rows),'requests':len(systems),'runtime_checks':len(report['checks'])},indent=2))


if __name__=='__main__':main()
