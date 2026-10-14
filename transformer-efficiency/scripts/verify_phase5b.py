"""Checkpoint audit; append Phase 5B evidence without rewriting earlier claims."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path.cwd();P=ROOT/'transformer-efficiency/results/phase5b'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def raw(p):return [json.loads(line) for line in gzip.open(p,'rt',encoding='utf-8')]


def main():
    start=read(P/'start-audit.json');ledger=read(ROOT/'CLAIM_EVIDENCE.json');baseline=start['baseline_ledger']
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==start['baseline_commit']
    assert ledger['claims'][:len(baseline['claims'])]==baseline['claims']
    assert ledger['artifacts'][:len(baseline['artifacts'])]==baseline['artifacts']
    for artifact in ledger['artifacts']:
        assert sha(ROOT/artifact['path'])==artifact['sha256'],artifact['path']
    for path,digest in start['baseline_source_sha256'].items():assert sha(Path(path))==digest,path
    manifest=ROOT/'transformer-efficiency/configs/phase5b-cpu-v1.json';m=read(manifest)
    assert m['protocol_sha256']==sha(P/'PREREGISTRATION.md')
    for path,digest in m['runtime_sha256'].items():assert sha(ROOT/path)==digest,path
    for path,digest in m['checkpoints_sha256'].items():assert sha(ROOT/path)==digest,path
    data=ROOT/'transformer-efficiency/data/phase5b-extended-v3/manifest.json'
    assert sha(data)==m['data_sha256']
    frozen=read(P/'frozen-manifest.json');assert frozen['manifest_sha256']==sha(manifest)
    calibration=read(P/'validation-v1/calibration.json')
    assert calibration['manifest_sha256']==sha(manifest)
    validation=read(P/'validation-v1/report.json');quality=read(P/'evaluation-v1/report.json')
    system=read(P/'systems-v1/report.json');analysis=read(P/'analysis-v1/summary.json')
    importance=read(P/'importance-v1/report.json');profile=read(P/'profile-v1/report.json')
    tests=read(P/'tests-final-v1.json');gpu=read(P/'gpu-availability-v1.json')
    assert all(r['passed'] for r in [validation,quality,system,analysis,importance,profile])
    assert tests['exit_code']==0 and not tests['source_modified_during_run']
    test_count=int(re.search(r'(\d+) passed',tests['stdout'])[1])
    for name in ['validation-command-v1.json','evaluation-command-v1.json','systems-command-v1.json','importance-command-v1.json','profile-command-v1.json']:
        cmd=read(P/name);assert cmd['exit_code']==0 and not cmd['source_modified_during_run'],name
    test_start=datetime.fromisoformat(read(P/'evaluation-command-v1.json')['started_at_utc'])
    system_start=datetime.fromisoformat(read(P/'systems-command-v1.json')['started_at_utc'])
    assert datetime.fromisoformat(m['frozen_at_utc'])<datetime.fromisoformat(read(P/'validation-command-v1.json')['started_at_utc'])
    assert datetime.fromisoformat(calibration['completed_at_utc'])<test_start<system_start
    rows=raw(P/'evaluation-v1/trials.jsonl.gz');requests=raw(P/'systems-v1/systems.jsonl.gz')
    assert len(rows)==quality['trials']==analysis['quality_trials']==490
    assert len(requests)==system['requests']==analysis['actual_system_requests']==320
    assert validation['trials']==26
    assert sha(P/'evaluation-v1/trials.jsonl.gz')==quality['raw_sha256']==analysis['source_sha256']['quality']
    assert sha(P/'systems-v1/systems.jsonl.gz')==system['raw_sha256']==analysis['source_sha256']['systems']
    expected={(case,name,budget,seed,recovery) for case in m['quality_cases'] for name in m['policies']
        for budget in [4,8,16] for seed in [17,29] for recovery in [False,True]}
    actual={(r['case'],r['policy'],r['budget'],r['seed'],r['recovery']) for r in rows if r['policy']!='full'}
    assert actual==expected and len(rows)==len(actual)+10
    assert all(r['output_tokens']<=r['generation_limit'] for r in rows)
    assert all(c['passed'] for c in quality['checks'])
    assert all(c['passed'] for r in quality['upstream_sdpa'] for c in r['checks'])
    assert len(system['equivalence'])==104 and all(c['passed'] for c in system['equivalence'])
    assert all(c['modified_prefix_missed'] for c in system['lifecycle'])
    assert all(c['passed'] for c in system['shared_prefix_checks'])
    assert gpu['executed'] is False and not gpu['cuda_available']
    previous=read(ROOT/'transformer-efficiency/data/phase5-short-v1/manifest.json')
    current=read(data);val={r['group'] for r in current['rows'] if r['split']=='validation'}
    heldout={r['group'] for r in current['rows'] if r['split']=='test'}
    assert val.isdisjoint(heldout) and (val|heldout).isdisjoint({r['group'] for r in previous['rows']})
    changes=subprocess.check_output(['git','diff','--name-only','HEAD'],text=True).splitlines()
    protected=['signalrank/','transformer-efficiency/src/htis/',*[f'transformer-efficiency/results/phase{i}/' for i in range(6)],
        'transformer-efficiency/data/phase3-public-v1/','transformer-efficiency/data/phase5-short-v1/']
    assert not any(p.startswith(tuple(protected)) for p in changes),'Prior research modified'
    if '5B' in ledger['phases']:
        print(json.dumps(dict(status=ledger['phases']['5B'],verified_artifact_hashes=len(ledger['artifacts']),
            tests=test_count,quality_trials=len(rows),requests=len(requests),repeat_audit_read_only=True),indent=2));return
    checkpoint=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),baseline_commit=start['baseline_commit'],
        status='VERIFIED COMPLETE - preregistered extended CPU reference scope',phase6='NOT STARTED',
        previous_artifacts_verified=len(baseline['artifacts']),tests_passed=test_count,validation_trials=26,
        heldout_quality_trials=len(rows),actual_system_requests=len(requests),budget_checks=sum(c.get('policy') is not None for c in quality['checks']),
        sdpa_checks=sum(len(r['checks']) for r in quality['upstream_sdpa']),prefix_paired_cells=len(system['equivalence']),
        prefix_compared_logits=sum(c['compared_steps'] for c in system['equivalence']),
        source_checkpoints_and_previous_claims_unchanged=True,manifest_sha256=sha(manifest),
        calibration_sha256=sha(P/'validation-v1/calibration.json'),contributions=analysis['contributions'],
        gpu='NOT EXECUTED - CPU-only Torch / no compatible local CUDA hardware',
        official_long_context_benchmark='Two complete LongBench v1 official-source cases executed; full benchmark NOT EXECUTED',
        allocator_attribution='NOT MEASURED - physical tensors measured, structural estimates and process-wide working set separately labeled',
        paid_compute_push_merge_deploy_reset_phase6=False,
        scope_limitations=analysis['limitations'])
    write(P/'checkpoint-verification.json',checkpoint)
    paths=[p for p in P.rglob('*') if p.is_file() and not any(part.startswith('pytest-temp') for part in p.parts)]
    paths+=list(data.parent.glob('*'))+[manifest]
    paths+=list((ROOT/'transformer-efficiency/src/htis_context').glob('phase5b*.py'))
    paths+=[ROOT/'transformer-efficiency/tests/test_phase5b.py',Path(__file__),ROOT/'transformer-efficiency/scripts/audit_phase5b_start.py']
    for path in sorted(set(paths)):
        ledger['artifacts'].append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size))
    ledger['phases']['5B']=checkpoint['status'];ledger['phase5b_scope']=dict(cpu=checkpoint['status'],gpu=checkpoint['gpu'],official_benchmark=checkpoint['official_long_context_benchmark'],phase6='NOT STARTED')
    evidence=['transformer-efficiency/results/phase5b/'+name for name in ['PREREGISTRATION.md','evaluation-v1/report.json','systems-v1/report.json','analysis-v1/summary.json','FINDINGS.md','checkpoint-verification.json']]
    for name,decision in analysis['contributions'].items():
        ledger['claims'].append(dict(id='P5B-'+name,claim={'A':'Frozen task-conditioned learned KV importance improves constrained-memory answer quality',
            'B':'Selective recovery improves answer quality enough to justify its measured cost',
            'C':'Coordinated exact prefix reuse and learned KV management improves beyond strong simple compositions'}[name],
            status=decision['status'],scope='Preregistered small extended CPU study; no global novelty, official benchmark or GPU claim',evidence=evidence,decision=decision))
    ledger['claims'].append(dict(id='P5B-REUSE',claim='Compatible exact prefix reuse preserves logits/outputs and saves prefill in the measured CPU workload',
        status='measured-extended-cpu',scope='104 paired cells / two sources / two repeats; additional prefix storage and admission costs separately charged',
        evidence=evidence,metrics=analysis['prefix_reuse_summary']))
    ledger['continuations'].append(dict(baseline_commit=start['baseline_commit'],selected_phase='5B',status=checkpoint['status'],
        prior_artifacts_reused=len(baseline['artifacts']),next_unfinished_phase='Phase 6 remains NOT STARTED; broader GPU/official benchmark scope remains unexecuted',
        paid_compute=False,deployment_or_push=False,scope='Extended supported-context inputs,16/32-token generation; matched4/8/16 pages; adapted public QA/authored sessions and two complete LongBench-source cases, one model/CPU'))
    write(ROOT/'CLAIM_EVIDENCE.json',ledger)
    print(json.dumps(dict(status=checkpoint['status'],tests=test_count,quality_trials=len(rows),actual_requests=len(requests),
        previous_hashes=len(baseline['artifacts']),total_artifact_hashes=len(ledger['artifacts'])),indent=2))


if __name__=='__main__':main()
