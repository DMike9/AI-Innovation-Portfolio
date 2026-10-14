"""Checkpoint existing files only. Never imports or executes model/test code."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'transformer-efficiency/results/phase5b'
OUT = BASE / 'paused-checkpoint-v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def key(row):
    return {name: row[name] for name in
            ('case', 'policy', 'budget', 'seed', 'recovery', 'repeat', 'traffic')}


def rows(path, fields):
    completed = []
    error = None
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                completed.append({name: row[name] for name in fields})
    except (EOFError, OSError, UnicodeError, ValueError) as caught:
        error = f'{type(caught).__name__}: {caught}'
    return completed, error


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing checkpoint')
    OUT.mkdir(exist_ok=True)
    config_path = ROOT / 'transformer-efficiency/configs/phase5b-cpu-v1.json'
    manifest = read(config_path)
    cfg = manifest['configuration']
    data = read(ROOT / 'transformer-efficiency/data/phase5b-extended-v3/manifest.json')
    quality, quality_error = rows(BASE / 'evaluation-v1/trials.jsonl.gz',
        ('case', 'policy', 'budget', 'seed', 'recovery'))
    systems, systems_error = rows(BASE / 'systems-v1/systems.jsonl.gz',
        ('case', 'policy', 'budget', 'seed', 'recovery', 'repeat', 'traffic'))
    pairs, pairs_error = rows(BASE / 'systems-v1/equivalence.jsonl.gz',
        ('case', 'policy', 'budget', 'seed', 'recovery', 'repeat', 'passed', 'compared_steps'))
    validation, validation_error = rows(BASE / 'validation-v1/trials.jsonl.gz',
        ('case', 'policy', 'budget', 'seed', 'recovery'))
    expected = []
    for row in data['rows']:
        if row['id'] not in manifest['systems_cases']:
            continue
        cells = [(policy, budget, recovery) for policy in manifest['policies']
                 for budget in cfg['budgets'] for recovery in (False, True)]
        cells += [('full', 4, recovery) for recovery in (False, True)]
        random.Random(901 + row['input_tokens']).shuffle(cells)
        for policy, budget, recovery in cells:
            for repeat in range(cfg['timing_repeats']):
                seed = cfg['seeds'][repeat % len(cfg['seeds'])]
                traffic = ['cold-admission'] + (['cold-no-cache', 'warm-repeat']
                          if repeat % 2 == 0 else ['warm-repeat', 'cold-no-cache'])
                if policy == 'full' and not recovery:
                    traffic.append('modified-prefix')
                for name in traffic:
                    expected.append(dict(case=row['id'], policy=policy,
                        budget=None if policy == 'full' else budget, seed=seed,
                        recovery=recovery, repeat=repeat, traffic=name))
    for row in data['rows']:
        if row['group'] == 'session5b/switch':
            for traffic in ('shared-prefix-task-change', 'shared-prefix-cold-control'):
                expected.append(dict(case=row['id'], policy='query', budget=8,
                    seed=17, recovery=False, repeat=0, traffic=traffic))
    assert len(expected) == 320
    canonical = lambda value: json.dumps(value, sort_keys=True)
    seen = {canonical(value) for value in systems}
    assert len(seen) == len(systems), 'Duplicate saved request identifiers'
    remaining = [value for value in expected if canonical(value) not in seen]
    assert all(canonical(value) in {canonical(x) for x in expected} for value in systems)
    inventory = []
    excluded = []
    paths = []
    for parent, directories, filenames in os.walk(BASE):
        for name in list(directories):
            if name.startswith('pytest-temp') or Path(parent) / name == OUT:
                directories.remove(name)
                excluded.append(str((Path(parent) / name).relative_to(ROOT)))
        paths.extend(Path(parent) / name for name in filenames)
    for folder in ('src/htis_context', 'scripts', 'tests', 'configs'):
        paths.extend((ROOT / 'transformer-efficiency' / folder).glob('*phase5b*'))
    paths.append(ROOT / 'transformer-efficiency/tests/conftest.py')
    for version in ('v1', 'v2', 'v3'):
        paths.extend((ROOT / f'transformer-efficiency/data/phase5b-extended-{version}').rglob('*'))
    paths = sorted(set(path for path in paths if path.is_file()))
    invalid_json = []
    for path in paths:
        entry = dict(path=str(path.relative_to(ROOT)).replace('\\', '/'),
                     bytes=path.stat().st_size, sha256=sha(path))
        if path.suffix == '.json':
            try:
                read(path)
                entry['json_valid'] = True
            except (ValueError, UnicodeError) as error:
                entry['json_valid'] = False
                invalid_json.append(dict(path=entry['path'], error=str(error)))
        inventory.append(entry)
    report = read(BASE / 'evaluation-v1/report.json')
    raw_matches = sha(BASE / 'evaluation-v1/trials.jsonl.gz') == report['raw_sha256']
    checks = report['checks']
    record = dict(status='PARTIALLY EXECUTED / PAUSED',
        paused_at_utc=datetime.now(timezone.utc).isoformat(),
        reason='User explicitly stopped research execution; checkpoint only; no automatic continuation.',
        base_commit='2c37eaf6333c11e114246395390e2d778d94ff45',
        configuration=manifest, configuration_sha256=sha(config_path),
        calibration=read(BASE / 'validation-v1/calibration.json'),
        calibration_sha256=sha(BASE / 'validation-v1/calibration.json'),
        counts=dict(validation_trials=len(validation), quality_trials=len(quality),
                    quality_checks=len(checks), quality_checks_passed=all(x['passed'] for x in checks),
                    sdpa_checks=sum(len(x.get('checks', [])) for x in report['upstream_sdpa']),
                    systems_requests=len(systems), verified_paired_cells=len(pairs),
                    paired_cells_passed=all(x['passed'] for x in pairs),
                    remaining_systems_requests=len(remaining), planned_systems_requests=320,
                    planned_paired_cells=104),
        integrity=dict(quality_raw_hash_matches_report=raw_matches,
                       gzip_errors=dict(quality=quality_error, systems=systems_error,
                                        equivalence=pairs_error, validation=validation_error),
                       invalid_json=invalid_json, inventory_files=len(inventory), excluded_directories=excluded),
        keep_awake=read(BASE / 'keep-awake-record-v1.json'),
        interrupted_command=read(BASE / 'systems-command-v1.json'),
        not_completed=['Remaining systems requests and complete lifecycle/shared-prefix controls',
                       'Profiling, importance and budget-rank diagnostics',
                       'Exploratory zero-archive and eight-hit workload controls',
                       'Memory accounting audit, final paired statistics/Pareto analysis and figures',
                       'Final acceptance verification; GPU unavailable; Phase 6 not started'],
        resume_constraint='Original systems runner has no resume support. Use remaining-systems-requests.json as an exact skip-list contract for a continuation runner in a new output directory. Do not rerun validation, test, or completed systems requests. In-memory cache/logits/lifecycle state is not persisted.',
        no_models_tests_or_analysis_executed_by_checkpoint_script=True)
    write('completed-quality-trials.json', quality)
    write('completed-validation-trials.json', validation)
    write('completed-systems-requests.json', systems)
    write('verified-paired-cells.json', pairs)
    write('remaining-systems-requests.json', remaining)
    write('artifact-inventory.json', inventory)
    write('checkpoint.json', record)
    print(json.dumps(dict(counts=record['counts'], integrity=record['integrity'],
                         keep_awake_active=record['keep_awake']['active']), indent=2))


if __name__ == '__main__':
    main()
