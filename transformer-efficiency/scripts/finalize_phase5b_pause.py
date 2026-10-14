"""Save pause documentation and additive evidence; stdlib/file operations only."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'transformer-efficiency/results/phase5b'
OUT = BASE / 'paused-checkpoint-v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    ledger_path = ROOT / 'CLAIM_EVIDENCE.json'
    prior = read(ledger_path)
    if '5B' in prior['phases']:
        raise ValueError('Pause already recorded; refusing duplicate mutation')
    (OUT / 'prior-claim-evidence.json').write_bytes(ledger_path.read_bytes())
    record = read(OUT / 'checkpoint.json')
    config = record['configuration']
    frozen_checks = []
    for name, expected in config['runtime_sha256'].items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        frozen_checks.append(dict(path=name, passed=sha(path) == expected))
    assert all(item['passed'] for item in frozen_checks)
    helper = ROOT / 'transformer-efficiency/scripts/resume_phase5b_systems.py'
    ast.parse(helper.read_text(encoding='utf-8'))
    stamp = datetime.now(timezone.utc).isoformat()
    write(OUT / 'stop-record.json', dict(recorded_at_utc=stamp,
        user_instruction='STOP Phase 5B execution; checkpoint only; do not continue automatically',
        identified_model_processes=[dict(pid=39088, parent_pid=9956), dict(pid=24160, parent_pid=39088)],
        stop_action='Stop-Process -Force, guarded by exact htis_context.phase5b systems command-line match',
        research_python_process_inventory_after_stop=[],
        process_inventory_query='Get-CimInstance Win32_Process -Filter "Name LIKE \'%python%\'"',
        keep_awake_released=not record['keep_awake']['active'],
        interrupted_command_exit_code=record['interrupted_command']['exit_code'],
        trial_files_unchanged=True, no_inference_training_benchmarks_or_tests_started=True,
        git_mutations=[], phase6='NOT STARTED', automatic_continuation=False))
    record['frozen_runtime_hash_checks'] = frozen_checks
    record['resume_helper'] = dict(path=str(helper.relative_to(ROOT)).replace('\\', '/'),
        sha256=sha(helper), plan_only_exit_code=0, syntax_valid=True,
        inference_path_executed=False, inference_path_verified=False,
        limitation='Saved original logits/lifecycle RAM data unavailable; interrupted pair numerical equivalence remains incomplete')
    record['counts']['interrupted_pair_saved_requests'] = 2
    record['interrupted_pair'] = read(OUT / 'remaining-systems-requests.json')[0]
    record['resume_constraint'] = 'Use the new resume helper and remaining-request list after explicit user authorization. Original runner has no resume support. Do not rerun completed trials. Lost original logits prevent complete interrupted-pair numerical proof.'
    write(OUT / 'checkpoint.json', record)
    status_path = ROOT / 'IMPLEMENTATION_STATUS.md'
    status = status_path.read_text(encoding='utf-8')
    begin = status.index('### Phase 5B continuation')
    end = status.index('### Phase 5 continuation', begin)
    replacement = '''### Phase 5B continuation (2026-10-14) — PARTIALLY EXECUTED / PAUSED

Research execution stopped at the user's explicit request. No automatic
continuation is authorized. Model processes exited and temporary sleep
prevention was released. **Phase 6 NOT STARTED.** All source changes,
configurations, data, weights/checkpoints, results, earlier figures and logs are
preserved locally; no commit, push, merge, deployment, reset or history rewrite.

Actual completed work: **26 validation trials/checks**, **490 held-out quality
trials / 490 passing checks**, **40 upstream SDPA checks**, and **59 written
systems requests / 19 verified paired cells**. The systems progress file reports
57 requests because two additional intact requests belong to the interrupted
20th pair. **261 of 320 primary requests remain**; the planned paired total is104.
197 tests passed before the experiment freeze; no tests were rerun for this stop.
The later collection guard and future resume inference path have not been tested.

All four JSONL gzip streams decoded successfully; inspected JSON files parse,
quality raw SHA matches its completed report, and frozen runtime hashes match.
No damaged experiment file was found or overwritten. Temporary pytest directories
remain preserved and excluded due to access restrictions. All 161 prior artifact
hashes and pretrained weights were verified at the start, not expensively rechecked.

Frozen manifest SHA is `dbca76151292e48cf0f4fbe163901a944927f099ba4df78b320d8bc57454a590`;
SmolLM2-135M revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, CPU FP32,
four threads, seeds17/29, recent/query/random/HTIS, recovery OFF/ON, budgets4/8/16
eight-token pages. Ten held-out cases/nine groups span450–2328 tokens with16/32
generation limits. Validation-only interval8/cooldown2 calibration is sealed.
The complete configuration and calibration are embedded in the checkpoint.

Remaining: systems controls, independent profiling/importance/budget diagnostics,
memory audit, exploratory controls (zero executed), formal paired/Pareto analysis,
Phase 5B figures and final acceptance verification. CUDA/GPU NOT EXECUTED.
The interrupted pair's original logits and earlier lifecycle data were in RAM
only; numerical verification remains incomplete. Quality timing outliers are
preserved and unqualified latency claims are unsupported. No final A/B/C verdict.

See [saved checkpoint](transformer-efficiency/results/phase5b/paused-checkpoint-v1/checkpoint.json)
and [resume instructions](transformer-efficiency/results/phase5b/PAUSED_RESUME.md).
The prepared continuation skips all 59 saved requests and writes a new directory;
only its inference-free plan mode was run. Closing VS Code is safe; a new explicit
user instruction is required to resume.

'''
    status = status[:begin] + replacement + status[end:]
    status = status.replace('Updated 2026-10-13.', 'Updated 2026-10-14.', 1)
    status_path.write_text(status, encoding='utf-8')
    plan_path = ROOT / 'RESEARCH_PLAN.md'
    plan = plan_path.read_text(encoding='utf-8')
    begin = plan.index('Current authorized continuation:') if 'Current authorized continuation:' in plan else plan.index('Current state: **Phase 5B')
    end = plan.index('1. **Phase 0:**', begin)
    plan = plan[:begin] + '''Current state: **Phase 5B PARTIALLY EXECUTED / PAUSED** at the user's explicit
request on 2026-10-14. No research execution or automatic continuation is authorized.
Phase 6 is NOT STARTED. The frozen protocol/configuration and all completed work
remain preserved. Completed:26 validation trials;490 held-out quality trials and
checks;40 SDPA checks;59 systems requests and19 verified paired cells. Remaining:
261 primary systems requests, diagnostics/profiling, exploratory controls, memory
audit, paired statistics/Pareto figures and final acceptance. Original logits for
the interrupted pair were not persisted; its numerical check remains incomplete.
Only lightweight artifact checks were performed during the stop; no full tests.

Use the [local checkpoint](transformer-efficiency/results/phase5b/paused-checkpoint-v1/checkpoint.json)
and [explicit resume runbook](transformer-efficiency/results/phase5b/PAUSED_RESUME.md)
after a new user instruction. Never rerun the completed validation/quality study
or the59 written systems requests. The prepared continuation consumes the exact
remaining-request list, preserves old outputs and labels unavailable verification.
It has not executed inference. The original reproduction commands start fresh
studies and are not resume commands. No retraining, paid compute or Phase 6.

''' + plan[end:]
    plan_path.write_text(plan, encoding='utf-8')
    commands_path = ROOT / 'transformer-efficiency/COMMANDS_EXECUTED.md'
    entries = []
    for path in sorted(BASE.glob('*.json')):
        value = read(path)
        if isinstance(value, dict) and isinstance(value.get('command'), list):
            entries.append(dict(record=path.name, command=value['command'],
                exit_code=value.get('exit_code'), started_at_utc=value.get('started_at_utc')))
    write(OUT / 'command-record-index.json', entries)
    command_text = '''
## Phase 5B actual execution and user-requested pause (2026-10-14)

Status: **PARTIALLY EXECUTED / PAUSED**. No automatic continuation or Phase 6.
Exact command arrays, UTC times, output, source hashes and exit codes are retained
in `results/phase5b/*command*.json` and indexed in
`results/phase5b/paused-checkpoint-v1/command-record-index.json`.

| Executed command / stage | Actual outcome |
|---|---|
| `python scripts/audit_phase5b_start.py` | Baseline HEAD2c37eaf,161 earlier artifact hashes and pretrained files verified; original ledger captured. |
| Data prepare v1/v2/v3 and official acquisition | v1 adapted inputs preserved; v2 failed Windows text encoding before evaluation; v3 UTF8 succeeded,12rows. Network-access retry/pinned acquisition records preserved. |
| `python -m htis_context.phase5b freeze` | Frozen primary manifest SHA dbca76151292e48cf0f4fbe163901a944927f099ba4df78b320d8bc57454a590; preregistration/calibration sealed. |
| Phase5B contract tests and full tests before freeze | Initial fixture3failed/8passed preserved; sandbox temporary-directory failure preserved; corrected normal-access full run197passed38.71s with pretrained/Phase3 opt-ins. |
| `python -m htis_context.phase5b validation` |26trials/checks complete; interval8/cooldown2 selected using validation only. |
| Existing GPU runner dry availability | CPU-only Torch/no CUDA; NOT EXECUTED on GPU. |
| `python -m htis_context.phase5b test` |490quality trials /490checks complete;40upstream SDPA checks; exit0; runtime sources unchanged. Internal wall53097.421s includes unexplained multi-hour timing gaps. |
| Artifact-only quality timing audit / hardware and event queries | Raw timings retained; three multi-hour outliers recorded; suspension cause unconfirmed. No new inference. |
| `python scripts/run_phase5b_supplements.py freeze` | Exploratory configuration frozen only; zero supplemental trials/requests executed. |
| `python scripts/phase5b_keep_awake.py` | Temporary thread-owned idle-sleep request; release flag caused clean exit and `active:false`. No persistent power-setting changes. |
| `python -m htis_context.phase5b systems` via recorded wrapper | User interrupted.59written requests,19verified pairs; progress57because next pair incomplete; wrapper exit1. Original artifacts preserved, no final systems report. |
| `Get-CimInstance Win32_Process` / guarded `Stop-Process` | Sandbox CIM denied; normal-access query identified task processes39088/24160 and stopped only matching systems children. Recorded wrapper exited/wrote log. Final Python inventory empty. |
| `python scripts/checkpoint_phase5b_paused.py` | Stdlib file-only gzip/JSON/hash audit. First attempt encountered a report-list count TypeError before any checkpoint file was written; corrected retry succeeded. No model/test imports.61files inventoried;490quality/26validation/59systems rows; all gzip/JSON intact. |
| `python scripts/resume_phase5b_systems.py --checkpoint .../paused-checkpoint-v1 --plan-only` | Exit0;261remaining requests; no inference imports/execution. Actual continuation path NOT EXECUTED. |
| `python scripts/finalize_phase5b_pause.py` | File-only frozen source checks, additive ledger/status/plan/command documentation and checkpoint seal. No inference/full tests/statistical recalculation. |

The interrupted pair has two saved requests but no persisted original logits;
numerical equivalence remains incomplete. No completed result was overwritten.
Profiling, importance/budget diagnostics, exploratory controls, memory audit,
final statistics/Pareto figures and final Phase5B verification remain unfinished.
Resume commands and limits are in `results/phase5b/PAUSED_RESUME.md`.
No push, merge, deploy, reset, commit or history rewrite; checkpoint saved locally.
'''
    with commands_path.open('a', encoding='utf-8') as stream:
        stream.write(command_text)
    ledger = read(ledger_path)
    ledger['recorded_at_utc'] = stamp
    ledger['phases']['5B'] = 'PARTIALLY EXECUTED / PAUSED (user-requested stop)'
    ledger['phase5b_scope'] = dict(status='PARTIALLY EXECUTED / PAUSED', counts=record['counts'],
        gpu='NOT EXECUTED', phase6='NOT STARTED', automatic_continuation=False,
        checkpoint='transformer-efficiency/results/phase5b/paused-checkpoint-v1/checkpoint.json',
        contribution_verdicts='UNASSESSED; final analysis not executed',
        limitations=['Original interrupted-pair logits and completed lifecycle RAM data unavailable',
                    'Quality timing outliers retained; no unqualified performance conclusion',
                    'No final statistics, figures or full Phase5B acceptance audit'])
    ledger['claims'].append(dict(id='C022', claim='Phase5B held-out quality execution completed',
        status='measured', scope='Frozen small CPU FP32 study; not complete Phase5B acceptance',
        metrics=dict(trials=490, checks=490, sdpa_checks=40),
        evidence=['transformer-efficiency/results/phase5b/evaluation-v1/report.json',
                  'transformer-efficiency/results/phase5b/evaluation-command-v1.json']))
    ledger['claims'].append(dict(id='C023', claim='Phase5B systems execution is partial and paused',
        status='partial', metrics=dict(written_requests=59, verified_pairs=19, remaining_requests=261),
        evidence=['transformer-efficiency/results/phase5b/paused-checkpoint-v1/checkpoint.json'],
        limitations=['Two additional intact requests belong to an unverified interrupted pair; original logits unavailable']))
    ledger['claims'].append(dict(id='C024', claim='User-requested local pause checkpoint passed file-integrity checks',
        status='verified', scope='JSON/gzip decoding, report raw hash and frozen source hashes only; no new inference or full tests',
        evidence=['transformer-efficiency/results/phase5b/paused-checkpoint-v1/artifact-inventory.json',
                  'transformer-efficiency/results/phase5b/paused-checkpoint-v1/stop-record.json']))
    old_paths = {item['path'] for item in ledger['artifacts']}
    added = read(OUT / 'artifact-inventory.json')
    for path in [*OUT.glob('*'), helper, BASE / 'PAUSED_RESUME.md', Path(__file__)]:
        if path.is_file():
            added.append(dict(path=str(path.relative_to(ROOT)).replace('\\', '/'),
                              sha256=sha(path), bytes=path.stat().st_size))
    for entry in added:
        if entry['path'] not in old_paths:
            ledger['artifacts'].append({name: entry[name] for name in ('path', 'sha256', 'bytes')})
            old_paths.add(entry['path'])
    assert ledger['claims'][:len(prior['claims'])] == prior['claims']
    assert ledger['artifacts'][:len(prior['artifacts'])] == prior['artifacts']
    assert all(ledger['phases'][key] == value for key, value in prior['phases'].items())
    write(ledger_path, ledger)
    write(OUT / 'documentation-seal.json', dict(recorded_at_utc=stamp,
        earlier_claims_preserved=True, earlier_artifact_entries_preserved=True,
        frozen_runtime_checks=len(frozen_checks), frozen_runtime_checks_passed=True,
        files=[dict(path=str(path.relative_to(ROOT)).replace('\\', '/'), sha256=sha(path), bytes=path.stat().st_size)
               for path in (ledger_path, status_path, plan_path, commands_path, BASE / 'PAUSED_RESUME.md', helper)]))
    print(json.dumps(dict(status=ledger['phases']['5B'], counts=record['counts'],
        earlier_claims_preserved=True, frozen_runtime_hashes_passed=len(frozen_checks),
        checkpoint=str(OUT)), indent=2))


if __name__ == '__main__':
    main()
