# Commands and actual outcomes for the 2026-10-13 implementation session

Commands below ran from the repository root unless noted. JSON command records
retain argv, working directory, UTC timestamp, wall time, exit code, actual
stdout/stderr and, for final runs, source-file hashes. Git operations and initial
dependency/hardware probes were executed directly through the terminal tool.

## Git, instructions and environment

- `git status --short --branch`: clean `main`, behind origin by seven commits.
- `git remote -v`: public `DMike9/AI-Innovation-Portfolio` origin.
- `rg --files --hidden -g AGENTS.md -g '!.git' -g '!node_modules'`, plus explicit
  ancestor AGENTS.md checks: none found.
- `git fetch origin`: first failed opening `.git/FETCH_HEAD` under the sandbox;
  succeeded with Git-metadata access. No reset/merge/push.
- `git log --oneline --decorate -15 --all`, `git diff --stat HEAD
  origin/research/htis-kv-importance-benchmarks`, `git ls-tree -r --name-only
  origin/research/htis-kv-importance-benchmarks`: inspected the fetched research
  tip and its file additions.
- `git show` for the root README, build brief, architecture and research note;
  `Get-Content` for the full existing package, tests, pyproject and CI workflows.
- `git switch -c implementation/htis-context-phase-0-2
  origin/research/htis-kv-importance-benchmarks`: created separate working branch.
- `python --version`, `python -m pip --version`, `python -m pip list`: inspected
  initial Python 3.13.14 environment; NumPy/Torch/Transformers absent.
- `nvidia-smi`: executable unavailable. Initial `Get-CimInstance` queries were
  denied by sandbox; with read access, CPU/RAM/AMD graphics were identified.
- `python -m venv .venv`: created isolated workspace environment.
- `.venv/Scripts/python.exe -m pip install --disable-pip-version-check
  numpy==2.3.5 pytest==9.0.3 transformers==4.57.6`: succeeded.
- `.venv/Scripts/python.exe -m pip install --disable-pip-version-check
  torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu`: succeeded,
  installing `torch==2.9.1+cpu`.
- Official model card/config were read; an API query returned immutable revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. The sandbox blocked the initial
  network query; normal network access succeeded.
- `.venv/Scripts/python.exe transformer-efficiency/scripts/audit_environment.py`:
  wrote [environment.json](results/phase0/environment.json).
- `.venv/Scripts/python.exe -m pip freeze`: wrote the tested dependency lock.
- `.venv/Scripts/python.exe -m pip install -e ./transformer-efficiency --no-deps
  --no-build-isolation`: editable build and install succeeded.

## Recorded research commands

The recorder prefix is:

```powershell
.venv/Scripts/python.exe transformer-efficiency/scripts/run_recorded.py --output <command-record.json> -- <command>
```

During source-path runs, `$env:PYTHONPATH` was assigned the resolved
`transformer-efficiency/src` directory. The pretrained pytest run set
`$env:HTIS_RUN_PRETRAINED = '1'`. The download set
`$env:HF_HUB_DISABLE_XET = '1'` and used the public checkpoint only.

| Actual command following the recorder | Outcome / record |
|---|---|
| `.venv/Scripts/python.exe -m pytest transformer-efficiency/tests -q` | 55 passed; pytest cache directory warning; [baseline](results/phase0/baseline-tests.json) |
| `.venv/Scripts/python.exe -m htis_context.cli download --config transformer-efficiency/configs/smollm2-135m.json --model-dir .model-cache/smollm2-135m --output transformer-efficiency/results/phase0/checkpoint-manifest.json` | Pinned public weights downloaded and checksummed; [record](results/phase0/checkpoint-download.json) |
| `.venv/Scripts/python.exe -m htis.experiment --output transformer-efficiency/results/phase0/reproduced-synthetic.json` | All four synthetic scenarios completed; [record](results/phase0/simulator-command.json) |
| `.venv/Scripts/python.exe -m htis.ablation --output transformer-efficiency/results/phase0/reproduced-ablation.json` | Held-out synthetic ablation completed, validation chose 0.5; [record](results/phase0/ablation-command.json) |
| `.venv/Scripts/python.exe -m pytest transformer-efficiency/tests -q -p no:cacheprovider` | Intermediate runs: 70 passed/1 skipped; 104 passed/1 skipped; later 106 passed/1 skipped with one temporary-directory setup error; repaired access gave 108 passed/1 skipped |
| Same pytest command with `HTIS_RUN_PRETRAINED=1` | MHA-fixture run: 114 passed/1 failed; retained [failure](results/phase2/tests-before-mha-repair.json). Repaired final: 115 passed; [record](results/phase2/tests-final.json) |
| `.venv/Scripts/python.exe -m htis_context.cli equivalence --config transformer-efficiency/configs/smollm2-135m.json --model-dir .model-cache/smollm2-135m --output transformer-efficiency/results/phase1/equivalence-final.json` | 97 passed checks; [record](results/phase1/equivalence-command-final.json). Earlier 94-check runs also retained |
| `.venv/Scripts/python.exe -m htis_context.cli phase2 --config transformer-efficiency/configs/smollm2-135m.json --model-dir .model-cache/smollm2-135m --workload transformer-efficiency/configs/phase2-smoke.json --output transformer-efficiency/results/phase2/pretrained-final.json` | Final outcome in [record](results/phase2/pretrained-command-final-success.json). Initial run: 105 passed checks and 72 real-pretrained trials. One invocation before editable installation completed failed import; [failure](results/phase2/pretrained-command-final.json) |
| `.venv/Scripts/python.exe transformer-efficiency/scripts/compress_initial_trials.py` | Losslessly compacted the session's original 25 MB report into summary plus compressed raw JSONL; [record](results/phase2/compact-initial-command.json) |
| `.venv/Scripts/python.exe -m htis_context.cli --help` | Module CLI help succeeded; [record](results/phase0/cli-help-module.json) |
| `.venv/Scripts/python.exe -m htis_context.cli inspect --config transformer-efficiency/configs/smollm2-135m.json --model-dir .model-cache/smollm2-135m --output transformer-efficiency/results/phase1/model-inspection.json` | Loaded verified checkpoint; 134,515,008 parameters, 538,060,032 FP32 weight bytes; [record](results/phase1/inspect-command.json) |
| `.venv/Scripts/python.exe -m htis_context.cli dense --config transformer-efficiency/configs/smollm2-135m.json --model-dir .model-cache/smollm2-135m --prompt 'The capital of France is Paris. The capital of Italy is' --tokens 8 --output transformer-efficiency/results/phase1/dense-example.json` | Eight-token live pretrained example; [record](results/phase1/dense-example-command.json) |

## Applied demo and repository checks

- `node --test signalrank/tests/scoring-engine.test.js`: sandbox failed at
  child-process spawn before running cases. Recorder console UTF-8 was repaired.
  Normal process access: **4 passed**; [record](results/phase0/signalrank-tests-unsandboxed.json).
- `npm ci` in `signalrank`: 66 packages installed; locked dependencies reported
  five audit findings. Source and lockfile were preserved.
- `npm.cmd run build` in `signalrank`, through recorder `--cwd signalrank`:
  **passed**, 24 modules transformed; [record](results/phase0/signalrank-build.json).
- `git diff --check`: passed at the inspected checkpoint.
- Final Python verification: 22 artifact SHA-256 hashes verified, all numerical
  run model/test/workload source hashes matched current files, raw compressed
  trial count 72, paired reuse-on/off mean-KL maximum difference 0.0. Pyproject
  TOML and CI workflow YAML parsed; all three CI jobs present. `git rev-parse
  main` remained `ba13a37c87e45d9b9252b0fa6b48c8b076a2a5f2`.
- `.venv/Scripts/htis-context.exe --help`, `inspect`, and `dense`: Windows
  denied starting the generated launcher even with normal process access;
  the identical module CLI works. No ACL/security-setting changes were made.

No remote GitHub CI run, model training, GPU benchmark, deployment, paid
compute, manuscript build, merge, push or commit was performed.

## Phase 3 continuation from `0e134c2` (2026-10-13)

The tree was clean. `git status --short --branch`, `git log -5 --oneline
--decorate`, instruction-file searches, existing status/plan/ledger/brief reads,
and source/test/artifact inspection selected Phase 3. No fetch, branch import,
switch, merge, push, deployment or training replay occurred. Python verification
checked all 22 prior artifact hashes, relevant model/test/configuration source
hashes and unchanged dependency versions; that evidence was reused.

Public acquisition was executed through `.venv/Scripts/python.exe` with a
stdin Python script calling `public_corpus.acquire` on `phase3-small.json`.
It fetched the twelve configured public book URLs, stripped ebook header/footer
material and retained 40,345 UTF-8 extract bytes. Actual upstream/extract SHA-256,
URL, HTTP metadata, preparation rule and acquisition time are in
`data/phase3-public-v1/corpus.json`. No private data or paid services were used.

Commands below used the existing command recorder from the repository root:

| Command | Actual outcome / record |
|---|---|
| `.venv/Scripts/python.exe -m pytest transformer-efficiency/tests/test_phase3.py transformer-efficiency/tests/test_context_pages.py -q -p no:cacheprovider`, with `HTIS_RUN_PRETRAINED=1` | **58 passed**, including actual checkpoint capture/intervention; [initial record](results/phase3/tests-initial.json) |
| `.venv/Scripts/python.exe -m htis_context.phase3 trace` | **31 traces, 1,302 feature rows, 558 actions**; **191.77 s** internal CPU trace cost; [record](results/phase3/trace-command-v1.json) |
| `.venv/Scripts/python.exe -m htis_context.phase3 train` | **16 actual trained safe-tensor checkpoints**, train/validation/test rows 180/90/195; **12.07 s** training; [record](results/phase3/train-command-v1.json) |
| `.venv/Scripts/python.exe -m htis_context.phase3 evaluate` | **528 checks passed** and compressed raw trials; **203.55 s** evaluation; [record](results/phase3/evaluate-command-v1.json) |
| Same targeted pytest command, also with `HTIS_RUN_PHASE3_ARTIFACTS=1` | **61 passed**, including saved-weight reload/validation-MSE reproduction and corruption/provenance rejection; [final record](results/phase3/tests-final.json) |

The final tests used normal temporary-directory access for artifact corruption
fixtures. No test/model correctness failure occurred. Python compilation
(`python -m compileall -q transformer-efficiency/src/htis_context`), raw gzip
counts, dataset/document/task separation, all saved-weight checksums, raw runtime
trial hashes, recollection hashes, and `git diff --check` passed.

Observed negatives were retained: HTIS seed-17 four-page mean KL **0.319004**
versus protected recency **0.260575**, selection time **61.87 ms** versus
**6.00 ms**, and exploration mean KL **0.398958**. These are small-study CPU
diagnostics, not formal significance or end-task accuracy. The tested runtime,
corpus and trained weights were not replaced/tuned to improve the held-out result.
Phase 4 remains NOT STARTED; this continuation stops at the Phase 3 checkpoint.

## Phase 4 continuation from `28a959d` (2026-10-13)

The tree was clean and Phase 3 had been committed after the earlier session.
`git status --short --branch`, `git log -3 --oneline`, instruction-file searches,
complete source-of-truth document reads, and implementation/artifact inspection
selected Phase 4 only. No AGENTS.md was found. A Python SHA-256 verification
checked all **70** existing evidence artifacts with zero mismatches. Installed
Torch 2.9.1+cpu, Transformers 4.57.6, NumPy 2.3.5 and pytest 9.0.3 matched the
existing environment. The pinned local pretrained checkpoint and prior corpus,
weights, negative conclusions, SignalRank and original HTIS were reused.

Actual intermediate commands used the existing recorder:

| Command / environment | Outcome / record |
|---|---|
| `python -m pytest transformer-efficiency/tests/test_phase4.py -q -p no:cacheprovider --basetemp transformer-efficiency/results/phase4/pytest-temp`, `HTIS_RUN_PRETRAINED=1` | 17 passed, 6 failed; [initial record](results/phase4/tests-initial.json). New fixture protection filled its budget, fixture archive was too large to evict, a corruption probe needed inference mode, and absolute checkpoint loading changed strict configuration metadata. Corrected fixtures and used the established model path; identity checks stayed intact. |
| Same Phase 4 test module with `--basetemp .../pytest-temp-repaired` | **23 passed** in 24.36 s; [repair record](results/phase4/tests-repaired.json). This was intermediate verification; additional coverage and final source accounting followed. |
| `python -m htis_context.phase4` | Failed before inference: loader was accidentally inside inference mode, so weights lacked mutation counters; [record](results/phase4/evaluate-command-v1.json). Changed runner to `no_grad`; preserved the adapter guard. No fake replacement experiment was used. |
| `python -m pytest transformer-efficiency/tests -q -p no:cacheprovider --basetemp .../pytest-temp-final`, both pretrained and Phase 3 artifact flags | Sandbox denied fixture/cleanup access; [record](results/phase4/tests-final.json). The identical suite was rerun with normal local temporary-directory access. |
| `python -m pytest transformer-efficiency/tests -q -p no:cacheprovider`, `HTIS_RUN_PRETRAINED=1`, `HTIS_RUN_PHASE3_ARTIFACTS=1`, normal local access | **162 passed** in 53.73 s; [record](results/phase4/tests-final-access.json), including real pretrained integration and all prior tests. |

All commands above were invoked through `.venv/Scripts/python.exe`; the recorder
was `transformer-efficiency/scripts/run_recorded.py --output <record> -- <command>`.
The artifact corruption fixtures used ordinary temporary-directory access.
Generated fixture files are ignored separately from research evidence.
Failed records are retained. No trained weights or held-out thresholds were changed.

| Final continuation command | Actual outcome / record |
|---|---|
| `python -m htis_context.phase4 --output transformer-efficiency/results/phase4/evaluation-v2` | Eight workloads completed, then the supplied-KV perturbation attempted an inference-tensor edit outside inference mode; [failure record](results/phase4/evaluate-command-v2.json). Intermediate metrics had not been persisted and are not reported as results. Added explicit inference-mode perturbation and incremental gzip checkpoints. |
| Complete pytest command above, normal temporary access, both opt-in flags | **165 passed** in **110.25 s** after the final prefix isolation, chained lineage, working-budget, immutable-accounting and value-sensitive/cooldown repairs; [record](results/phase4/tests-final-current.json). |
| `python -m htis_context.phase4 --output transformer-efficiency/results/phase4/evaluation-v3` | **198 trials / 1,804 passed checks**, nine workloads / eleven arms / two repeats; **974.63 s** internal CPU wall time; [record](results/phase4/evaluate-command-v3.json), [report](results/phase4/evaluation-v3/report.json). Full tests finished before this timing run; runtime sources stayed unchanged during execution. Same frozen configuration and trained weights, no held-out tuning. |
| `python transformer-efficiency/scripts/summarize_phase4.py transformer-efficiency/results/phase4/evaluation-v3` | Derived **131,388 recovery events**, **44,280 controller decisions**, **44,271 local head comparisons**, zero observed bound violations, preserved failure/churn examples; [record](results/phase4/diagnostics-command.json). |

The isolated comparison retains negative quality and cost outcomes. Protected
recency KL **0.235721**, simple recovery **0.313246**, HTIS recovery **0.417444**.
Fallback KL **5.48067e-7** at **1103.86 ms/step**, with one explicit latency-budget
tolerance failure. No lower QK count was relabeled a hardware speedup. The expensive
archive arm uses a declared cost prior; measured CPU copy latencies were not changed.
Active KV, archive/index, working estimates and independent prefix-cache storage
are separated. No GPU measurements, new training, paid compute, commit, push,
merge, deployment or Phase 5/6 execution occurred.

- `python transformer-efficiency/scripts/diagnose_phase4_outlier.py`, through the
  recorder: measured first-query page mass (mean 1.925%, uniform fraction 8.081%),
  original/scaled real-KV value norms and local omission damage (0.086582 to
  8.898569). This was an **offline diagnostic only**, not runtime teacher access;
  [record](results/phase4/outlier-diagnostic-command.json).

- `python transformer-efficiency/scripts/verify_phase4.py`: **70 prior hashes
  verified, 93 total hashed artifacts, 198 raw trials / 1,804 checks**; final
  runtime/configuration source hashes matched, sources stayed unchanged during
  the comparison, incremental checkpoints matched canonical trials exactly,
  and protected prior files remained unchanged. Wrote
  [checkpoint verification](results/phase4/checkpoint-verification.json) and
  updated the existing claim/evidence ledger. `git diff --check` passed; CRLF
  conversion notices were informational. Later metadata-only reconciliation
  added the ledger/checkpoint themselves to the complete changed-file inventory
  and refreshed that checkpoint's ledger hash.

The artifact verifier was subsequently made repeat-safe: when Phase 4 is already
recorded, it checks artifacts and runtime/test/configuration/packaging/recorder
hashes without rewriting historical evidence or appending duplicates. This
post-run utility change does not alter inference, policy code, tests or results.
Two direct repeat audits passed with **93 artifacts**, and left the ledger and
checkpoint bytes unchanged. The original command snapshots retain their original
utility-source hashes; numerical source hashes still match the final tree.
