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
