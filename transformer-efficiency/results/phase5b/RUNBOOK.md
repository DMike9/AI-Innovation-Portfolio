# Phase 5B reproduction runbook

Run from the repository root in the existing `.venv`. No training or checkpoint
download is needed. Earlier Phase 5 results are immutable. All new experiment
outputs live under `results/phase5b`; each runner refuses an existing output
directory. Use a new output name for a reproduction, never overwrite evidence.

## Frozen inputs

- `configs/smollm2-135m.json`: pinned 135M base-model revision, FP32 CPU, 8192 positions.
- `configs/phase5b-cpu-v1.json`: protocol/data/runtime/checkpoint hashes and fixed matrix.
- `data/phase5b-extended-v3/manifest.json`: outcome-independent, untruncated selected inputs.
- `results/phase5b/PREREGISTRATION.md`: primary metrics, controls, decisions and limits.
- `results/phase5b/validation-v1/calibration.json`: validation-only common recovery schedule.
- `.model-cache/smollm2-135m`: unchanged pretrained files; ignored by Git.
- `results/phase3/trained-v1/htis-seed-17` and `htis-seed-29`: unchanged learned checkpoints.

All numerical commands verify frozen runtime/data/model/scorer hashes. Reference
sources remain unchanged; added modules orchestrate existing inference/cache
interfaces. A different model/backend requires a new identity and acceptance.

## Commands

Each model experiment must execute sequentially, without concurrent model tests,
profilers or other benchmarks. Command records use the unchanged `run_recorded.py`
wrapper to retain actual UTC timestamps, output, exit codes and source hashes.

```powershell
.venv/Scripts/python.exe -m htis_context.phase5b validation --output transformer-efficiency/results/phase5b/reproduce-validation
.venv/Scripts/python.exe -m htis_context.phase5b test --output transformer-efficiency/results/phase5b/reproduce-evaluation
.venv/Scripts/python.exe -m htis_context.phase5b systems --output transformer-efficiency/results/phase5b/reproduce-systems
.venv/Scripts/python.exe -m htis_context.phase5b_diagnostics profile --output transformer-efficiency/results/phase5b/reproduce-profile
.venv/Scripts/python.exe -m htis_context.phase5b_diagnostics importance --output transformer-efficiency/results/phase5b/reproduce-importance
$env:MPLCONFIGDIR='.venv/phase5b-mpl-cache'
.venv/Scripts/python.exe -m htis_context.phase5b_analysis --output transformer-efficiency/results/phase5b/reproduce-analysis
$env:HTIS_RUN_PRETRAINED='1'
$env:HTIS_RUN_PHASE3_ARTIFACTS='1'
.venv/Scripts/python.exe -m pytest transformer-efficiency/tests -q -p no:cacheprovider
.venv/Scripts/python.exe transformer-efficiency/scripts/verify_phase5b.py
git diff --check
```

The test/systems/profile runners intentionally use the original immutable
`validation-v1/calibration.json`. Analysis intentionally reads the original
`evaluation-v1` and `systems-v1` artifacts. Reanalysis therefore reproduces the
reported experiment without silently replacing its evidence or thresholds.
To register a different protocol, use a separately versioned configuration and
explicitly labeled experiment; do not edit this manifest after seeing results.

The initial preparation attempt produced `phase5b-extended-v1` (adapted-only).
An official-data addition failed before any model evaluation due to Windows
default text encoding; the failed command and empty v2 directory are preserved.
The final v3 selected manifest uses UTF-8 for official multilingual prompt files.
Only v3 is evaluated. A tokenizer warning while scanning overlength official
cases describes excluded candidates; those cases are never run or truncated.

## Numerical and accounting acceptance

Full reference versus upstream SDPA compares complete prompts and common-token
continuations within atol5e-5/rtol1e-5. Prefix tests compare all generated step
logits/tokens on 104 matched cells, plus shared-prefix task changes. Modified
prefixes must miss. Every bounded teacher/free step must obey its physical page
and KV byte ceiling. Actual cold admission requests are charged and counted.

Dense prefill precedes bounded decoding and exceeds the decode-page budget.
Active/archive/scorer tensor bytes are measured; structural index/recipe bytes
are estimates. Independent dense prefix cache storage is extra. OS process memory
includes model, interpreter, allocator and retained research fixtures. No isolated
cache allocator peak or GPU memory/speedup is claimed.

## GPU boundary

The existing `htis_context.phase5_gpu` runner is preserved. The local CPU-only
Torch build has no CUDA device. Dry-run availability is not measured GPU evidence.
Remaining acceptance on compatible CUDA hardware: adapter/upstream/common-token
numerical checks; synchronized warmup/timing; actual allocated/reserved peaks;
CPU/GPU prefix transfers; supported retention controls; separately validated GPU
archive/recovery integration (the current recovery controller is CPU FP32 only).
No paid compute or hardware provisioning is part of this study.

Stop at the Phase 5B checkpoint. No Phase 6 files or work are created here.
