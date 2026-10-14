# Phase 5B: PARTIALLY EXECUTED / PAUSED

User-requested stop on 2026-10-14. No automatic continuation is authorized.
All research/model processes stopped. Temporary sleep prevention was released.
Phase 6 is NOT STARTED. This is a local filesystem checkpoint, not a Git commit.

## Preserved progress

- Validation: 26 saved trials/checks; calibration frozen before held-out inference.
- Held-out quality: all 490 trials, 490 passing checks, 40 upstream SDPA checks,
  final report and successful command record. Raw artifact SHA matches the report.
- Systems: 59 complete written requests out of 320, 19 numerically verified paired
  cells out of 104. The progress file reports 57 because it updates after a pair.
- The interrupted 20th cell is Amazon rainforest, recent, eight pages, seed29,
  recovery ON, repeat1. Its cold admission and warm repeat were written intact.
  Its cold-no-cache request was interrupted before any row was written.
- Existing code/configuration/data, pretrained weights, frozen learned checkpoints,
  prior phase results/figures, preparation failure logs, 197-test passing record,
  diagnostic timing audit and all command records are preserved. No Phase 5B
  figures or final statistical analysis were completed.

The four gzip streams pass sequential decoding/JSON-line checks; inspected JSON
files are valid. No corrupted or truncated experiment file was found and no raw
result was repaired or overwritten. `paused-checkpoint-v1/artifact-inventory.json`
records 61 files with SHA256/size; temporary pytest directories are preserved but
excluded from inspection because Windows denies access. Model weights were
verified at start, not hashed again during this lightweight checkpoint.

`paused-checkpoint-v1/checkpoint.json` embeds the exact frozen configuration and
calibration. Separate lists identify every saved quality/validation trial, systems
request and verified paired cell, plus all remaining requests in original order.

## Exact experiment identity

- Starting HEAD: `2c37eaf6333c11e114246395390e2d778d94ff45`.
- Manifest: `configs/phase5b-cpu-v1.json`, SHA256
  `dbca76151292e48cf0f4fbe163901a944927f099ba4df78b320d8bc57454a590`.
- Data: `data/phase5b-extended-v3/manifest.json`, ten held-out cases/nine groups,
  two separate validation cases; held-out inputs 450–2328 tokens, generation16/32.
- SmolLM2-135M revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, CPU FP32,
  Torch2.9.1+cpu, four Torch threads, batch/concurrency1, greedy with EOS stopping.
- Frozen HTIS seeds17/29; recent/query/random/HTIS; recovery OFF/ON; page size8;
  page budgets4/8/16; protected sink/prompt/recent tokens8 each. KV ceilings
  1,474,560 / 2,949,120 / 5,898,240 bytes. Dense prefill remains outside that budget.
- Archive128MiB, index64MiB, estimated working512MiB, exact prefix cache192MiB.
- Calibration SHA256 `a9f9691826dd7ec2426fe9b285f38bdb0a396063d5c14262c0b018f9838f3d53`:
  interval8, cooldown2, query-change0.35, risk/uncertainty/ranking0.5,
  max-route100000ms, score prior0.1. Full details are embedded in the checkpoint.
- Local Radeon/CPU-only Torch provides no compatible CUDA run. GPU NOT EXECUTED.

## Future resume commands — only after an explicit new user instruction

Run from the repository root. Do not run the original `phase5b test`, `validation`
or `systems` commands to resume: they have no skip support. Their existing output
directories must stay immutable. The following plan command imports no Torch or
model code and executes no inference:

```powershell
.venv/Scripts/python.exe transformer-efficiency/scripts/resume_phase5b_systems.py --checkpoint transformer-efficiency/results/phase5b/paused-checkpoint-v1 --plan-only
```

Prepared continuation command (NOT EXECUTED; inference path only reviewed
statically, deliberately not tested during the user-requested stop):

```powershell
.venv/Scripts/python.exe transformer-efficiency/scripts/run_recorded.py --output transformer-efficiency/results/phase5b/systems-resume-command-v1.json -- .venv/Scripts/python.exe transformer-efficiency/scripts/resume_phase5b_systems.py --checkpoint transformer-efficiency/results/phase5b/paused-checkpoint-v1 --output transformer-efficiency/results/phase5b/systems-resume-v1
```

This uses the 261-request remaining list, verifies preserved artifact hashes,
refuses an existing output directory and never regenerates the 59 saved requests.
The first request is the interrupted cell's missing cold control. An interrupted
warm request in another checkpoint could require separately logged prefix-state
restoration; that would not repeat completed generation or enter request timings.

The original paired logits and completed lifecycle records existed only in RAM.
The interrupted cell's numerical equivalence cannot be recovered from output IDs.
The continuation records that numerical comparison as INCOMPLETE and may compare
saved tokens; it does not fabricate logits or claim full acceptance. Any additional
numerical verification would require separately authorized, labeled inference.
Keep timing sessions distinct. If the continuation is stopped, checkpoint its new
rows and extend the skip list before another attempt; this helper currently reads
only the original checkpoint. Do not rerun the same continuation command in a new
directory after partial execution without updating that skip list.

## Remaining work

261 primary systems requests; completion of pairing/shared-prefix/invalidation
acceptance with the lost-logit/lifecycle limitation above; independent profiling,
learned-importance and budget-rank diagnostics; source memory-accounting audit;
formal paired statistics/Pareto analysis and figures; final verification.
Exploratory frozen controls have executed zero trials: 80 zero-archive quality,
48 zero-archive requests, and 80 high-reuse requests. These are separate from the
primary 490/320 matrix. No final contribution A/B/C verdict has been established.

Later diagnostic commands are recorded here for reproducibility, not queued:

```powershell
.venv/Scripts/python.exe -m htis_context.phase5b_diagnostics profile
.venv/Scripts/python.exe -m htis_context.phase5b_diagnostics importance
.venv/Scripts/python.exe transformer-efficiency/scripts/diagnose_phase5b_budget_ranks.py
.venv/Scripts/python.exe transformer-efficiency/scripts/run_phase5b_supplements.py archive-control
.venv/Scripts/python.exe transformer-efficiency/scripts/run_phase5b_supplements.py high-reuse
.venv/Scripts/python.exe transformer-efficiency/scripts/audit_phase5b_memory.py
```

Analysis and `verify_phase5b.py` currently assume one complete original systems
report. Adapt them to explicitly consume the continuation and retain incomplete
checks before running them; do not label this paused checkpoint verified complete.
No full test suite was rerun for the checkpoint. The last passing 197-test record
predates the later collection guard and resume helper.

Quality timings contain three multi-hour outliers retained in
`quality-timing-audit-v1.json`; their cause is unconfirmed. Do not silently remove
them or claim unqualified latency improvements. Active attention KV is not total
resident memory: dense fixtures and diagnostics remain separately chargeable.

VS Code may be closed safely after this checkpoint. Nothing is scheduled to
continue; a new explicit user instruction is required.
