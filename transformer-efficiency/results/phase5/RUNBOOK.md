# Phase 5 CPU pilot reproduction

Run from the **repository root**. The pinned local model directory must retain
its established relative spelling `.model-cache/smollm2-135m`; changing that
loader path changes strict upstream configuration metadata. Weights are genuinely
pretrained SmolLM2-135M at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
Phase 3 controller safetensors and feature normalizers are unchanged.

Use the isolated environment and `transformer-efficiency/requirements-phase5.lock`
with the PyTorch CPU index. `software-v1.json` contains the actual environment.
The CUDA runner requires a compatible local CUDA-enabled Torch 2.9.1 installation;
the CPU lock cannot enable CUDA. Do not buy/provision compute automatically.

All evidence destinations reject overwrite. For a reproduction, use new output
versions, regenerate an explicitly named manifest, and pass that manifest and
its matching validation calibration to the test command. Existing artifacts
are immutable. The saved corpus has complete official paragraphs and a release
checksum; see its `NOTICE.md` and `manifest.json`.

Executed protocol (arguments abbreviated here; exact commands are in the JSON
command records and `../../COMMANDS_EXECUTED.md`):

1. `python -m htis_context.phase5_profile --output <new-profile-directory>`
2. `python -m htis_context.phase5 freeze --profile <profile>/report.json --output <new-manifest.json>`
3. `HTIS_RUN_PRETRAINED=1 HTIS_RUN_PHASE3_ARTIFACTS=1 python -m pytest transformer-efficiency/tests -q`
4. `python -m htis_context.phase5 validation --manifest <manifest> --output <new-validation-directory>`
5. `python -m htis_context.phase5 test --manifest <manifest> --calibration <validation>/calibration.json --output <new-evaluation-directory>`
6. `python transformer-efficiency/scripts/analyze_phase5.py --input <evaluation> --manifest <manifest> --output <new-analysis-directory>`

The final analysis wrapper preserves the original utility but fixes matched-seed
pairing and figure readability. Initial `analysis-v1` is preserved; `analysis-v2`
is the current interpretation. Statistical means/interval estimands are described
below. The checkpoint utility verifies all numerical runtime source hashes and
leaves prior claims and 93 earlier evidence hashes unchanged.

PowerShell flags use `$env:HTIS_RUN_PRETRAINED='1'` and
`$env:HTIS_RUN_PHASE3_ARTIFACTS='1'`. Invoke the interpreter as
`.venv/Scripts/python.exe`. Record every execution with the existing
`transformer-efficiency/scripts/run_recorded.py --output <record.json> -- <command>`.
Tests need normal temporary-directory access on this Windows installation.
Use a writable `MPLCONFIGDIR` such as `.venv/phase5-mpl-cache` for plotting.

Core quality uses independent teacher-forced and six-token greedy trajectories
starting from the same dense context/question prefix. The last question token
is physically sparse before the first answer prediction. Its decode timings
exclude the shared prefill fixture. Upstream SDPA uses unmodified upstream
forward and four common-token equivalence comparisons per case. It is a real
CPU SDPA comparator, not a FlashAttention/GPU result.

Separate actual systems requests compute cold prefixes or restore validated exact
CPU snapshots, process the question suffix, and generate six tokens. Factorial
cells have two sequential repetitions, batch/concurrency1. Prefix population
and admission costs are recorded separately; warm lookup storage and copying
are not free. “Importance” in this factorial means learned HTIS compared with
the validation-selected simple physical retention policy. Both use physical
page selection. Modified-token ancestry intentionally misses the exact cache.

Matched bounded core arms share four pages, eight tokens per page and prompt/
sink/recent protection of eight. Pure window removes early protection and is
a separate stratum. Sinks+recent aliases protected recency under these settings.
Full and upstream SDPA keep all KV. Fixed eight-page sensitivity is not adaptive
budgeting. Archives, indices, scorer/query state and independent prefix entries
are separate storage costs. Temporary working memory is an estimate, and
model weights/activations, Python heap, allocator/RSS peaks and kernel scratch
are not completely measured on CPU.

The feature-cache ablation temporarily restores the existing feature hook around
each forward and recomputes summaries only from available physical KV. It is
single-thread experimental instrumentation, not a concurrent-serving interface.
Synthetic equality/hook-restoration fixtures are unit tests; the pilot itself
uses the real model and original trained controller weights.

Statistical tables average cases, with saved seed means. Paired bootstrap first
averages within document/session clusters and weights those clusters equally;
its estimand differs from the eight-case average when cluster sizes differ.
Beginning/middle/end variants form one cluster. Six clusters, two timing repeats
and two ablation cases are exploratory evidence, not a significance claim.
Per-token latency percentiles are descriptive, never independent quality samples.

The CUDA runner's dry-run is:
`python -m htis_context.phase5_gpu --dry-run --output <new-availability.json>`.
On appropriate **already available local hardware**, omit `--dry-run` to execute
its acceptance gates before timings. It disables TF32, synchronizes boundaries,
checks upstream eager/SDPA common-token logits, compares reuse trajectories,
warms twice/repeats three times and records allocated/reserved CUDA peaks.
Its CPU-only dry-run and synchronization/gating tests passed; all CUDA execution
paths remain **NOT EXECUTED** here. CPU archive recovery deliberately refuses CUDA.

Do not mistake the local fixed-state bound for final-answer correctness or a
repair of earlier approximation. Periodic-bound skips explicitly provide no
certificate. The benefit-cost wrapper's risk-margin-to-ms conversion is a
validation candidate heuristic, not a learned quality guarantee.

Phase 6 is not started. The complete Phase 5 hardware/long-context/generalization
program remains unexecuted beyond this narrow CPU pilot.
