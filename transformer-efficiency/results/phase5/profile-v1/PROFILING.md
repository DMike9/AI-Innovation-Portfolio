# Existing-path CPU profile, before controller experiments

Pinned SmolLM2-135M FP32, four Torch threads, one 112-token prefix from an
existing validation document, one appended token. The 4,608.10 ms prefill is
a cold sample, not an amortized prefill benchmark. Each arm had two warmups,
three uninstrumented measurements, and one nested-span/cProfile measurement.
Instrumentation was removed afterward and logits passed the existing tolerance.
Native Torch CPU operators were profiled separately with memory tracking.

| Existing policy | Mean uninstrumented ms | Instrumented ms | Measured combined profiler/wrapper difference ms |
|---|---:|---:|---:|
| Protected recency | 175.61 | 441.58 | 265.97 |
| HTIS | 272.76 | 311.89 | 39.13 |
| Recency + recovery | 643.24 | 855.01 | 211.77 |
| HTIS + recovery | 771.47 | 1,036.16 | 264.70 |
| Bound + fallback | 1,279.59 | 1,770.57 | 490.97 |

These samples expose large profiler overhead and host jitter. Instrumented
times are diagnostic attribution, never substituted for comparative latency.
Inclusive spans overlap; exclusive spans subtract instrumented children.

HTIS's instrumented selection took 143.50 ms: runtime features 101.03 ms
(97.27 exclusive), learned network 29.39 ms, and selection's remaining exclusive
work 13.08 ms. The analytic projection alone took 3.75 ms. Feature extraction's
cProfile self time was 49.34 ms, with small reductions, indexing and locality
construction adding many interpreter/native-boundary calls. This supports
testing scoring frequency before another learned network. The feature cache
already avoids rescanning unchanged pages for insertion summaries.

Recovery's recency route took 587.77 ms inclusive, 311.01 ms exclusive, with
offload 135.62 ms exclusive, page checksums 77.35 ms, page splitting/copies
56.49 ms and explicit cache cloning 29.87 ms. cProfile identifies dataclass
serialization, NumPy reductions and checksum work. These are costs of this
safe reference implementation, not inherent lower bounds for archival memory.
First compaction mainly offloads; later request experiments measure actual
reloads. Keeping an archive still costs work when a cost gate rejects retrieval.

Fallback adds bound computation (385.78 ms exclusive), streamed attention
(163.30 ms exclusive), fetch-priority work inside adaptive attention, copies
and checksums. It evaluates retained attention before deciding to stream the
full supplied snapshot, and may evaluate the bound again after retrieval.
This is repeated work and a local reference, not an accelerated dense kernel.

Model linear operations consumed about 70–87 ms in the HTIS/recovery profiles.
Attention matmul/softmax usually consumed only a few ms there; the recency
instrumented sample instead had 71.29/21.72 ms, illustrating jitter. Merely
reducing QK operands cannot establish overall wall-clock improvement.

The prior Phase 4 evidence remains unchanged. History sees selected survivors
only and never reloaded a page there: absent pages cannot acquire new history.
Coordinate-wise query upper bounds can overestimate relevance, and frozen HTIS
was trained on few interventions/layers. These are plausible explanations for
bad replacements, not causal findings proven by this profile. Subsequent paired
free-generation and branch/controller ablations test their practical effects.

The fixed-state omission reference compounds relaxed mass and value-distance
bounds; individually achievable coordinate extrema need not co-occur in a real
key. Its Phase 4 median bound/error ratio was 45.25. Validity in observed local
comparisons does not imply tightness, final-answer correctness or repair of
earlier approximate states. No bound code or past evidence was changed.

CPU eager grouped attention, small kernels, Python routing, source validation
and CPU archives limit systems generalization. Upstream SDPA and actual cold/
warm requests are separate Phase 5 comparisons. GPU execution remains absent.
See `report.json`, the five `.pstats` artifacts and `../profile-command-v1.json`.
