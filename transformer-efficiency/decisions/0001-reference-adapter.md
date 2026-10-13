# ADR 0001: a model-specific exact CPU reference before learned memory control

Decision date: 2026-10-13. The clean starting tree was `main` at `ba13a37`.
`git fetch origin` refreshed refs; the requested research tip was
`d918894e0803691c5acfef7a92060b6e1b4c2281`. A new branch,
`implementation/htis-context-phase-0-2`, was created from that ref. No merge,
commit, reset, force push, deployment, or history rewriting was performed.
No applicable AGENTS.md was found in the workspace, the research tree, or
the checked ancestor directories. The detailed brief governs implementation.

## Model and environment choice

Use the genuine pretrained **HuggingFaceTB/SmolLM2-135M** base checkpoint at
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Its official [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/README.md)
lists Apache 2.0 and describes pretraining. The official [configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json)
specifies 8192 positions, 30 layers, 9 query heads, 3 KV heads, 64 dimensions
per head, theta 100000, and no scaled RoPE. These were checked against the
downloaded files, whose checksums are in the checkpoint manifest.

The accessible host has an AMD Ryzen 7 PRO 250, 8 cores/16 threads,
16,296,779,776 bytes RAM, and AMD Radeon 780M integrated graphics. No NVIDIA
device was detected and PyTorch reports CUDA unavailable. Use **CPU FP32,
four threads**, Python 3.13.14, PyTorch 2.9.1+cpu and Transformers 4.57.6.
Model weights reside in an ignored local cache; no paid resources are used.

## Reference boundary

Implement an independent Llama forward orchestration and grouped attention
with unmodified upstream projections, normalization, MLP, and RoPE utilities.
Do not monkey-patch the upstream model used for comparison. Use eager FP32
with `atol=5e-5`, `rtol=1e-5`, declared before validation. This is an
inspectable correctness reference, not a production fast kernel.

Only binary key-padding plus causal slot order is accepted. Original RoPE
positions are separate from compact KV indices. Reject scaled/dynamic RoPE,
unvalidated adapters, tensor-parallel variants, and arbitrary additive masks.
The dense path supports batched/multi-token append. Initial sparse decode
supports one request and one new token, with a shared page subset across
KV groups per layer. Parameter-version and config guards reject stale model
identity after trusted-process mutations.

## Phase map and acceptance tests

| Phase | Modules/artifacts | Acceptance |
|---|---|---|
| 0 | environment audit, command recorder, pinned model/workload configs | Existing tests run, real hardware/version manifest, preserved history and demos |
| 1 | `checkpoint`, `adapter`, `provenance`, `state`, `equivalence`, CLI | Pretrained cached/uncached logits and eight-step greedy match upstream; positions/masks/padding/GQA/isolation/invalidation tests |
| 2 | `pages`, `prefix_cache`, `experiment` | Selected keys physically precede attention, full-capacity equivalence, exact prefix hits and negative misses, byte/copy accounting, recorded real-model policy degradation |
| 3 | future counterfactual data, linear/MLP, attention/locality/content fusion | Session/task split, no lookahead, checksum-linked trained artifacts and held-out ablations |
| 4 | future bounded CPU archive, fetch/controller/runtime bound | Same-lineage reload, costs, full omitted-page coverage, no false dense-history restoration |
| 5 | future controlled public tasks and hardware runner | Composed baselines, factorial ablations, optimized dense comparator, repeated/paired metrics |
| 6 | future paper and research explorer | Tables/visuals generated from actual evidence; complete attribution and claim audit |

## Deliberate limitations

Dense prefill still runs before sparse decode. Page features are cached at
insertion; only touched pages are resummarized. Attention history is observed
after selected attention. The query-summary baseline is a simplified
[Quest-inspired](https://arxiv.org/abs/2406.10774) heuristic, not a faithful
reproduction or a learned HTIS controller. Under identical protection the
recent and sinks+recent policies can be identical; this is retained honestly.

Prefix entries are complete independent prefixes, not deduplicated radix
blocks. They preserve all ancestry and therefore have no orphan-parent
eviction dependency. LRU/LFU are initial baselines, not learned reuse value.
Sparse omissions are permanent in Phase 2; recoverable archive memory is
Phase 4. Snapshots persist using safe tensors and validated provenance,
not arbitrary-code pickle. This trusted local API is not a production
multi-tenant authorization service.

Memory ceilings account for retained KV and estimated known cache/attention
tensor buffers; Python overhead, model/MLP/logit activations, allocator peaks,
and hidden kernel scratch are not certified. CPU prefix copies are reported
separately. There is no GPU residency, GPU timing, speedup, trained-importance,
long-context benchmark, novelty, or publication claim from this implementation.
