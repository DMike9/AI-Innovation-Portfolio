# HTIS-Context research plan

The implementation contract is [CODEX_BUILD_PROMPT.md](transformer-efficiency/CODEX_BUILD_PROMPT.md).
The architectural starting point is [HTIS_CONTEXT_ARCHITECTURE.md](transformer-efficiency/HTIS_CONTEXT_ARCHITECTURE.md).
Read [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) and
[CLAIM_EVIDENCE.json](CLAIM_EVIDENCE.json) before making claims or continuing.

## Acceptance gates

1. **Phase 0:** clean baseline or documented failures; preserve user changes;
   inspect hardware and dependencies; pin a pretrained model after reviewing
   its official card/license/config. Record real commands and artifacts.
2. **Phase 1:** independent exact adapter with approximation disabled; compare
   cached/uncached logits, teacher forcing and greedy generation to unchanged
   upstream. Validate original positions, RoPE, padding/masks, GQA, append,
   snapshots, isolation, model revision, and safe restart/reload.
3. **Phase 2:** physically selected attention inputs, bounded page allocations,
   exact prefix lookup/invalidation and deterministic LRU/LFU; record costs
   and quality changes on real weights. Full capacity must remain equivalent.
4. **Phase 3:** offline interventions for single/multiple pages; local output
   damage and next-token KL labels separate from runtime features. Hold out
   documents/sessions and a task family. Train linear and MLP baselines, then
   ablatable attention/locality/content branches. No future-token features.
5. **Phase 4:** bounded CPU archive with same-lineage page recovery, transfer
   counters and measured latency; missing/stale-page failures. Integrate the
   local omission reference with complete omitted-page coverage and correct
   current-snapshot fallback semantics. Never declare dense-history recovery
   after approximate steps without valid-prefix recomputation.
6. **Phase 5:** controlled quality/resource experiments, prefix reuse x policy
   factorial, composed strong baselines, benchmark revisions, multiple seeds,
   paired uncertainty and optimized dense kernels. Use available local hardware
   or request approval before paid resources. A second model family is needed
   for generalization claims.
7. **Phase 6:** build the paper and evidence explorer from measured records.
   Refresh primary literature before novelty assertions; retain negative results.

## Reproducibility and scientific boundaries

- Preserve the legacy `htis` simulator, original `src/filename.py`, historical
  artifacts, and separate SignalRank applied demo.
- Keep source recipes, semantic routing vectors, reusable KV, and active
  attention tensors separately typed. Semantic similarity cannot authorize KV.
- Exact reuse includes model/checkpoint, tokenizer/template, ordered ancestry,
  positions, masks, RoPE, adapters, precision/format, scope and implementation
  lineage. Never insert approximate states into the exact namespace.
- Declare tolerances before model validation. Report unsupported cases and
  numerical failures; do not substitute random fixtures for pretrained runs.
- Command records retain output, exit codes, timestamps, configurations and
  source hashes. Weights stay in the ignored local cache, never Git.
- CPU tensor byte estimates and eager-reference timings do not establish GPU
  memory savings, throughput gains or long-context task quality.

The phase/module/acceptance mapping and model choice rationale are in
[ADR 0001](transformer-efficiency/decisions/0001-reference-adapter.md).
