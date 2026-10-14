# Separately labeled exploratory controls

These controls supplement the frozen primary matrix without changing its
configuration, examples, thresholds, results or A/B/C decision rules. Their
own configuration/source/calibration hashes are frozen before execution.
They are not new independent source/session samples.

The existing RecoveryPolicy changes candidate availability and routing. In
particular, a protected-recency wrapper refreshes using query-summary ranking;
comparing it only with direct recency cannot isolate the benefit of archive
retrieval from the initial ranking change. Use an identical wrapper/configuration
with archive capacity **zero**, index capacity unchanged, at the fixed middle
budget of **eight pages**. Include every primary quality case, all four policies
and both seeds: **80 quality trials**. Include both original systems cases,
all four policies, both seeds, actual cold admission/cold control/warm repeat:
**48 requests**. Actual discard-copy costs and retained index memory are charged;
this control is not equal-memory to a direct no-recovery policy. No new controller
architecture or scoring weights are introduced. Compare with corresponding
primary archive-enabled cells, separately from the primary tests.

To test actual high reuse, rather than only deriving an eight-hit traffic mix,
run both systems cases and all four policies at eight pages, seed17, no recovery.
Each independent cache receives one actual admission request, **eight actual
warm requests**, and a cold no-cache control: **80 requests**. Verify tokens and
all captured logits. Charge cache admission and stored prefix memory. These
requests test high reuse on a narrow fixed-budget subset; primary factorial
coverage at all three budgets remains separate.

An external timestamp hook records when the first greedy argmax result becomes
available, including untimed Python phase gaps. The primary runner's TTFT is
a sum of measured phase spans; report that distinction and the observed gap.
The hook does not alter scores, tokens or selection, is always restored, and
does not constitute a network/streaming serving benchmark.

The primary driver retains dense prefix/prepared states during decoding. A
separate source-accounting audit must charge these driver-owned tensor payloads
in total-residency tables. Active attention KV capacity is not whole-process
memory. OS working-set/lifetime-peak, physical tensors, structural estimates,
diagnostic logit buffers and untracked allocator/activation attribution remain
distinct. Accounting corrections are not protocol tuning or favorable-result
selection; preserve the original raw metrics and qualified primary tables.
