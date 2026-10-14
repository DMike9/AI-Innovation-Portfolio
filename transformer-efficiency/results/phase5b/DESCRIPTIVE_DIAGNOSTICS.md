# Separately labeled descriptive budget-score diagnostic

Specified after the primary protocol freeze and before primary held-out model
evaluation. The primary manifest, protocol, candidates and performance paths
remain unchanged. This supplement produces no primary-quality decision or tuning.

The primary protocol's offline dense snapshots establish that fixed-state scores
have no explicit budget input. To inspect changes arising from *actual retained
states*, run `scripts/diagnose_phase5b_budget_ranks.py` after primary numerical
experiments, sequentially. Use every held-out case, frozen HTIS seeds17/29, all
4/8/16-page budgets, first at most five common teacher tokens and sampled layers
0/14/29. Teacher tokens enter only as current/past inputs; future tokens/answers
are never scorer features. External read-only hooks record existing network
outputs and normalized inputs without changing selection or architecture.

Preserve common-page score differences and ordinal rank correlations, including
negative/task-dependent findings. Differences can arise from changed survivor
features and downstream query drift; do not call them a direct learned response
to a budget feature. Hooked timings are diagnostic and are excluded from primary
runtime comparisons. Record raw compressed traces, source/configuration hashes
and a separate command. Do not use these observations to recalibrate or retrain.
