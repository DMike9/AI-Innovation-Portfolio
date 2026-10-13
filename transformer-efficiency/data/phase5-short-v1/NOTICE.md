# Source and scope

`dev-v1.1.json` is the unmodified SQuAD v1.1 development release downloaded
from the official SQuAD explorer. The release SHA256 and URL are recorded in
`manifest.json`. SQuAD is distributed under **CC BY-SA 4.0**; retain that license
and source attribution for the release and its paragraph/question/answer excerpts.

Official source and license notice:
https://rajpurkar.github.io/SQuAD-explorer/

License: https://creativecommons.org/licenses/by-sa/4.0/

The manifest selects one eligible QA per distinct article using stable article
and QA hashes before model evaluation: two validation and two held-out examples.
Paragraphs and questions are complete. Length and first-answer-token eligibility
are explicit, rather than silent truncation. These four examples do not constitute
an official SQuAD evaluation or leaderboard submission.

Other rows are explicitly authored short memory probes, not SQuAD, RULER or
LongBench examples. Beginning/middle/end variants share a fact/template and are
clustered together for statistical resampling. Validation/test facts differ,
but authored templates are shared; this does not establish template, language,
task-family or model-family generalization. Public-source presence in the base
model's pretraining was not audited. “Held-out” means held out from this controller
calibration/evaluation protocol, not proven absent from base-model pretraining.
