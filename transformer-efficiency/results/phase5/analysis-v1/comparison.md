# Phase 5 measured CPU pilot

Eight cases, six document/session groups after clustering positional probes; only two cases for ablations and factorials. No statistical significance claim.

| Policy | EM | F1 | KL | NLL | Decode ms | Active KV bytes | Total cache/index/scorer bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0.000 | 0.254 | 0.000000 | 4.353 | 131.22 | 10644480 | 12804960 |
| recent | 0.000 | 0.050 | 0.738001 | 7.341 | 134.16 | 1474560 | 1772640 |
| window | 0.000 | 0.021 | 3.621609 | 12.637 | 139.44 | 1474560 | 1772640 |
| sinks_recent | 0.000 | 0.050 | 0.738001 | 7.341 | 136.04 | 1474560 | 1772640 |
| random | 0.000 | 0.050 | 0.748942 | 7.154 | 135.80 | 1474560 | 1772640 |
| history | 0.000 | 0.073 | 0.541597 | 6.551 | 150.13 | 1474560 | 1772640 |
| query | 0.000 | 0.133 | 0.561378 | 6.834 | 139.64 | 1474560 | 1772640 |
| deterministic/recovery | 0.000 | 0.050 | 0.726438 | 6.959 | 373.67 | 1474560 | 17166183 |
| htis | 0.000 | 0.110 | 0.622581 | 6.435 | 148.74 | 1474560 | 1775132 |
| htis/recovery | 0.000 | 0.110 | 0.624669 | 6.441 | 523.19 | 1474560 | 17168707 |
| recent/recovery | 0.000 | 0.222 | 0.477930 | 6.781 | 373.59 | 1474560 | 17166229 |
| bound/fallback | 0.000 | 0.254 | 0.000000 | 4.353 | 942.90 | 1474560 | 17166251 |
| conditional-4-.35 | 0.000 | 0.087 | 0.654945 | 6.125 | 177.12 | 1474560 | 1844252 |
| cost/recovery | 0.000 | 0.050 | 0.738001 | 7.341 | 427.53 | 1474560 | 17166190 |
| benefit1/recovery | 0.000 | 0.050 | 0.738001 | 7.341 | 443.54 | 1474560 | 17148881 |
| calibrated | 0.000 | 0.050 | 0.738001 | 7.341 | 134.78 | 1474560 | 1772640 |
| simple-selected | 0.000 | 0.050 | 0.702392 | 7.208 | 137.53 | 1474560 | 1772640 |

Pure window has different prompt protection. Full has an unbounded KV budget. Sinks+recent aliases protected recency. Extra archive and prefix memory is not an equal-total-memory comparison.

## Dominated configurations (F1, CPU decode, total storage)

- `sinks_recent`: dominated by `recent`, `random`, `calibrated`
- `random`: dominated by `recent`, `calibrated`
- `history`: dominated by `query`
- `deterministic/recovery`: dominated by `full`, `recent`, `sinks_recent`, `random`, `history`, `query`, `htis`, `conditional-4-.35`, `calibrated`, `simple-selected`
- `htis`: dominated by `query`
- `htis/recovery`: dominated by `full`, `query`, `htis`, `recent/recovery`
- `recent/recovery`: dominated by `full`
- `bound/fallback`: dominated by `full`
- `conditional-4-.35`: dominated by `query`, `htis`
- `cost/recovery`: dominated by `full`, `recent`, `sinks_recent`, `random`, `history`, `query`, `deterministic/recovery`, `htis`, `conditional-4-.35`, `calibrated`, `simple-selected`
- `benefit1/recovery`: dominated by `full`, `recent`, `sinks_recent`, `random`, `history`, `query`, `htis`, `conditional-4-.35`, `calibrated`, `simple-selected`
- `calibrated`: dominated by `recent`
- `simple-selected`: dominated by `recent`, `sinks_recent`, `random`, `calibrated`

## Reuse × importance × recovery

| Policy | Reuse | TTFT ms | End-to-end ms | Tokens/s | Extra prefix bytes |
|---|---|---:|---:|---:|---:|
| simple-selected | False | 951.58 | 1618.11 | 3.83 | 0 |
| simple-selected | True | 379.62 | 1004.71 | 6.00 | 11525602 |
| htis | False | 1038.20 | 1740.92 | 3.54 | 0 |
| htis | True | 525.10 | 1202.50 | 5.03 | 11525602 |
| simple-selected/recovery | False | 1565.02 | 3225.06 | 1.95 | 0 |
| simple-selected/recovery | True | 1075.14 | 2715.76 | 2.32 | 11525602 |
| htis/recovery | False | 2120.70 | 4577.55 | 1.42 | 0 |
| htis/recovery | True | 1332.44 | 3787.53 | 1.70 | 11525602 |
