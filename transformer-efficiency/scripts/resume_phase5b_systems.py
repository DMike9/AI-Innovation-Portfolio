"""Explicit future continuation; skips saved requests, never rewrites old files.

No inference occurs with --plan-only. Interrupted in-memory logit comparisons
cannot be reconstructed from saved output IDs; they remain unverified.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(row):
    return tuple(row[x] for x in ('case', 'policy', 'budget', 'seed', 'recovery', 'repeat', 'traffic'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    checkpoint = read(args.checkpoint / 'checkpoint.json')
    pending = read(args.checkpoint / 'remaining-systems-requests.json')
    if args.plan_only:
        print(json.dumps(dict(status=checkpoint['status'], remaining_requests=len(pending),
            first_request=pending[0] if pending else None,
            inference_executed=False, original_files_will_not_be_overwritten=True), indent=2))
        return
    if args.output is None:
        parser.error('--output must name a NEW continuation directory')
    # These imports and all model execution occur only on explicit future invocation.
    import torch
    from htis_context import phase5b as run
    from htis_context.prefix_cache import ExactPrefixCache
    from htis_context.provenance import SourceRecipe

    for entry in read(args.checkpoint / 'artifact-inventory.json'):
        if run.sha(Path(entry['path'])) != entry['sha256']:
            raise ValueError('Checkpoint artifact changed: ' + entry['path'])
    args.output.mkdir(parents=True, exist_ok=False)
    saved = {}
    with gzip.open(run.RESULT / 'systems-v1/systems.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            saved[identity(row)] = row
    groups = defaultdict(list)
    for request in pending:
        groups[identity(request)[:-1]].append(request)
    with torch.no_grad():
        manifest, model, tok, dense, _ = run.load_run()
        cfg = manifest['configuration']
        calibration = run.read(run.RESULT / 'validation-v1/calibration.json')
        if run.sha(run.MANIFEST) != checkpoint['configuration_sha256']:
            raise ValueError('Frozen configuration changed')
        if run.sha(run.RESULT / 'validation-v1/calibration.json') != checkpoint['calibration_sha256']:
            raise ValueError('Frozen calibration changed')
        rec = calibration['recovery_configuration']
        data = {row['id']: row for row in run.read(run.DATA)['rows']}
        count = 0
        checks = []
        shared_cache = ExactPrefixCache(cfg['prefix_bytes'])
        for cell, requests in groups.items():
            case, policy, budget, seed, recovery, repeat = cell
            row = data[case]
            shared = any(x['traffic'].startswith('shared-prefix-') for x in requests)
            cache = shared_cache if shared else ExactPrefixCache(cfg['prefix_bytes'])
            current = {k[-1]: (value, None) for k, value in saved.items() if k[:-1] == cell}
            # An interrupted warm hit may require restoration of prefix KV only.
            # This is logged separately; no completed generation trial is repeated.
            if any(x['traffic'] == 'warm-repeat' for x in requests) and 'cold-admission' in current:
                state = dense.forward(torch.tensor([row['prefix_ids']])).state
                if not cache.put(state):
                    raise AssertionError('Resume prefix restoration failed')
                run.append(args.output / 'restorations.jsonl.gz', dict(case=case,
                    policy=policy, budget=budget, seed=seed, recovery=recovery,
                    repeat=repeat, prefix_tokens=len(row['prefix_ids']),
                    note='Prefix state restoration only; excluded from request latency comparisons'))
            for target in requests:
                traffic = target['traffic']
                modified = traffic == 'modified-prefix'
                trial_row = row
                if modified:
                    trial_row = dict(row)
                    trial_row['prefix_ids'] = list(row['prefix_ids'])
                    trial_row['prefix_ids'][0] = (trial_row['prefix_ids'][0] + 1) % model.config.vocab_size
                    trial_row['prefix_recipe'] = dict(tokens=[trial_row['prefix_ids']],
                        positions=[list(range(len(trial_row['prefix_ids'])))],
                        mask=[[1] * len(trial_row['prefix_ids'])])
                    if cache.get(dense.identity, SourceRecipe(**trial_row['prefix_recipe'])) is not None:
                        raise AssertionError('Modified prefix lookup unexpectedly hit')
                reuse = traffic not in ('cold-no-cache', 'shared-prefix-cold-control')
                result, logits = run.request(trial_row, policy, budget or 4, seed, recovery,
                    reuse, repeat, model, dense, tok, cfg, rec, cache)
                result['traffic'] = traffic
                if modified:
                    result['original_case'] = case
                    if result['hit']:
                        raise AssertionError('Modified prefix unexpectedly hit')
                if traffic == 'warm-repeat' and not result['hit']:
                    raise AssertionError('Warm continuation missed')
                if identity(result) != identity(target) or identity(result) in saved:
                    raise AssertionError('Resume request identity mismatch or duplicate')
                run.append(args.output / 'systems.jsonl.gz', result)
                current[traffic] = (result, logits)
                count += 1
                run.write(args.output / 'progress.json', dict(requests=count, planned=len(pending),
                    original_requests=len(saved), at_utc=run.stamp()))
                print(f'continuation {count}/{len(pending)} {case} {traffic}', flush=True)
            names = (['shared-prefix-task-change', 'shared-prefix-cold-control'] if shared
                     else ['cold-admission', 'cold-no-cache', 'warm-repeat'])
            values = [current[name] for name in names]
            tokens_passed = all(value[0]['output_ids'] == values[0][0]['output_ids'] for value in values)
            numeric_available = all(value[1] is not None for value in values)
            numerical_passed = None
            if numeric_available:
                numerical_passed = all(len(value[1]) == len(values[0][1]) and
                    all(torch.allclose(a, b, atol=run.read(run.MODEL)['atol'],
                        rtol=run.read(run.MODEL)['rtol']) for a, b in zip(values[0][1], value[1]))
                    for value in values)
            check = dict(case=case, policy=policy, budget=budget, seed=seed, recovery=recovery,
                repeat=repeat, tokens_passed=tokens_passed, numerical_passed=numerical_passed,
                status='VERIFIED' if numeric_available else 'INCOMPLETE: original logits were in memory only')
            run.append(args.output / 'equivalence.jsonl.gz', check)
            checks.append(check)
            if not tokens_passed or numerical_passed is False:
                raise AssertionError('Continuation equivalence failed')
            # Save lifecycle evidence for new cells separately from original lost RAM data.
            run.append(args.output / 'lifecycle.jsonl.gz', dict(case=case, policy=policy, budget=budget,
                seed=seed, recovery=recovery, repeat=repeat,
                events=cache.events, scope='Continuation only; original lifecycle RAM records unavailable'))
        run.verify_frozen(manifest)
        run.write(args.output / 'report.json', dict(requests=count, original_requests=len(saved),
            total_written_requests=count + len(saved), checks=checks, completed_at_utc=run.stamp(),
            raw_sha256=run.sha(args.output / 'systems.jsonl.gz'),
            status='REQUEST MATRIX COMPLETED; final acceptance still incomplete',
            limitations=['Interrupted original pair has no saved logits; numerical check remains incomplete.',
                'Completed original lifecycle records were in RAM only; do not fabricate them.',
                'Continuation is a separate timing session; do not silently pool host timing regimes.']))


if __name__ == '__main__':
    main()
