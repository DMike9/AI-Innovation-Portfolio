"""Source-audited resident driver charges; never relabel these as allocator peaks."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics

from htis_context.phase5b import read,DATA,RESULT,MANIFEST
from htis_context.phase5_data import write,sha


def raw(path):
    return [json.loads(line) for line in gzip.open(path,'rt',encoding='utf-8')]


def main():
    output=RESULT/'memory-audit-v1';output.mkdir(parents=True,exist_ok=False)
    cfg=read('transformer-efficiency/configs/smollm2-135m.json');data={r['id']:r for r in read(DATA)['rows']}
    layers,heads,dim=cfg['layers'],cfg['kv_heads'],cfg['head_dim'];page_size=8
    per_token=layers*heads*dim*2*4
    per_page_features=heads*dim*3*4+heads*4+8
    def metadata(tokens):
        return layers*(tokens*(8+heads*4)+((tokens+7)//8)*per_page_features)
    def dense_tensors(tokens):return tokens*per_token+metadata(tokens)
    quality=raw(RESULT/'evaluation-v1/trials.jsonl.gz');systems=raw(RESULT/'systems-v1/systems.jsonl.gz')
    vocab=read('.model-cache/smollm2-135m/config.json')['vocab_size'];charges=[]
    # Validate the shape-based metadata calculation against ACTUAL full-state
    # measured tensor bytes for every saved full-reference step.
    full_checks=[]
    for row in quality:
        if row['policy']!='full':continue
        for stage in ['teacher_steps','free_steps']:
            for i,step in enumerate(row[stage]):
                tokens=row['input_tokens']+i
                passed=step['active_kv_bytes']==tokens*per_token and step['metadata_bytes']==metadata(tokens)
                full_checks.append(dict(case=row['case'],stage=stage,step=i,passed=passed))
                if not passed:raise AssertionError('Source-derived dense tensor byte formula disagrees with measurement')
    for dataset,rows in [('quality',quality),('systems',systems)]:
        for row in rows:
            source=data[row.get('original_case',row['case'])]
            prefix_tokens=len(source['prefix_ids']);prepared_tokens=source['input_tokens']-1
            prepared_kv=row['free_steps'][0]['input_state_kv_bytes']
            if prepared_kv!=prepared_tokens*per_token:raise AssertionError('Prepared KV measurement differs from source shape')
            prefix_tensors=dense_tensors(prefix_tokens);prepared_tensors=dense_tensors(prepared_tokens)
            current=max(s['known_cache_system_storage_bytes'] for s in row['free_steps'])
            extra=prefix_tensors+prepared_tensors
            components=dict(retained_dense_prefix_tensor_bytes=prefix_tensors,retained_dense_prepared_tensor_bytes=prepared_tensors)
            if dataset=='quality':
                # The outer evaluator retains the final dense teacher state,
                # and quality() retains its final teacher trial state while
                # the separate free-generation request executes.
                final_teacher_tokens=source['input_tokens']+len(source['answer_ids'])-1
                outer_teacher=dense_tensors(final_teacher_tokens)
                inner_teacher=row['teacher_steps'][-1]['known_cache_system_storage_bytes']
                held_logits=(prefix_tokens+len(source['suffix_ids'])-1+len(source['answer_ids'])+2)*vocab*4
                extra+=outer_teacher+inner_teacher+held_logits
                components.update(retained_outer_dense_teacher_tensor_bytes=outer_teacher,
                    retained_inner_teacher_known_storage_bytes=inner_teacher,
                    retained_prefill_and_teacher_logit_tensor_bytes=held_logits)
                prefix_storage=0
            else:
                if row['dense_prefill_kv_bytes']!=prefix_tokens*per_token:raise AssertionError('Actual prefix KV measurement differs')
                prefix_storage=row['prefix_storage_bytes']
                extra+=row['logit_capture_bytes']
                components['own_captured_logit_tensor_bytes']=row['logit_capture_bytes']
            charges.append(dict(dataset=dataset,case=row['case'],policy=row['policy'],budget=row['budget'],recovery=row['recovery'],seed=row['seed'],
                traffic=row.get('traffic'),repeat=row.get('repeat'),known_current_cache_storage_bytes=current,
                independently_stored_prefix_bytes=prefix_storage,driver_retained_bytes=extra,
                source_audited_known_resident_lower_bound_bytes=current+prefix_storage+extra,components=components,
                note='Current cache/index/recipe terms retain labeled structural estimates; added dense tensor/logit payloads follow audited source lifetimes and measured shapes. Unreachable controller cycles, previous numerical-proof buffers, allocator slack and untracked activations are not fully attributed.'))
    grouped=defaultdict(list)
    for row in charges:grouped[(row['dataset'],row['traffic'],row['policy'],row['budget'],row['recovery'])].append(row)
    table=[]
    for key,rows in sorted(grouped.items(),key=lambda x:str(x[0])):
        dataset,traffic,policy,budget,recovery=key
        table.append(dict(dataset=dataset,traffic=traffic,policy=policy,budget=budget,recovery=recovery,
            max_source_audited_known_resident_lower_bound_bytes=max(r['source_audited_known_resident_lower_bound_bytes'] for r in rows),
            mean_driver_retained_bytes=statistics.mean(r['driver_retained_bytes'] for r in rows),
            max_known_current_cache_storage_bytes=max(r['known_current_cache_storage_bytes'] for r in rows)))
    report=dict(passed=all(c['passed'] for c in full_checks),full_metadata_formula_checks=full_checks,rows=charges,table=table,
        primary_manifest_sha256=sha(MANIFEST),source_sha256=sha(Path(__file__)),
        scope='Separately labeled source-accounting audit of primary artifacts; current attention KV is not total resident memory. No changes to raw timings, quality, architecture, protocol or weights.',
        limitations=['Source-audited known resident lower bound, not exact CPU allocator/RSS attribution or isolated cache peak.',
            'Quality driver additionally holds teacher/prefill fixtures and logits; systems driver retains full prefix/prepared states.',
            'OS process counters include model weights, interpreter, allocator and accumulated research metadata; do not attribute all RSS to a cache policy.',
            'Primary current-cache Pareto charts must retain their qualified labels; use these resident charges when discussing overall memory.'])
    write(output/'report.json',report)
    lines=['# Source-audited Phase 5B resident memory','',
        'These lower bounds charge retained dense driver states and diagnostic tensors in addition to current cache/index/recipe estimates. They are not isolated allocator peaks.','',
        '| Dataset/traffic | Policy | Pages | Recovery | Current cache MiB | Charged resident lower bound MiB |',
        '|---|---|---:|---|---:|---:|']
    for row in table:
        lines.append(f'| {row["dataset"]}/{row["traffic"] or "all"} | {row["policy"]} | {row["budget"] or "full"} | {row["recovery"]} | {row["max_known_current_cache_storage_bytes"]/2**20:.3f} | {row["max_source_audited_known_resident_lower_bound_bytes"]/2**20:.3f} |')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'full_metadata_checks':len(full_checks),'audited_trials_and_requests':len(charges)}))


if __name__=='__main__':main()
