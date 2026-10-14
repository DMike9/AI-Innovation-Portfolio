"""Descriptive offline score traces across budgets; no primary-run changes."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import time

import torch

from htis_context.phase5b import load_run,read,DATA,RESULT,MANIFEST,make_adapter,verify_frozen,stamp
from htis_context.phase5b_diagnostics import ordinal_correlation
from htis_context.phase5_data import write,sha


@torch.no_grad()
def main():
    output=RESULT/'budget-ranks-v1';output.mkdir(parents=True,exist_ok=False)
    manifest,model,tok,dense,_=load_run();config=manifest['configuration'];records=[];trajectories=0
    for row in [r for r in read(DATA)['rows'] if r['split']=='test']:
        prepared=dense.forward(torch.tensor([row['prefix_ids']+row['suffix_ids'][:-1]])).state
        # Common teacher tokens, ONLY their current/past values reach the scorer.
        tokens=([row['suffix_ids'][-1]]+row['answer_ids'])[:5]
        for budget in [4,8,16]:
            for seed in [17,29]:
                adapter=make_adapter(model,dense,'htis',budget,seed,config)
                policy=adapter.policy;original_select=policy.select
                current={};scores=[]
                def select(slots,history,features,query,*,now):
                    current.update(pages=features.ids.tolist(),layer=policy.layer_cursor,now=now)
                    return original_select(slots,history,features,query,now=now)
                def hook(module,inputs,outputs):
                    if current['layer'] in [0,14,29]:
                        scores.append(dict(case=row['id'],group=row['group'],kind=row['kind'],budget=budget,seed=seed,
                            step=current['step'],layer=current['layer'],now=current['now'],pages=list(current['pages']),
                            scores=outputs.detach().tolist(),normalized_current_features=inputs[0].detach().tolist()))
                policy.select=select
                handle=policy.estimator.register_forward_hook(hook)
                try:
                    state=prepared
                    for step,token in enumerate(tokens):
                        current['step']=step
                        result=adapter.forward(torch.tensor([[token]]),state=state);state=result.state
                finally:
                    handle.remove();policy.select=original_select
                records.extend(scores);trajectories+=1
        print('budget rank traces '+row['id'],flush=True)
    comparisons=[];index={(r['case'],r['budget'],r['seed'],r['step'],r['layer']):r for r in records}
    for key,a in index.items():
        case,budget,seed,step,layer=key
        for other in [4,8,16]:
            if other<=budget:continue
            b=index.get((case,other,seed,step,layer))
            if b is None:continue
            common=sorted(set(a['pages'])&set(b['pages']))
            values=lambda r:[r['scores'][r['pages'].index(page)] for page in common]
            pairs=list(zip(values(a),values(b)))
            comparisons.append(dict(case=case,seed=seed,step=step,layer=layer,budgets=[budget,other],common_pages=len(common),
                ordinal_rank_correlation=ordinal_correlation(values(a),values(b)),
                mean_abs_score_change=sum(abs(x-y) for x,y in pairs)/len(pairs),
                maximum_abs_score_change=max(abs(x-y) for x,y in pairs),
                note='Common input tokens but budget-dependent retained states and downstream queries; not an isolated causal effect of a budget feature.'))
    with gzip.open(output/'score-traces.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in records:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    verify_frozen(manifest)
    report=dict(trajectories=trajectories,score_snapshots=len(records),comparisons=comparisons,
        raw_sha256=sha(output/'score-traces.jsonl.gz'),manifest_sha256=sha(MANIFEST),completed_at_utc=stamp(),passed=True,
        experiment_label='Separately labeled descriptive diagnostic, specified before primary held-out evaluation; not an additional primary quality experiment.',
        scope='Frozen HTIS seeds17/29, every held-out case, budgets4/8/16, first up-to-five common teacher tokens, layers0/14/29; external read-only score hooks; no retraining or scoring changes.',
        timing='Hooked execution is not included in primary performance results.')
    write(output/'report.json',report)
    print(json.dumps(dict(trajectories=trajectories,score_snapshots=len(records),passed=True)))


if __name__=='__main__':main()
