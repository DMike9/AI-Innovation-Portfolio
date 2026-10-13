"""Offline actual-attention diagnostic of the frozen supplied-state stress case.

Never imported by runtime routing and never feeds policy selection or tuning.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from htis_context.adapter import LlamaReferenceAdapter
from htis_context.checkpoint import load
from htis_context.offline_labels import CapturePolicy, local_damage
from htis_context.phase4 import perturb_value_page


def main():
    root=Path.cwd();d=root/'transformer-efficiency'
    report_path=d/'results/phase4/evaluation-v3/report.json'
    report=json.loads(report_path.read_text())
    w=next(x for x in report['workloads'] if x['perturbation'])
    config=json.loads((d/'configs/smollm2-135m.json').read_text())
    model,tokenizer,dense,_=load(config,Path('.model-cache/smollm2-135m'))
    with torch.inference_mode():
        recipe=w['prefix_recipe']
        prefix=dense.forward(torch.tensor(recipe['tokens']),position_ids=torch.tensor(recipe['positions']),
                             attention_mask=torch.tensor(recipe['mask'])).state
        capture=CapturePolicy(report['configuration']['page_size'],config['layers'])
        actual=LlamaReferenceAdapter(model,dense.identity,policy=capture).forward(torch.tensor([[w['tokens'][0]]]),state=prefix)
        layer,page=14,1;kv=actual.state.layers[layer];q=capture.calls[layer]['query']
        groups=config['query_heads']//config['kv_heads']
        weights=(q.reshape(1,config['kv_heads'],groups,config['head_dim'])@kv.keys.transpose(-2,-1)/config['head_dim']**0.5).double().softmax(-1)
        idx=kv.slots//report['configuration']['page_size']==page
        masses=weights[:,:,:,idx].sum(-1).flatten().tolist()
        original_damage=local_damage(q,kv,(page,),report['configuration']['page_size'])
        original_mean_norm=float(kv.values[:,:,idx].norm(dim=-1).mean())
        perturb_value_page(prefix,layer,page,w['perturbation']['scale'],report['configuration']['page_size'])
        stress=LlamaReferenceAdapter(model,dense.identity).forward(torch.tensor([[w['tokens'][0]]]),state=prefix)
        changed_damage=local_damage(q,stress.state.layers[layer],(page,),report['configuration']['page_size'])
        result={'scope':'Offline fixed-state first-query diagnostic, not runtime features or validated answers',
            'source_report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'execution_identity':asdict(dense.identity),
            'layer':layer,'page':page,'value_scale':w['perturbation']['scale'],'natural_model_state':False,
            'per_query_head_page_mass':masses,'mean_page_mass':sum(masses)/len(masses),'max_page_mass':max(masses),
            'uniform_page_fraction':int(idx.sum())/len(kv.slots),'original_mean_value_norm':original_mean_norm,
            'scaled_mean_value_norm':float(stress.state.layers[layer].values[:,:,idx].norm(dim=-1).mean()),
            'unscaled_omission_local_mean_l2':original_damage,'scaled_omission_local_mean_l2':changed_damage,
            'note':'Scaling V at layer 14 does not change that layer query or attention logits on this first step. Later queries can diverge.'}
    out=d/'results/phase4/evaluation-v3/diagnostics/outlier-first-query.json'
    if out.exists():raise FileExistsError('Never overwrite a diagnostic')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
