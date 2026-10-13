"""External, reversible instrumentation of the unchanged Phase 4 execution path.

Profiles validation data only. Timing instrumentation is opt-in and restored;
normal experiments run without these wrappers or PyTorch/cProfile profilers.
"""
from contextlib import contextmanager
from dataclasses import asdict
import cProfile
import functools
import json
from pathlib import Path
import pstats
import time

import torch
from . import adapter, learned, pages, recovery, runtime_features, state, provenance
from .checkpoint import load
from .phase4 import make_policy


class Spans:
    def __init__(self):self.stack=[];self.rows={}

    def wrap(self,name,fn):
        @functools.wraps(fn)
        def call(*args,**kwargs):
            entry=[time.perf_counter(),0.0];self.stack.append(entry)
            try:return fn(*args,**kwargs)
            finally:
                elapsed=(time.perf_counter()-entry[0])*1000;self.stack.pop()
                row=self.rows.setdefault(name,{'calls':0,'inclusive_ms':0.0,'exclusive_ms':0.0})
                row['calls']+=1;row['inclusive_ms']+=elapsed;row['exclusive_ms']+=elapsed-entry[1]
                if self.stack:self.stack[-1][1]+=elapsed
        return call


@contextmanager
def instrument(model,recorder):
    patches=[]
    def patch(obj,key,name):
        old=getattr(obj,key);patches.append((obj,key,old));setattr(obj,key,recorder.wrap(name,old))
    plan=[(learned,'extract','importance_features'),(runtime_features,'extract','importance_features'),
        (runtime_features,'projection','analytic_projection'),(learned.RiskEstimator,'forward','learned_network'),
        (learned.TrainedPagePolicy,'select','learned_selection'),(pages.BaselinePolicy,'select','heuristic_ranking'),
        (adapter,'update_features','insertion_features'),(pages,'update_features','insertion_features'),
        (recovery,'split_pages','split_copy'),(recovery,'join_pages','join_copy'),
        (state.LayerKV,'clone','cache_copy'),(recovery.CPUArchive,'admit','offload'),
        (recovery.CPUArchive,'get','reload'),(recovery.CPUArchive,'integrity','archive_integrity'),
        (recovery,'tensors_sha','page_integrity'),(state,'_fingerprint','state_integrity'),
        (recovery,'_fingerprint','state_integrity'),(recovery.RecoveryController,'route','controller_route'),
        (recovery.RecoveryController,'begin','controller_begin'),(recovery.RecoveryController,'finish','controller_finish'),
        (recovery.RecoveryController,'_bound','omission_bound'),(recovery.RecoveryController,'_stream','blockwise_fallback'),
        (recovery.RecoveryController,'attention','adaptive_attention'),(adapter,'digest','identity_json')]
    try:
        for obj,key,name in plan:patch(obj,key,name)
        # Only base model modules. Learned network timing is separate above.
        for name,module in model.named_modules():
            if isinstance(module,(torch.nn.Linear,torch.nn.Embedding)) or 'RMSNorm' in type(module).__name__ or 'RotaryEmbedding' in type(module).__name__:
                patch(module,'forward','model_'+('linear' if isinstance(module,torch.nn.Linear) else 'embedding' if isinstance(module,torch.nn.Embedding) else 'norm_rope'))
        # These calls belong to attention in this adapter. Feature matmuls use @.
        patch(torch,'matmul','torch_matmul');patch(torch,'softmax','torch_softmax')
        yield recorder
    finally:
        for obj,key,old in reversed(patches):setattr(obj,key,old)


@torch.no_grad()
def run(output:Path):
    output.mkdir(parents=True,exist_ok=False)
    mc=json.loads(Path('transformer-efficiency/configs/smollm2-135m.json').read_text())
    config=json.loads(Path('transformer-efficiency/configs/phase4-small.json').read_text())
    model,tok,dense,_=load(mc,Path('.model-cache/smollm2-135m'))
    corpus=json.loads(Path('transformer-efficiency/data/phase3-public-v1/corpus.json').read_text())
    # Phase 3 validation document, never final held-out Phase 5 labels.
    document=next(x for x in corpus['documents'] if x['split']=='validation')
    text=document.get('text',document.get('extract'))
    if text is None:raise ValueError('Inspect public corpus text schema')
    ids=tok(text,return_tensors='pt').input_ids[:,:112]
    before=time.perf_counter();prefix=dense.forward(ids).state;prefill_ms=(time.perf_counter()-before)*1000
    token=tok(' Continue',add_special_tokens=False).input_ids[0]
    results=[]
    for name in ['recent/no-recovery','htis/no-recovery','recent/recovery','htis/recovery','recent/bound-fallback']:
        policy=make_policy(name,config,dense.identity,mc['layers'],Path('transformer-efficiency/results/phase3/trained-v1'))
        a=adapter.LlamaReferenceAdapter(model,dense.identity,policy=policy)
        timings=[];expected=None
        for _ in range(2):a.forward(torch.tensor([[token]]),state=prefix)
        for _ in range(3):
            begin=time.perf_counter();run=a.forward(torch.tensor([[token]]),state=prefix);timings.append((time.perf_counter()-begin)*1000);expected=run.logits
        spans=Spans();prof=cProfile.Profile()
        begin=time.perf_counter()
        with instrument(model,spans):
            prof.enable();measured=a.forward(torch.tensor([[token]]),state=prefix);prof.disable()
        instrumented_ms=(time.perf_counter()-begin)*1000
        torch.testing.assert_close(measured.logits,expected,atol=mc['atol'],rtol=mc['rtol'])
        stats=pstats.Stats(prof)
        rows=[{'file':key[0],'line':key[1],'function':key[2],'primitive_calls':v[0],'calls':v[1],
               'self_seconds':v[2],'cumulative_seconds':v[3]} for key,v in stats.stats.items()]
        rows.sort(key=lambda x:-x['self_seconds'])
        prof.dump_stats(str(output/(name.replace('/','-')+'.pstats')))
        results.append({'policy':name,'uninstrumented_ms':timings,'instrumented_ms':instrumented_ms,
            'instrumentation_plus_cprofile_overhead_ms':instrumented_ms-sum(timings)/len(timings),
            'spans':spans.rows,'cprofile_top_self':rows[:60],'metrics':measured.metrics})
    # Native operator profile is separate from comparative timings.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True) as native:
        dense.forward(torch.tensor([[token]]),state=prefix)
    operators=[{'operator':e.key,'self_cpu_us':e.self_cpu_time_total,'cpu_us':e.cpu_time_total,
        'calls':e.count,'self_cpu_memory_bytes':e.self_cpu_memory_usage} for e in native.key_averages()]
    operators.sort(key=lambda x:-x['self_cpu_us'])
    report={'scope':'Existing Phase 4 code, validation public document; profiling not benchmark timing',
        'execution_identity':asdict(dense.identity),'source_document':document['id'],'prefix_tokens':ids.shape[1],
        'prefill_ms':prefill_ms,'results':results,'native_dense_operators':operators,
        'notes':['Spans are nested: exclusive time subtracts instrumented children; inclusive times must not be added.',
            'Interpreter/native binding work appears in cProfile self time; unwrapped native work is in forward residual.',
            'Model projections/MLP/norm/RoPE and matmul/softmax attention calls are separate.',
            'Profiler and wrapper overhead measured separately; no theoretical FLOPs substitute for runtime.']}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'prefill_ms':prefill_ms,'profiles':[{k:r[k] for k in ['policy','uninstrumented_ms','instrumented_ms','spans']} for r in results]},indent=2))
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.output)
