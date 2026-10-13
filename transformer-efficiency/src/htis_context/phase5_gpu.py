"""Local CUDA acceptance/systems runner. CPU dry-run is not GPU validation."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import torch
from .adapter import LlamaReferenceAdapter
from .checkpoint import load
from .pages import BaselinePolicy
from .phase5_data import write
from .prefix_cache import ExactPrefixCache


def availability():
    return {'torch':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
        'device_count':torch.cuda.device_count(),'execution_status':'READY' if torch.cuda.is_available() else 'BLOCKED: no compatible local CUDA device/build',
        'cpu_dry_run_is_gpu_execution':False,'archive_recovery_supported':False}


def timed(call, synchronize):
    synchronize();beg=time.perf_counter();result=call();synchronize()
    return result,(time.perf_counter()-beg)*1000


@torch.no_grad()
def run(output,dry_run=False):
    if output.exists():raise FileExistsError('Do not overwrite GPU evidence')
    state=availability()
    if dry_run or not state['cuda_available']:
        state['executed']=False;write(output,state);return state
    cfg=json.loads(Path('transformer-efficiency/configs/smollm2-135m.json').read_text());cfg['device']='cuda'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,tok,reference,manifest=load(cfg,Path('.model-cache/smollm2-135m'))
    identity=replace(reference.identity,implementation=reference.identity.implementation+'/device-cuda-fp32-tf32-off')
    dense=LlamaReferenceAdapter(model,identity)
    ids=tok('The reference memory contains a blue book. '*12,return_tensors='pt').input_ids.to('cuda')
    sync=torch.cuda.synchronize;checks=[];runs=[]
    # Upstream and independent adapter, common tokens, masks, GQA and cached append.
    expected=model(ids,use_cache=True);actual=dense.forward(ids);cache=expected.past_key_values
    reference_logits=[];common_tokens=[]
    for i in range(4):
        reference_logits.append(actual.logits[:,-1].detach().clone())
        checks.append({'step':i,'max_abs':float((expected.logits[:,-1]-actual.logits[:,-1]).abs().max()),
            'passed':torch.allclose(expected.logits[:,-1],actual.logits[:,-1],atol=cfg['atol'],rtol=cfg['rtol'])})
        token=expected.logits[:,-1].argmax(-1)[:,None]
        common_tokens.append(token.detach().clone())
        if i<3:
            expected=model(token,past_key_values=cache,use_cache=True);cache=expected.past_key_values
            actual=dense.forward(token,state=actual.state)
    old=model.config._attn_implementation
    try:
        model.config._attn_implementation='sdpa';native=model(ids,use_cache=True);native_cache=native.past_key_values
        for i,ref in enumerate(reference_logits):
            checks.append({'kernel':'upstream-sdpa','step':i,'max_abs':float((native.logits[:,-1]-ref).abs().max()),
                'passed':torch.allclose(native.logits[:,-1],ref,atol=cfg['atol'],rtol=cfg['rtol'])})
            if i<3:
                native=model(common_tokens[i],past_key_values=native_cache,use_cache=True);native_cache=native.past_key_values
    finally:model.config._attn_implementation=old
    if not all(c['passed'] for c in checks):
        state.update(executed=True,passed=False,checks=checks);write(output,state);return state
    del expected,actual,cache,native,native_cache,reference_logits,common_tokens
    fixture=dense.forward(ids).state;prefix_cache=ExactPrefixCache(64*1024**2)
    admitted,admission_ms=timed(lambda:prefix_cache.put(fixture),sync)
    if not admitted:raise AssertionError('GPU runner exact-prefix admission failed')
    admission_bytes=prefix_cache.bytes_used
    del fixture,prefix_cache
    for name in ['full','recent','htis']:
        if name=='htis':
            from .phase5 import policy
            from .phase5 import definition
            p=policy(name,definition()['configuration'],reference.identity,len(model.model.layers));p.estimator.to('cuda')
        else:p=BaselinePolicy(name,max_pages=4 if name!='full' else None,prompt_tokens=8,sink_tokens=8,recent_tokens=8)
        adapter=LlamaReferenceAdapter(model,identity,policy=p)
        # Cached prefix excludes its last token; processing that token supplies
        # first-token logits without storing logits in the exact KV namespace.
        short=dense.forward(ids[:,:-1]).state
        cache_for_request=ExactPrefixCache(64*1024**2);cache_for_request.put(short);short_recipe=short.recipe;del short
        for reuse in (False,True):
            for repeat in range(5):
                torch.cuda.reset_peak_memory_stats();sync();beg=time.perf_counter()
                if reuse:
                    state_kv,prefill_ms=timed(lambda:cache_for_request.get(identity,short_recipe,device='cuda'),sync)
                    if state_kv is None:raise AssertionError('Exact GPU restore missed')
                else:
                    prefill,prefill_ms=timed(lambda:dense.forward(ids[:,:-1]),sync);state_kv=prefill.state;del prefill
                steps=[];outputs=[];token=int(ids[0,-1])
                for i in range(6):
                    result,ms=timed(lambda:adapter.forward(torch.tensor([[token]],device='cuda'),state=state_kv),sync)
                    state_kv=result.state;token=int(result.logits[0,-1].argmax());outputs.append(token);steps.append(ms)
                    if i==0:ttft_ms=(time.perf_counter()-beg)*1000
                    del result
                sync();end_to_end_ms=(time.perf_counter()-beg)*1000
                if repeat>=2:runs.append({'policy':name,'reuse':reuse,'repeat':repeat-2,'prefix_prefill_or_restore_ms':prefill_ms,
                    'ttft_ms':ttft_ms,'end_to_end_ms':end_to_end_ms,'tokens_per_second':6/(end_to_end_ms/1000),
                    'decode_ms':steps,'output_ids':outputs,'active_kv_bytes':state_kv.kv_bytes,
                    'prefix_cpu_storage_bytes':cache_for_request.bytes_used if reuse else 0,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'batch':1,'concurrency':1})
                del state_kv
        matched=[r for r in runs if r['policy']==name]
        checks.append({'policy':name,'name':'prefix_reuse_same_outputs','passed':all(r['output_ids']==matched[0]['output_ids'] for r in matched)})
    # Upstream optimized SDPA, same device/dtype/input/generation length.
    old=model.config._attn_implementation
    try:
        model.config._attn_implementation='sdpa'
        for repeat in range(5):
            torch.cuda.reset_peak_memory_stats();sync();beg=time.perf_counter()
            out,ms=timed(lambda:model(ids,use_cache=True),sync);cache=out.past_key_values;steps=[];outputs=[]
            for i in range(6):
                token=out.logits[:,-1].argmax(-1)[:,None]
                outputs.append(int(token[0,0]))
                if i<5:
                    out,step_ms=timed(lambda:model(token,past_key_values=cache,use_cache=True),sync);cache=out.past_key_values;steps.append(step_ms)
            sync();end_to_end_ms=(time.perf_counter()-beg)*1000
            if repeat>=2:runs.append({'policy':'upstream-sdpa','repeat':repeat-2,'prefill_ms':ms,'decode_ms':steps,
                'ttft_ms':ms,'end_to_end_ms':end_to_end_ms,'tokens_per_second':6/(end_to_end_ms/1000),'output_ids':outputs,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            del out,cache
    finally:model.config._attn_implementation=old
    state.update(executed=True,passed=all(c['passed'] for c in checks),checks=checks,runs=runs,execution_identity=asdict(identity),
        prefix_admission_ms=admission_ms,prefix_cpu_bytes=admission_bytes,
        checkpoint_files_sha256=manifest['files_sha256'],device=torch.cuda.get_device_name(),
        scope='CUDA acceptance and batch1 dense/retention systems only; no archive recovery or full quality evaluation')
    write(output,state);return state


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    result=run(a.output,a.dry_run);print(json.dumps(result,indent=2))
