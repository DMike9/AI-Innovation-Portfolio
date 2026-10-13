"""Phase 5 controlled CPU pilot extending the validated adapter and policies."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import platform
import statistics
import time
import torch
import transformers

from .adapter import LlamaReferenceAdapter
from .checkpoint import load
from .learned import load_estimator, TrainedPagePolicy
from .offline_labels import kl
from .pages import BaselinePolicy, MemoryBudget
from .phase5_data import sha, write
from .phase5_metrics import answer_metrics
from .phase5_policy import ConditionalHTIS, choose_validation, BenefitCostController, PeriodicBoundController
from .prefix_cache import ExactPrefixCache
from .provenance import SourceRecipe
from .recovery import RecoveryConfig, RecoveryPolicy
from .state import _fingerprint

ROOT=Path('transformer-efficiency')
MODEL=ROOT/'configs/smollm2-135m.json'
TRAIN=ROOT/'results/phase3/trained-v1'
DATA=ROOT/'data/phase5-short-v1/manifest.json'


def definition():
    cfg=json.loads((ROOT/'configs/phase4-small.json').read_text())
    cfg.update(experiment='phase5-controlled-short-cpu-v1',generation_tokens=6,
        trained_seeds=[17,29],warmup_steps=2,system_repeats=2,bootstrap_repetitions=2000)
    cfg['recovery'].update(max_archive_bytes=16*1024**2,max_index_bytes=8*1024**2,max_working_bytes=96*1024**2)
    return {'configuration':cfg,
        'core_arms':['full','recent','window','sinks_recent','random','history','query','deterministic/recovery',
            'htis','htis/recovery','recent/recovery','bound/fallback','conditional-4-.35','cost/recovery','benefit1/recovery','calibrated','simple-selected'],
        'validation_arms':['recent','history','query','random','htis','conditional-4-.35','conditional-4-.8','conditional-2-.35','cost/recovery','recent/recovery','benefit1/recovery','benefit10/recovery'],
        'ablation_arms':['no_attention','no_locality','no_content','no_query','no_value','conditional-2-.35',
            'htis/budget8','recent/budget8','htis/recovery-cooldown0','htis/recovery-always','bound/adaptive','bound/period4','htis/feature-recompute'],
        'seed_two_arms':['htis','htis/recovery','random','no_attention','no_locality','no_content','no_query','no_value'],
        'baseline_labels':{'history':'survivor-only history heuristic, not H2O reproduction',
            'query':'cached coordinate-bound heuristic, Quest inspired, not faithful Quest',
            'window':'pure window; zero sink/prompt protection, separate protection stratum',
            'sinks_recent':'identical to protected recency under common prompt=sink=8; explicit alias',
            'deterministic/recovery':'existing Phase 4 deterministic archive routing, not trained scoring',
            'cost/recovery':'existing query recovery gated by independently measured scoring/copy prior',
            'conditional':'current-query cosine change and fixed interval; first selection scored',
            'calibrated':'validation-selected existing candidate, not a new trained network'},
        'resource_comparison':'All core bounded arms four pages x eight tokens, common prompt/sink/recent=8 except pure window. Archive/index and exact-prefix storage extra. Full unbounded comparator.',
        'unexecuted_components':['Adaptive page budgets do not exist; fixed budget sensitivity only.',
            'Feature-cache recomputation ablation is a single-thread diagnostic hook; no dense-current-attention feature baseline.',
            'No faithful published-method reproduction, batched sparse serving, disk archive, GPU recovery, natural low-mass mining, or Phase 6.'],
        'statistics':'Per-example/session paired bootstrap, seeds averaged within group; tiny exploratory sample.',
        'timing':'CPU perf_counter, four Torch threads, batch=1 concurrency=1. Quality trials reuse common measured prefill fixture, not end-to-end timings. Separate systems repeats perform actual cold/warm requests.'}


def policy(arm,cfg,identity,layers,seed=17,calibration=None):
    if arm=='calibrated':arm=calibration['selected_arm']
    if arm.startswith('simple-selected'):
        arm=calibration['selected_simple_arm']+('/recovery' if arm.endswith('/recovery') else '')
    basekind=arm.split('/')[0]
    common=dict(page_size=cfg['page_size'],max_pages=8 if 'budget8' in arm else cfg['active_pages'],
        prompt_tokens=0 if arm=='window' else cfg['prompt_tokens'],
        sink_tokens=0 if arm=='window' else cfg['sink_tokens'],recent_tokens=cfg['recent_tokens'])
    record={}
    if basekind in {'htis','no_attention','no_locality','no_content','no_query','no_value'} or arm.startswith('conditional'):
        kind=basekind if basekind.startswith('no_') else 'htis'
        estimator,mean,std,record=load_estimator(TRAIN/f'{kind}-seed-{seed}',identity)
        base=TrainedPagePolicy(estimator,mean,std,layer_count=layers,**common)
        if arm.startswith('conditional'):
            _,interval,threshold=arm.split('-');base=ConditionalHTIS(base,interval=int(interval),threshold=float(threshold))
    else:
        kind={'full':'full','random':'prompt_random','history':'attention_history','query':'query_summary',
            'sinks_recent':'sinks_recent'}.get(basekind,'recent')
        base=BaselinePolicy(kind,seed=seed,**common)
    if '/recovery' in arm or basekind in {'bound','cost','deterministic'}:
        rec=RecoveryConfig(**cfg['recovery'])
        if arm.endswith('cooldown0'):rec=replace(rec,cooldown=0)
        if arm.endswith('always'):rec=replace(rec,interval=1,cooldown=0,max_route_ms=1e6)
        if basekind=='deterministic':rec=replace(rec,deterministic=True)
        if basekind=='bound':rec=replace(rec,tolerance=cfg['bound_tolerance'],fallback=arm.endswith('fallback'))
        if basekind=='cost':
            priors=cfg['profile_priors'];rec=replace(rec,score_ms_prior=priors['scorer_per_layer_ms'],
                transfer_ms_per_mb=priors['copy_ms_per_mb'],max_route_ms=priors['route_admission_ms'])
        base=RecoveryPolicy(base,rec,calibration=record)
        if basekind.startswith('benefit'):
            base.recovery=BenefitCostController(base,rec,record,benefit_ms_per_margin=float(basekind[7:]),
                processing_ms_prior=cfg['profile_priors']['controller_per_layer_ms'])
        if arm=='bound/period4':base.recovery=PeriodicBoundController(base,rec,record,every=4)
    return base


def make_adapter(model,dense,arm,cfg,seed,calibration):
    p=policy(arm,cfg,dense.identity,len(model.model.layers),seed,calibration)
    if p.name=='full':return LlamaReferenceAdapter(model,dense.identity,policy=p)
    maximum=p.max_pages*cfg['page_size']*len(model.model.layers)*model.config.num_key_value_heads*model.model.layers[0].self_attn.head_dim*8
    cls=LlamaReferenceAdapter
    if arm=='htis/feature-recompute':
        from .phase5_feature_ablation import RecomputedFeatureAdapter
        cls=RecomputedFeatureAdapter
    return cls(model,dense.identity,policy=p,memory_budget=MemoryBudget(maximum,cfg['recovery']['max_working_bytes']))


def score_model_bytes(p):
    p=p.base if isinstance(p,RecoveryPolicy) else p
    if isinstance(p,ConditionalHTIS):p=p.base
    if not hasattr(p,'estimator'):return 0
    return sum(x.numel()*x.element_size() for x in p.estimator.parameters())+p.mean.numel()*8


def observation(run,p):
    m=run.metrics;r=m.get('recovery',{});s=run.state
    query_bytes=getattr(p,'query_storage_bytes',0)
    return {'forward_ms':m['elapsed_ms'],'selection_ms':m['selection_ms'],'feature_update_ms':m['feature_update_ms'],
        'compaction_copy_bytes':m['compaction_copy_bytes'],'active_kv_bytes':s.kv_bytes,
        'metadata_bytes':m['metadata_tensor_bytes'],'archive_bytes':r.get('archive_storage_bytes',0),
        'index_bytes':r.get('index_estimate_bytes',0),'scorer_bytes':score_model_bytes(p),
        'feature_working_bytes':getattr(p,'feature_bytes',0),'query_storage_bytes':query_bytes,
        'total_storage_bytes':s.kv_bytes+m['metadata_tensor_bytes']+r.get('archive_storage_bytes',0)+r.get('index_estimate_bytes',0)+score_model_bytes(p)+query_bytes,
        'working_tensor_estimate_bytes':r.get('combined_known_working_estimate_bytes',m['estimated_cache_and_attention_working_tensor_bytes'])+query_bytes,
        'active_pages':[x.features.ids.tolist() for x in s.layers], 'recovery':r, 'lineage':s.lineage}


@torch.no_grad()
def quality_trial(row,arm,seed,prefix,teacher,model,dense,tokenizer,cfg,calibration):
    a=make_adapter(model,dense,arm,cfg,seed,calibration)
    state=prefix;teacher_steps=[];nll=[];divergence=[]
    tokens=[row['suffix_ids'][-1]]+row['answer_ids'][:-1]
    for i,token in enumerate(tokens):
        run=a.forward(torch.tensor([[token]]),state=state);state=run.state
        logits=run.logits[:,-1]
        divergence.append(kl(teacher[i],logits))
        nll.append(float(-torch.log_softmax(logits.float(),-1)[0,row['answer_ids'][i]]))
        teacher_steps.append(observation(run,a.policy))
    # Fresh policy/request for free generation: teacher state, outputs and labels
    # never enter routing. No ground-truth teacher attention is computed for scoring.
    a=make_adapter(model,dense,arm,cfg,seed,calibration)
    state=prefix;token=row['suffix_ids'][-1];outputs=[];free=[]
    for _ in range(cfg['generation_tokens']):
        run=a.forward(torch.tensor([[token]]),state=state);state=run.state
        token=int(run.logits[0,-1].argmax());outputs.append(token);free.append(observation(run,a.policy))
    text=tokenizer.decode(outputs,skip_special_tokens=True)
    allsteps=teacher_steps+free
    metrics=answer_metrics(text,row['answers'])
    conditional={}
    if isinstance(a.policy,ConditionalHTIS):conditional={'trigger_ms':a.policy.trigger_ms,'scoring_calls':a.policy.scoring_calls,'events':a.policy.events}
    return {'case':row['id'],'group':row['group'],'kind':row['kind'],'split':row['split'],'arm':arm,'seed':seed,
        'input_tokens':row['input_tokens'],'mean_kl':statistics.mean(divergence),'mean_nll':statistics.mean(nll),
        'kl_per_teacher_token':divergence,'nll_per_teacher_token':nll,'answers':row['answers'],'output_ids':outputs,'raw_output':text,**metrics,
        'mean_decode_ms':statistics.mean(x['forward_ms'] for x in free),'p50_decode_ms':float(torch.tensor([x['forward_ms'] for x in free]).quantile(.5)),
        'p95_decode_ms':float(torch.tensor([x['forward_ms'] for x in free]).quantile(.95)),
        'max_active_kv_bytes':max(x['active_kv_bytes'] for x in allsteps),'max_total_storage_bytes':max(x['total_storage_bytes'] for x in allsteps),
        'teacher_steps':teacher_steps,'free_steps':free,'conditional':conditional}


@torch.no_grad()
def upstream_sdpa(model,ids,outputs,*,atol,rtol,dense):
    """Untouched upstream forward; change only documented attention backend."""
    old=model.config._attn_implementation
    try:
        model.config._attn_implementation='sdpa'
        for _ in range(2):model(torch.tensor([[1,2,3,4,5,6,7,8]]),use_cache=True)
        start=time.perf_counter();out=model(torch.tensor([ids]),use_cache=True)
        prefill=(time.perf_counter()-start)*1000;cache=out.past_key_values;logits=[out.logits[:,-1]];times=[];generated=[]
        for i in range(len(outputs)):
            token=int(out.logits[0,-1].argmax());generated.append(token)
            if i<len(outputs)-1:
                start=time.perf_counter();out=model(torch.tensor([[token]]),past_key_values=cache,use_cache=True)
                times.append((time.perf_counter()-start)*1000);cache=out.past_key_values
        # Teacher-forced equivalence on common tokens, not comparisons between
        # diverged greedy trajectories.
        out=model(torch.tensor([ids]),use_cache=True);cache=out.past_key_values
        dense_state=None;reference=dense  # caller identity checked after restoring below
        upstream=[out.logits[:,-1].detach().clone()]
        for token in outputs[:3]:
            out=model(torch.tensor([[token]]),past_key_values=cache,use_cache=True);cache=out.past_key_values;upstream.append(out.logits[:,-1].detach().clone())
    finally:
        model.config._attn_implementation=old
    ref=reference.forward(torch.tensor([ids]));reference_logits=[ref.logits[:,-1]];dense_state=ref.state
    for token in outputs[:3]:
        ref=reference.forward(torch.tensor([[token]]),state=dense_state);dense_state=ref.state;reference_logits.append(ref.logits[:,-1])
    checks=[{'step':i,'max_abs':float((a-b).abs().max()),'passed':torch.allclose(a,b,atol=atol,rtol=rtol)} for i,(a,b) in enumerate(zip(upstream,reference_logits))]
    if not all(c['passed'] for c in checks):raise AssertionError('Upstream SDPA equivalence failed')
    return {'prefill_ms':prefill,'decode_ms':times,'output_ids':generated,'checks':checks,'kernel':'upstream Transformers SDPA, torch CPU automatic backend'}


@torch.no_grad()
def systems_trial(row,arm,seed,reuse,repeat,model,dense,tokenizer,cfg,calibration,cache,recipe):
    adapter=make_adapter(model,dense,arm,cfg,seed,calibration)
    start=time.perf_counter();lookup_ms=0;prefill_ms=0;hit=False
    if reuse:
        lookup_start=time.perf_counter();state=cache.get(dense.identity,recipe);lookup_ms=(time.perf_counter()-lookup_start)*1000;hit=state is not None
        if state is None:raise AssertionError('Preregistered warm exact cache missed')
    else:
        prefill_start=time.perf_counter();state=dense.forward(torch.tensor([row['prefix_ids']])).state;prefill_ms=(time.perf_counter()-prefill_start)*1000
    suffix_start=time.perf_counter();state=dense.forward(torch.tensor([row['suffix_ids'][:-1]]),state=state).state
    suffix_ms=(time.perf_counter()-suffix_start)*1000
    tokens=[];steps=[];token=row['suffix_ids'][-1];ttft=None
    for i in range(cfg['generation_tokens']):
        run=adapter.forward(torch.tensor([[token]]),state=state);state=run.state
        token=int(run.logits[0,-1].argmax());tokens.append(token);steps.append(observation(run,adapter.policy))
        if i==0:ttft=(time.perf_counter()-start)*1000
    elapsed=(time.perf_counter()-start)*1000
    return {'case':row['id'],'group':row['group'],'arm':arm,'seed':seed,'reuse':reuse,'repeat':repeat,'hit':hit,
        'context_tokens':row['input_tokens'],'output_ids':tokens,'raw_output':tokenizer.decode(tokens,skip_special_tokens=True),
        'prefill_ms':prefill_ms,'lookup_ms':lookup_ms,'question_suffix_ms':suffix_ms,'ttft_ms':ttft,'end_to_end_ms':elapsed,
        'tokens_per_second':cfg['generation_tokens']/(elapsed/1000),'inter_token_ms':[x['forward_ms'] for x in steps[1:]],
        'steps':steps,'prefix_storage_bytes':cache.bytes_used if reuse else 0,'batch_size':1,'concurrency':1}


def append(path,row):
    with gzip.open(path,'at',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')


@torch.no_grad()
def evaluate(mode,output,manifest_path,calibration_path):
    if output.exists():raise FileExistsError('Never overwrite evidence')
    output.mkdir(parents=True);start=time.perf_counter()
    model_cfg=json.loads(MODEL.read_text());model,tok,dense,checkpoint=load(model_cfg,Path('.model-cache/smollm2-135m'))
    manifest=json.loads(manifest_path.read_text());cfg=manifest['configuration'];data=json.loads(DATA.read_text())
    if manifest['data_sha256']!=sha(DATA):raise ValueError('Frozen data changed')
    calibration=json.loads(calibration_path.read_text()) if mode=='test' else None
    if calibration is not None and calibration['experiment_manifest_sha256']!=sha(manifest_path):raise ValueError('Calibration manifest mismatch')
    rows=[r for r in data['rows'] if r['split']==('validation' if mode=='validation' else 'test')]
    trials=[];systems=[];checks=[];sdpa=[];prefix_measurements=[]
    # Warm native matrix operations/adapter, without scoring any evaluation row.
    warm=dense.forward(torch.tensor([[1,2,3,4,5,6,7,8]])).state
    for _ in range(cfg['warmup_steps']):dense.forward(torch.tensor([[9]]),state=warm)
    for number,row in enumerate(rows):
        started=time.perf_counter();prefix=dense.forward(torch.tensor([row['prefix_ids']])).state
        cold_ms=(time.perf_counter()-started)*1000
        started=time.perf_counter();prepared=dense.forward(torch.tensor([row['suffix_ids'][:-1]]),state=prefix).state
        suffix_ms=(time.perf_counter()-started)*1000
        teacher=[];state=prepared
        for token in [row['suffix_ids'][-1]]+row['answer_ids'][:-1]:
            run=dense.forward(torch.tensor([[token]]),state=state);state=run.state;teacher.append(run.logits[:,-1])
        arms=list(manifest['validation_arms'] if mode=='validation' else manifest['core_arms'])
        if mode=='test' and row['id'] in manifest['ablation_cases']:arms+=manifest['ablation_arms']
        for arm in arms:
            seeds=[17,29] if mode=='test' and arm in manifest['seed_two_arms'] else [17]
            for seed in seeds:
                trial=quality_trial(row,arm,seed,prepared,teacher,model,dense,tok,cfg,calibration)
                trials.append(trial);append(output/'trials.jsonl.gz',trial)
                bounded=arm!='full'
                page_count=8 if 'budget8' in arm else cfg['active_pages']
                max_kv=page_count*cfg['page_size']*model_cfg['layers']*model_cfg['kv_heads']*model_cfg['head_dim']*8
                valid=not bounded or trial['max_active_kv_bytes']<=max_kv
                bound_rows=[b for s in trial['teacher_steps']+trial['free_steps'] for b in s['recovery'].get('bounds',[])]
                valid=valid and all(b.get('bound_covers_measured_error',True) for b in bound_rows)
                checks.append({'case':row['id'],'arm':arm,'seed':seed,'passed':valid})
                write(output/'progress.json',{'mode':mode,'completed_trials':len(trials),'case':row['id'],'arm':arm,'elapsed_seconds':time.perf_counter()-start})
        if mode=='test':
            full=next(t for t in trials if t['case']==row['id'] and t['arm']=='full')
            native=upstream_sdpa(model,row['prefix_ids']+row['suffix_ids'],full['output_ids'],atol=model_cfg['atol'],rtol=model_cfg['rtol'],dense=dense)
            native.update(case=row['id'],**answer_metrics(tok.decode(native['output_ids'],skip_special_tokens=True),row['answers']))
            sdpa.append(native)
            if row['id'] in manifest['systems_cases']:
                cache=ExactPrefixCache(64*1024**2);beg=time.perf_counter();admitted=cache.put(prefix);admission_ms=(time.perf_counter()-beg)*1000
                if not admitted:raise AssertionError('Prefix admission failed')
                recipe=prefix.recipe
                modified=list(row['prefix_ids']);modified[-1]=(modified[-1]+1)%model.config.vocab_size
                changed=SourceRecipe((tuple(modified),),recipe.positions,recipe.mask)
                if cache.get(dense.identity,changed) is not None:raise AssertionError('Modified prefix reused stale KV')
                for arm in manifest['factorial_arms']:
                    for repeat in range(cfg['system_repeats']):
                        for reuse in [False,True]:
                            s=systems_trial(row,arm,17,reuse,repeat,model,dense,tok,cfg,calibration,cache,recipe)
                            systems.append(s);append(output/'systems.jsonl.gz',s)
                    current=[s for s in systems if s['case']==row['id'] and s['arm']==arm]
                    checks.append({'case':row['id'],'arm':arm,'name':'reuse_outputs_identical',
                        'passed':all(x['output_ids']==current[0]['output_ids'] for x in current)})
                prefix_measurements.append({'case':row['id'],'admission_ms':admission_ms,'cold_prefix_ms':cold_ms,'suffix_ms':suffix_ms,
                    'prefix_storage_bytes':cache.bytes_used,'modified_prefix_missed':True,'events':cache.events})
        print(f'{mode}: {number+1}/{len(rows)} {row["id"]}; {len(trials)} quality trials, {len(systems)} actual requests',flush=True)
    report={'mode':mode,'manifest_sha256':sha(manifest_path),'configuration':cfg,'execution_identity':asdict(dense.identity),
        'checkpoint_files_sha256':checkpoint['files_sha256'],'environment':{'python':platform.python_version(),'platform':platform.platform(),
            'torch':torch.__version__,'transformers':transformers.__version__,'cuda_available':torch.cuda.is_available(),'threads':torch.get_num_threads()},
        'raw_trials_sha256':sha(output/'trials.jsonl.gz'),'trials':len(trials),'checks':checks,'passed':all(c['passed'] for c in checks),
        'prefix_measurements':prefix_measurements,'upstream_sdpa':sdpa,'wall_seconds':time.perf_counter()-start,
        'model_weight_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
        'limitations':manifest['unexecuted_components']+['CPU FP32 only; memory figures are measured tensor bytes plus stated estimates, not allocator/RSS peaks.',
            'Six-token fixed greedy generation; teacher and free runs independently start from same dense context/question prefix.',
            'Quality trial decode timings exclude shared prefill; separate systems requests include actual cold/warm prefix and suffix computation.',
            'No performance claim against multi-request batching or production serving. Tiny samples do not establish significance.']}
    if mode=='validation':
        selection=choose_validation(trials);selection['experiment_manifest_sha256']=sha(manifest_path)
        candidates=[]
        for name in ['recent','history','query','random']:
            current=[r for r in trials if r['arm']==name]
            candidates.append({'arm':name,**{key:statistics.mean(r[key] for r in current) for key in ['mean_nll','f1','mean_decode_ms']}})
        selection['selected_simple_arm']=min(candidates,key=lambda r:(-r['f1'],r['mean_nll'],r['mean_decode_ms'],r['arm']))['arm']
        selection['simple_table']=candidates
        selection['simple_rule']='Highest validation answer F1, then lowest target NLL, then lowest mean decode time; no final-test selection.'
        write(output/'calibration.json',selection);report['calibration']=selection
    write(output/'report.json',report)
    print(json.dumps({'mode':mode,'passed':report['passed'],'trials':len(trials),'checks':len(checks),'wall_seconds':report['wall_seconds']},indent=2))
    return 0 if report['passed'] else 1


def freeze(output,profile):
    if output.exists():raise FileExistsError('Manifest frozen; use a new version')
    data=json.loads(DATA.read_text());d=definition();d['data_sha256']=sha(DATA)
    test=[r['id'] for r in data['rows'] if r['split']=='test']
    # Fixed predeclared subset: one official QA and one authored opening probe.
    d['systems_cases']=[test[0],'probe/beginning'];d['ablation_cases']=d['systems_cases']
    d['factorial_arms']=['simple-selected','htis','simple-selected/recovery','htis/recovery']
    profile_data=json.loads(profile.read_text());profiles=profile_data['results']
    learned=next(x for x in profiles if x['policy']=='htis/no-recovery')
    recovered=next(x for x in profiles if x['policy']=='recent/recovery')
    # Measured instrumented costs are deliberately conservative admission priors,
    # never reported as uninstrumented transfer timings or a learned benefit model.
    scorer=learned['spans']['learned_selection']['inclusive_ms']/30
    copies=[e for e in recovered['metrics']['recovery']['events'] if e.get('copy_bytes',0)>0]
    copy_prior=sum(e.get('ms',0) for e in copies)/max(1e-12,sum(e['copy_bytes'] for e in copies)/1024**2)
    d['configuration']['profile_priors']={'scorer_per_layer_ms':scorer,'copy_ms_per_mb':copy_prior,
        'controller_per_layer_ms':recovered['spans']['controller_route']['exclusive_ms']/30,
        'route_admission_ms':max(.01,scorer*.5),'source_sha256':sha(profile),
        'meaning':'Conservative measured profile-cost gate; no quality-benefit prediction claimed.'}
    d['model_configuration_sha256']=sha(MODEL)
    d['frozen_before_evaluation']=True
    write(output,d);print(json.dumps({'manifest_sha256':sha(output),'test_cases':test,'profile_priors':d['configuration']['profile_priors']},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','validation','test']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'configs/phase5-cpu-v1.json')
    p.add_argument('--profile',type=Path,default=ROOT/'results/phase5/profile-v1/report.json')
    p.add_argument('--calibration',type=Path,default=ROOT/'results/phase5/validation-v1/calibration.json');a=p.parse_args()
    if a.mode=='freeze':freeze(a.output,a.profile);return 0
    return evaluate(a.mode,a.output,a.manifest,a.calibration)


if __name__=='__main__':raise SystemExit(main())
