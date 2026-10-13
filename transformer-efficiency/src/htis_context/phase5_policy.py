"""Optional cheap-trigger wrapper around unchanged Phase 3 scoring and recency."""
import time
import torch
from .pages import BaselinePolicy
from dataclasses import replace
from .recovery import RecoveryController


class ConditionalHTIS:
    name='conditional-htis'
    def __init__(self, base, *, interval=4, threshold=.35):
        if interval<1 or not 0<=threshold<=2:
            raise ValueError('Invalid conditional scoring threshold/interval')
        self.base=base
        for name in ('page_size','max_pages','prompt_tokens','sink_tokens','recent_tokens','layer_count'):
            setattr(self,name,getattr(base,name))
        self.cheap=BaselinePolicy('recent',page_size=self.page_size,max_pages=self.max_pages,
            prompt_tokens=self.prompt_tokens,sink_tokens=self.sink_tokens,recent_tokens=self.recent_tokens)
        self.interval,self.threshold=interval,threshold
        self.previous={};self.cursor=0;self.events=[];self.trigger_ms=0.;self.scoring_calls=0

    @torch.no_grad()
    def select(self,slots,history,features,query,*,now):
        start=time.perf_counter();layer=self.cursor;self.cursor=(layer+1)%self.layer_count
        current=query.flatten().float();previous=self.previous.get(layer)
        change=1. if previous is None else max(0.,min(2.,float(1-torch.nn.functional.cosine_similarity(current,previous,dim=0))))
        self.previous[layer]=current.detach().clone()
        # Request's first compaction is scored. Later scores require interval and
        # changed CURRENT query; no benchmark task labels are accepted.
        activate=previous is None or (now%self.interval==0 and change>=self.threshold)
        elapsed=(time.perf_counter()-start)*1000;self.trigger_ms+=elapsed
        self.events.append({'layer':layer,'now':now,'query_change':change,'activate':activate,'trigger_ms':elapsed})
        if activate:
            self.base.layer_cursor=layer
            self.scoring_calls+=1
            return self.base.select(slots,history,features,query,now=now)
        return self.cheap.select(slots,history,features,query,now=now)

    @property
    def feature_bytes(self):
        return self.base.feature_bytes+sum(t.numel()*t.element_size() for t in self.previous.values())

    @property
    def query_storage_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.previous.values())


def choose_validation(rows, tolerance=.02):
    """Choose fastest feasible candidate using ONLY validation records."""
    if not rows or any(r['split']!='validation' for r in rows):
        raise ValueError('Calibration accepts validation rows only')
    names=sorted({r['arm'] for r in rows});table=[]
    reference=[r for r in rows if r['arm']=='recent']
    if not reference:
        raise ValueError('Protected-recency validation reference required')
    refs={(r['case'],r['seed']):r for r in reference}
    for name in names:
        selected=[r for r in rows if r['arm']==name]
        if {(r['case'],r['seed']) for r in selected}!=set(refs):
            raise ValueError('Candidates must cover identical validation pairs')
        delta=sum(r['mean_kl']-refs[(r['case'],r['seed'])]['mean_kl'] for r in selected)/len(selected)
        f1=sum(r['f1']-refs[(r['case'],r['seed'])]['f1'] for r in selected)/len(selected)
        latency=sum(r['mean_decode_ms'] for r in selected)/len(selected)
        table.append({'arm':name,'mean_kl_delta':delta,'mean_f1_delta':f1,'mean_decode_ms':latency,
            'feasible':delta<=tolerance and f1>=0})
    chosen=min((x for x in table if x['feasible']),key=lambda x:(x['mean_decode_ms'],x['arm']))
    return {'selected_arm':chosen['arm'],'table':table,'rule':f'Fastest mean decode with validation KL delta <= {tolerance} and F1 delta >= 0',
            'validation_case_ids':sorted({r['case'] for r in rows})}


class BenefitCostController(RecoveryController):
    """Risk-margin proxy converted to ms by a declared validation candidate.

    This is an ablatable heuristic, not a calibrated correctness prediction.
    It extends the existing route, offload, lineage and integrity machinery.
    """
    def __init__(self,policy,config,calibration,*,benefit_ms_per_margin,processing_ms_prior):
        super().__init__(policy,config,calibration)
        self.benefit_scale=benefit_ms_per_margin;self.processing_prior=processing_ms_prior

    def route(self,index,active,query,now):
        began=time.perf_counter()
        grouped=query.reshape(1,active.features.key_min.shape[0],-1,1,query.shape[-1])
        def upper(features):
            return float((grouped*torch.where(grouped>=0,features.key_max[None,:,None],features.key_min[None,:,None])).sum(-1).amax())/query.shape[-1]**.5
        current=upper(active.features)
        archived=[upper(p.kv.features) for (l,_),p in self.archive._pages.items() if l==index]
        margin=max(0.,max(archived,default=current)-current)
        benefit=self.benefit_scale*margin
        cost=self.processing_prior+self.archive.score_ms+self.archive.transfer_ms_per_mb*active.kv_bytes/1e6
        enabled=bool(archived) and benefit>cost
        overhead=(time.perf_counter()-began)*1000
        old=self.config
        try:
            if not enabled:self.config=replace(old,recovery=False)
            result=super().route(index,active,query,now)
        finally:
            self.config=old
        self.controller_ms+=overhead
        self.decisions[-1].update(benefit_cost_gate=enabled,estimated_benefit_proxy_ms=benefit,
            estimated_cost_ms=cost,benefit_gate_ms=overhead,benefit_proxy_is_quality_guarantee=False)
        return result


class PeriodicBoundController(RecoveryController):
    """Bounds evaluated every N tokens. Skipped steps have NO bound guarantee."""
    def __init__(self,policy,config,calibration,*,every):
        super().__init__(policy,config,calibration);self.every=every

    def attention(self,index,active,query,weights,out,logits,scaling):
        if (self.recipe.length-1)%self.every:
            self.bounds.append({'layer':index,'coverage':False,'tolerance_met':False,'reason':'bound-frequency-skip',
                'bound_evaluated':False,'exact_current_state':False})
            return active.keys,active.values,active.slots,active.history,active.features,weights,out,logits
        return super().attention(index,active,query,weights,out,logits,scaling)
