"""Bounded request-owned CPU archive and causal, measured-cost routing.

Extends LayerKV/RequestState and the existing adapter/policies. No embeddings,
source reconstruction, dense-teacher attention, training, or global KV store.
CPU FP32 reference only. Immutable archive tensors are private, fingerprinted,
and shared by functional forks; every reload returns an owned copy.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import cached_property
import hashlib
import json
import math
import time

import numpy as np
import torch

from htis.bounded_omission import PageSummary, summarize_page, bound_omitted_pages
from .pages import BaselinePolicy, PageFeatures
from .runtime_features import extract, protected_pages
from .state import LayerKV, _fingerprint
from .provenance import digest


def tensors_sha(layer):
    h = hashlib.sha256()
    for x in (layer.keys, layer.values, layer.slots, layer.history, *layer.features.tensors()):
        h.update(str((x.shape, x.dtype)).encode())
        h.update(x.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def split_pages(layer, size):
    result = {}
    for i, p in enumerate(layer.features.ids.tolist()):
        idx = torch.nonzero(layer.slots // size == p).flatten()
        result[p] = LayerKV(layer.keys[:, :, idx].contiguous(), layer.values[:, :, idx].contiguous(),
                            layer.slots[idx], layer.history[:, idx],
                            layer.features.subset(torch.tensor([i], device=idx.device)))
    return result


def join_pages(parts):
    ordered = [parts[p] for p in sorted(parts)]
    return LayerKV(torch.cat([x.keys for x in ordered], 2), torch.cat([x.values for x in ordered], 2),
        torch.cat([x.slots for x in ordered]), torch.cat([x.history for x in ordered], 1),
        PageFeatures(torch.cat([x.features.ids for x in ordered]),
            *(torch.cat([x.features.tensors()[i] for x in ordered], 1) for i in range(1, 5))))


@dataclass(frozen=True)
class RecoveryConfig:
    max_archive_bytes: int = 8 * 1024 * 1024
    max_index_bytes: int = 4 * 1024 * 1024
    max_working_bytes: int = 64 * 1024 * 1024
    interval: int = 4
    cooldown: int = 2
    query_change: float = 0.35
    risk_trigger: float = 0.5
    uncertainty_trigger: float = 0.5
    ranking_trigger: float = 0.5
    max_route_ms: float = 20.0
    max_fallback_ms: float = 200.0
    transfer_ms_per_mb: float = 0.0
    score_ms_prior: float = 0.1
    tolerance: float | None = None
    max_adaptive_fetches: int = 1
    fallback: bool = False
    recovery: bool = True
    deterministic: bool = False

    def __post_init__(self):
        for n in ("max_archive_bytes", "max_index_bytes", "max_working_bytes", "interval", "cooldown", "max_adaptive_fetches"):
            v = getattr(self, n)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError("Recovery sizes/intervals must be nonnegative integers")
        if self.interval < 1 or min(self.max_index_bytes, self.max_working_bytes) < 1:
            raise ValueError("Positive interval/index/working capacity required")
        for n in ("query_change", "risk_trigger", "uncertainty_trigger", "ranking_trigger", "max_route_ms", "max_fallback_ms", "transfer_ms_per_mb", "score_ms_prior"):
            if not math.isfinite(getattr(self, n)) or getattr(self, n) < 0:
                raise ValueError("Finite nonnegative routing thresholds required")
        if self.tolerance is not None and (not math.isfinite(self.tolerance) or self.tolerance < 0):
            raise ValueError("Finite nonnegative local tolerance required")


@dataclass(frozen=True)
class ArchivedPage:
    layer: int
    page: int
    kv: LayerKV
    sha: str
    identity_key: str
    creation_recipe_key: str
    creation_lineage: str

    @cached_property
    def bytes(self):
        return self.kv.kv_bytes + self.kv.metadata_bytes + len(self.sha + self.identity_key + self.creation_recipe_key + self.creation_lineage) + 32


class CPUArchive:
    def __init__(self, identity, page_size, config):
        self.identity, self.page_size, self.config = identity, page_size, config
        self._pages = {}  # insertion-order FIFO admission/eviction, per layer/page
        self.summaries = {}  # includes permanently discarded pages for complete bound coverage
        self.availability = {}
        self.anchor = None
        self.seal = None
        self.last_queries, self.last_fetch, self.last_scores, self.last_tasks = {}, {}, {}, {}
        self.score_ms, self.transfer_ms_per_mb = config.score_ms_prior, config.transfer_ms_per_mb

    def fork(self):
        other = CPUArchive(self.identity, self.page_size, self.config)
        for n in ("_pages", "summaries", "availability", "last_queries", "last_fetch", "last_scores", "last_tasks"):
            setattr(other, n, getattr(self, n).copy())
        other.anchor, other.seal, other.score_ms, other.transfer_ms_per_mb = self.anchor, self.seal, self.score_ms, self.transfer_ms_per_mb
        return other

    @property
    def bytes(self):
        return sum(p.bytes for p in self._pages.values())

    @property
    def kv_bytes(self):
        return sum(p.kv.kv_bytes for p in self._pages.values())

    @property
    def index_bytes(self):
        arrays = sum(s.key_min.nbytes+s.key_max.nbytes+s.value_center.nbytes+8 for ss in self.summaries.values() for s in ss)
        arrays += sum(q.numel()*q.element_size() for q in self.last_queries.values())
        return arrays + len(repr(self.availability).encode()) + len(repr(self.last_scores).encode()) + 256 * len(self.last_tasks)

    def validate(self, state):
        if state.identity != self.identity or state.page_size != self.page_size or self.anchor != _fingerprint(state):
            raise ValueError("Stale archive identity, scope, ancestry, positions, masks, tensors or computation lineage")
        if self.seal != self.integrity():
            raise ValueError("Archive index/summary/page integrity mismatch")

    def integrity(self):
        h=hashlib.sha256()
        h.update(repr((self.identity,self.page_size,self.config,self.availability)).encode())
        for key,p in self._pages.items():
            h.update(repr((key,p.layer,p.page,p.identity_key,p.creation_recipe_key,p.creation_lineage,p.sha)).encode())
            h.update(tensors_sha(p.kv).encode())
        for key,ss in self.summaries.items():
            h.update(repr(key).encode())
            for s in ss:
                h.update(repr((s.count,s.value_radius)).encode())
                for a in (s.key_min,s.key_max,s.value_center):h.update(a.tobytes())
        return h.hexdigest()

    def get(self, layer, page, identity):
        if identity != self.identity:
            raise ValueError("Archive access scope or execution identity mismatch")
        p = self._pages.get((layer, page))
        if p is None:
            raise KeyError("Page permanently unavailable; exact recomputation is unsupported")
        if p.layer != layer or p.page != page or p.identity_key != identity.key or tensors_sha(p.kv) != p.sha:
            raise ValueError("Archive page provenance/integrity mismatch")
        return p.kv.clone()

    def admit(self, layer, page, kv, recipe, lineage, events):
        began = time.perf_counter()
        copy = kv.clone()
        record = ArchivedPage(layer, page, copy, tensors_sha(copy), self.identity.key, recipe.key, lineage)
        self._pages.pop((layer, page), None)
        while self._pages and self.bytes + record.bytes > self.config.max_archive_bytes:
            key = next(iter(self._pages))
            evicted = self._pages.pop(key)
            self.availability[key] = "discarded"
            events.append({"action": "archive-evict", "layer": key[0], "page": key[1], "bytes": evicted.bytes,
                           "recompute_supported": False})
        admitted = record.bytes <= self.config.max_archive_bytes
        if admitted:
            self._pages[layer, page] = record
        self.availability[layer, page] = "archived" if admitted else "discarded"
        events.append({"action": "offload" if admitted else "discard", "layer": layer, "page": page,
                       "copy_bytes": kv.kv_bytes + kv.metadata_bytes, "ms": (time.perf_counter()-began)*1000})


class RecoveryPolicy:
    """Wrapper of an existing policy, not a second attention/cache engine."""
    def __init__(self, base, config=RecoveryConfig(), *, calibration=None):
        self.base = base
        for n in ("page_size", "max_pages", "prompt_tokens", "sink_tokens", "recent_tokens"):
            setattr(self, n, getattr(base, n))
        if self.max_pages is None:
            raise ValueError("Explicit active-page budget required")
        self.name = "recoverable/" + base.name
        self.recovery = RecoveryController(self, config, calibration)

    def select(self, slots, history, features, query, *, now):
        # Physical selection already performed by the adapter's recovery hook.
        if len(features.ids) > self.max_pages:
            raise MemoryError("Recovery failed active capacity enforcement")
        return torch.arange(len(slots)), torch.arange(len(features.ids))


class RecoveryController:
    def __init__(self, policy, config, calibration):
        self.policy, self.config = policy, config
        self.calibration = calibration or {}
        self.task = "default"  # caller-supplied CURRENT task, not future task/answer

    def begin(self, state, identity, recipe):
        began=time.perf_counter()
        if len(recipe.tokens) != 1:
            raise ValueError("Recovery supports a single request only")
        if state is not None and state.archive is not None:
            state.archive.validate(state)
            if state.archive.config != self.config:
                raise ValueError("Archive configuration cannot change during a request")
            self.archive = state.archive.fork()
        else:
            self.archive = CPUArchive(identity, self.policy.page_size, self.config)
            if state is not None and any(len(x.slots) < state.recipe.length for x in state.layers):
                # Cannot summarize or recover already deleted values.
                for index,layer in enumerate(state.layers):
                    present=set(layer.features.ids.tolist())
                    for page in range((state.recipe.length+self.policy.page_size-1)//self.policy.page_size):
                        if page not in present:self.archive.availability[index,page]='unavailable-before-archive'
        self.identity, self.recipe = identity, recipe
        self.previous_chain = (state.memory_chain or _fingerprint(state)) if state else digest(asdict(identity))
        self.lineage = self.previous_chain
        self.events, self.decisions, self.bounds = [], [], []
        self.controller_ms = self.bound_ms = self.fallback_ms = 0.0
        self.scorer_ms = 0.0
        self.integrity_ms = (time.perf_counter()-began)*1000
        self.peak_buffer_bytes = self.feature_bytes = 0
        self.began = time.perf_counter()

    def _summarize(self, index, page, kv):
        mask = torch.tensor(self.recipe.mask[0])[kv.slots].bool()
        self.archive.summaries[index, page] = tuple(summarize_page(kv.keys[0,h,mask].double().numpy(),
            kv.values[0,h,mask].double().numpy()) for h in range(kv.keys.shape[1])) if mask.any() else ()

    def _fetch(self, index, page, now, action="reload"):
        start = time.perf_counter()
        kv = self.archive.get(index, page, self.identity)
        ms = (time.perf_counter()-start)*1000
        n = kv.kv_bytes + kv.metadata_bytes
        measured = ms * 1e6 / max(n, 1)
        self.archive.transfer_ms_per_mb = max(self.config.transfer_ms_per_mb,
            0.8*self.archive.transfer_ms_per_mb+0.2*measured)
        self.events.append({"action":action,"layer":index,"page":page,"copy_bytes":n,"ms":ms,
                            "original_slots":kv.slots.tolist()})
        self.archive.last_fetch[index, page] = now
        return kv

    def _switch(self, index, parts, chosen, now):
        # Fetch only selected archive pages. Own returned copies before archive
        # eviction can occur; remove their residency before admitting departures.
        fetched = {p:self._fetch(index,p,now) for p in chosen if p not in parts}
        for p in fetched:
            self.archive._pages.pop((index,p))
        for p,kv in parts.items():
            if p not in chosen:
                self.archive.admit(index,p,kv,self.recipe,self.lineage,self.events)
        result = {p:parts[p] if p in parts else fetched[p] for p in chosen}
        for p in result:
            self.archive.availability[index,p] = "active"
        self.peak_buffer_bytes = max(self.peak_buffer_bytes, sum(x.kv_bytes+x.metadata_bytes for x in parts.values())
                                  + sum(x.kv_bytes+x.metadata_bytes for x in result.values()))
        return join_pages(result)

    def route(self, index, active, query, now):
        start = time.perf_counter()
        if query.device.type != "cpu" or query.dtype != torch.float32:
            raise ValueError("Recovery is validated only on CPU FP32")
        parts = split_pages(active,self.policy.page_size)
        for p,kv in parts.items():
            if (index,p) not in self.archive.summaries or p==now//self.policy.page_size:
                self._summarize(index,p,kv)
            self.archive.availability[index,p] = "active"
        # Metadata-only candidate index; no full archived KV concatenation/QK.
        candidates = dict(parts)
        if self.config.recovery:
            candidates.update({p:r.kv for (l,p),r in self.archive._pages.items() if l == index})
        ids = sorted(candidates)
        slots = torch.cat([candidates[p].slots for p in ids])
        history = torch.cat([candidates[p].history for p in ids],1)
        features = PageFeatures(torch.tensor(ids), *(torch.cat([candidates[p].features.tensors()[i] for p in ids],1) for i in range(1,5)))
        protected = protected_pages(slots,now=now,page_size=self.policy.page_size,
            prompt_tokens=self.policy.prompt_tokens,sink_tokens=self.policy.sink_tokens,recent_tokens=self.policy.recent_tokens)
        if len(protected) > self.policy.max_pages:
            raise ValueError("Protected pages exceed active capacity")
        q = query.flatten().clone()
        oldq = self.archive.last_queries.get(index)
        change = 1.0 if oldq is None else float(1-torch.nn.functional.cosine_similarity(q[None],oldq[None]))
        self.archive.last_queries[index] = q
        task_change = self.archive.last_tasks.get(index) != self.task
        self.archive.last_tasks[index] = self.task
        grouped = query.reshape(1,features.key_min.shape[0],-1,1,query.shape[-1])
        scores = (grouped * torch.where(grouped>=0, features.key_max[None,:,None],features.key_min[None,:,None])).sum(-1).amax((0,1,2))/math.sqrt(query.shape[-1])
        score_map = dict(zip(ids,scores.tolist()))
        inactive = [p for p in ids if p not in parts]
        risk = max([score_map[p] for p in inactive],default=-1e9) - max([score_map[p] for p in parts],default=-1e9)
        oldscores = self.archive.last_scores.get(index,{})
        rank_change = max([abs(score_map[p]-oldscores.get(p,score_map[p])) for p in ids],default=0)
        self.archive.last_scores[index] = score_map
        uncertainty = float(self.calibration.get("validation_residual_log_target_radius",0))
        triggers = [n for n,on in [("task-change",task_change),("query-change",change>=self.config.query_change),
            ("interval",now%self.config.interval==0),("risk",risk>=self.config.risk_trigger),
            ("uncertainty",uncertainty>=self.config.uncertainty_trigger),
            ("ranking-change",rank_change>=self.config.ranking_trigger)] if on]
        estimate = self.archive.score_ms + self.archive.transfer_ms_per_mb * active.kv_bytes/1e6
        refresh = bool(triggers) and estimate <= self.config.max_route_ms and self.config.recovery and len(protected)<self.policy.max_pages
        score_ms=0.0
        if not refresh:
            cheap = BaselinePolicy("recent",self.policy.page_size,self.policy.max_pages,
                self.policy.prompt_tokens,self.policy.sink_tokens,self.policy.recent_tokens)
            _,pi = cheap.select(active.slots,active.history,active.features,query,now=now)
            chosen = set(active.features.ids[pi].tolist())
        elif self.config.deterministic:
            free = [p for p in ids if p not in protected]
            shift = (now//self.config.interval)%max(len(free),1)
            chosen = protected | set((free[shift:]+free[:shift])[:self.policy.max_pages-len(protected)])
        else:
            base = self.policy.base
            if base.name in {"recent","sinks_recent"}:
                base = BaselinePolicy("query_summary",self.policy.page_size,self.policy.max_pages,
                        self.policy.prompt_tokens,self.policy.sink_tokens,self.policy.recent_tokens)
            if hasattr(base,"layer_cursor"):
                base.layer_cursor = index
            score_start = time.perf_counter()
            _,pi = base.select(slots,history,features,query,now=now)
            score_ms = (time.perf_counter()-score_start)*1000
            self.scorer_ms+=score_ms
            self.archive.score_ms = 0.8*self.archive.score_ms+0.2*score_ms
            self.feature_bytes = max(self.feature_bytes,getattr(base,"feature_bytes",0))
            chosen = set(features.ids[pi].tolist())
        held = {p for p in parts if now-self.archive.last_fetch.get((index,p),-10**9)<self.config.cooldown}
        held -= protected
        for p in sorted(held):
            if p not in chosen:
                replace = [x for x in chosen if x not in protected and x not in held]
                if replace:
                    chosen.remove(min(replace,key=lambda x:(score_map[x],x))); chosen.add(p)
        self.decisions.append({"layer":index,"now":now,"task":self.task,"triggers":triggers,"refresh":refresh,
            "route":"expensive-refresh" if refresh else "cheap-recency", "query_change":change,"estimated_risk_margin":risk,
            "validation_uncertainty":uncertainty,"ranking_change":rank_change,"estimated_route_ms":estimate,
            "available_archive_pages":len(inactive),"chosen":sorted(chosen),"hysteresis_pages":sorted(held),
            "scorer_ms":score_ms,"recompute_supported":False})
        result = self._switch(index,parts,chosen,now)
        self.controller_ms += (time.perf_counter()-start)*1000
        return result.keys,result.values,result.slots,result.history,result.features

    def _bound(self,index,active,query):
        start=time.perf_counter()
        selected=set(active.features.ids.tolist())
        omitted=[(p,ss) for (l,p),ss in self.archive.summaries.items() if l==index and p not in selected]
        known=sum(sum(s.count for s in ss[:1]) for _,ss in omitted)
        mask=torch.tensor(self.recipe.mask[0])[active.slots].bool()
        coverage=(known+int(mask.sum())==sum(self.recipe.mask[0]))
        if not coverage or not mask.any():
            return None,{"coverage":False,"tolerance_met":False,"reason":"missing summaries or no retained visible keys"}
        values=[]
        groups=query.shape[1]//active.keys.shape[1]
        for h in range(query.shape[1]):
            g=h//groups
            values.append(bound_omitted_pages(query[0,h,0].double().numpy(),active.keys[0,g,mask].double().numpy(),
                active.values[0,g,mask].double().numpy(),[ss[g] for _,ss in omitted if ss]).error_upper_bound)
        self.bound_ms+=(time.perf_counter()-start)*1000
        return max(values),{"coverage":True,"bound_max_head_l2":max(values),"omitted_visible_tokens":known,
            "tolerance_met":max(values)<=self.config.tolerance,"per_head":values}

    def _selected_attention(self,active,query,scaling):
        g=query.shape[1]//active.keys.shape[1]
        logits=(query.reshape(1,active.keys.shape[1],g,-1)@active.keys.transpose(-2,-1))*scaling
        mask=torch.tensor(self.recipe.mask[0])[active.slots].bool()
        logits=logits.masked_fill(~mask[None,None,None],-torch.inf)
        weights=logits.softmax(-1)
        out=weights@active.values
        return weights.unsqueeze(3),out,logits.unsqueeze(3)

    def _stream(self,index,active,query,scaling):
        # One CPU archive page at a time, with numerically stable online softmax.
        # Does not mutate retention or claim to restore historical dense states.
        start=time.perf_counter()
        parts=split_pages(active,self.policy.page_size)
        pages=sorted(p for (l,p) in self.archive.availability if l==index)
        missing=[p for p in pages if p not in parts and (index,p) not in self.archive._pages and self.archive.summaries.get((index,p))]
        covered=set(int(s) for x in parts.values() for s in x.slots.tolist())
        covered.update(int(s) for (l,p),r in self.archive._pages.items() if l==index for s in r.kv.slots.tolist())
        if missing or len(covered)!=self.recipe.length:
            return None,{"fallback":"unavailable","missing_pages":missing,"exact_current_state":False}
        g=query.shape[1]//active.keys.shape[1]
        q=query.reshape(1,active.keys.shape[1],g,-1).double()
        m=torch.full(q.shape[:-1]+(1,),-torch.inf,dtype=torch.float64)
        z=torch.zeros_like(m); acc=torch.zeros_like(q)
        for p in pages:
            if (time.perf_counter()-start)*1000>=self.config.max_fallback_ms:
                self.fallback_ms+=(time.perf_counter()-start)*1000
                return None,{"fallback":"latency-budget","exact_current_state":False}
            block=parts[p] if p in parts else self._fetch(index,p,self.recipe.length-1,"fallback-read")
            buffer_bytes=2*active.kv_bytes+4*block.kv_bytes+q.numel()*8*8+query.shape[1]*len(block.slots)*8*4
            if buffer_bytes>self.config.max_working_bytes:
                return None,{"fallback":"working-budget","exact_current_state":False}
            self.peak_buffer_bytes=max(self.peak_buffer_bytes,buffer_bytes)
            visible=torch.tensor(self.recipe.mask[0])[block.slots].bool()
            if not visible.any():
                continue
            logits=(q@block.keys[:,:,visible].double().transpose(-2,-1))*scaling
            new_m=torch.maximum(m,logits.amax(-1,keepdim=True))
            factor=(m-new_m).exp(); e=(logits-new_m).exp()
            acc=acc*factor+e@block.values[:,:,visible].double()
            z=z*factor+e.sum(-1,keepdim=True);m=new_m
        self.fallback_ms+=(time.perf_counter()-start)*1000
        if not (z>0).all():
            return None,{"fallback":"no-visible-keys","exact_current_state":False}
        return (acc/z).float(),{"fallback":"exact-current-state-blockwise","exact_current_state":True,
            "historical_dense_repair":False,"log_z":(m+z.log()).squeeze(-1)}

    def attention(self,index,active,query,weights,out,logits,scaling):
        if self.config.tolerance is None:
            return active.keys,active.values,active.slots,active.history,active.features,weights,out,logits
        bound,record=self._bound(index,active,query)
        record.update(layer=index,initial_bound=bound,adaptive_fetches=0)
        start=time.perf_counter()
        visited=set()
        while bound is not None and bound>self.config.tolerance and record['adaptive_fetches']<self.config.max_adaptive_fetches:
            if not self.config.recovery or (time.perf_counter()-start)*1000>=self.config.max_route_ms:
                record['stop_reason']='latency-or-recovery-budget';break
            candidates=[p for (l,p) in self.archive._pages if l==index and p not in visited]
            parts=split_pages(active,self.policy.page_size)
            protected=protected_pages(active.slots,now=self.recipe.length-1,page_size=self.policy.page_size,
                prompt_tokens=self.policy.prompt_tokens,sink_tokens=self.policy.sink_tokens,recent_tokens=self.policy.recent_tokens)
            free=[p for p in parts if p not in protected and
                  self.recipe.length-1-self.archive.last_fetch.get((index,p),-10**9)>=self.config.cooldown]
            if not candidates or (not free and len(parts)>=self.policy.max_pages):
                record['stop_reason']='no-fetch-or-active-capacity';break
            scores=self.archive.last_scores[index]
            priority_start=time.perf_counter()
            priority={p:self.fetch_priority(index,p,query,out) for p in candidates}
            self.bound_ms+=(time.perf_counter()-priority_start)*1000
            chosen=set(parts);p=max(candidates,key=lambda x:(priority[x],x));visited.add(p)
            if len(parts)>=self.policy.max_pages:
                chosen.remove(min(free,key=lambda x:(scores.get(x,-1e9),x)))
            chosen.add(p)
            estimated=self.archive.transfer_ms_per_mb*sum(r.kv.kv_bytes for (l,x),r in self.archive._pages.items() if l==index and x==p)/1e6
            if estimated+(time.perf_counter()-start)*1000>self.config.max_route_ms:
                record['stop_reason']='estimated-transfer-budget';break
            active=self._switch(index,parts,chosen,self.recipe.length-1)
            self.events.append({'action':'adaptive-choice','layer':index,'page':p,
                'log_value_sensitive_priority':priority[p],'hysteresis_respected':True})
            weights,out,logits=self._selected_attention(active,query,scaling)
            record['adaptive_fetches']+=1
            bound,new_record=self._bound(index,active,query);record.update(new_record)
        if bound is not None and bound>self.config.tolerance and self.config.fallback:
            exact,diagnostic=self._stream(index,active,query,scaling)
            if exact is not None:
                error=torch.linalg.vector_norm(exact.double()-out.double(),dim=-1).flatten().tolist()
                record['measured_current_state_error_per_head']=error
                record['bound_covers_measured_error']=all(e<=b+1e-6 for e,b in zip(error,record['per_head']))
                out=exact
                z=diagnostic.pop('log_z')
                weights=(logits.squeeze(3).double()-z[:,:,:,None]).exp().float().unsqueeze(3)
                record['tolerance_met']=True
                record['residual_after_fallback']=0.0
            record.update(diagnostic)
        record['final_bound_before_fallback']=bound
        self.bounds.append(record)
        return active.keys,active.values,active.slots,active.history,active.features,weights,out,logits

    def fetch_priority(self,index,page,query,out):
        """Largest per-head relaxed numerator contribution: log(u_p * D_p).

        Uses only cached omitted-page summaries and the computed retained output;
        no archived dense QK pass and no guarantee of optimal joint allocation.
        """
        summaries=self.archive.summaries[index,page]
        if not summaries:return -1e300
        groups=query.shape[1]//len(summaries);priorities=[]
        for h in range(query.shape[1]):
            q=query[0,h,0].double().numpy();s=summaries[h//groups]
            y=out[0,h//groups,h%groups].double().numpy()
            log_u=math.log(s.count)+float(q@np.where(q>=0,s.key_max,s.key_min))/math.sqrt(len(q))
            distance=float(np.linalg.norm(s.value_center-y)+s.value_radius)
            priorities.append(log_u+math.log(max(distance,1e-300)))
        return max(priorities)

    def finish(self,state):
        start=time.perf_counter()
        if self.archive.bytes>self.config.max_archive_bytes or self.archive.index_bytes>self.config.max_index_bytes:
            raise MemoryError("Archive/index budget exceeded")
        persistent=state.kv_bytes+self.archive.bytes+self.archive.index_bytes
        if persistent+self.peak_buffer_bytes+self.feature_bytes>self.config.max_working_bytes:
            raise MemoryError("Recovery tensor/index working estimate exceeds budget")
        self.archive.seal=self.archive.integrity()
        state.memory_chain=digest([self.previous_chain,self.recipe.key,self.archive.seal,_fingerprint(state)])
        self.archive.anchor=_fingerprint(state)
        self.integrity_ms+=(time.perf_counter()-start)*1000
        return self.archive

    def validate_working(self,metrics,state):
        scorer=self.policy.base
        scorer_bytes=(sum(x.numel()*x.element_size() for x in scorer.estimator.parameters())
            +scorer.mean.numel()*scorer.mean.element_size()+scorer.std.numel()*scorer.std.element_size()) if hasattr(scorer,'estimator') else 0
        known=(metrics['estimated_cache_and_attention_working_tensor_bytes']+self.archive.bytes+self.archive.index_bytes
               +self.peak_buffer_bytes+self.feature_bytes+scorer_bytes)
        metrics['recovery']['combined_known_working_estimate_bytes']=known
        if known>self.config.max_working_bytes:
            raise MemoryError('Combined cache/archive/index/scorer/attention working estimate exceeds budget')

    def report(self):
        return {"configuration":asdict(self.config),"events":self.events,"decisions":self.decisions,"bounds":self.bounds,
            "archive_kv_bytes":self.archive.kv_bytes,"archive_storage_bytes":self.archive.bytes,"index_estimate_bytes":self.archive.index_bytes,
            "controller_ms":self.controller_ms,"scorer_ms":self.scorer_ms,"integrity_ms":self.integrity_ms,
            "bound_ms":self.bound_ms,"fallback_ms":self.fallback_ms,
            "peak_recovery_buffer_estimate_bytes":self.peak_buffer_bytes,"scorer_feature_bytes":self.feature_bytes,
            "transferred_copy_bytes":sum(x.get('copy_bytes',0) for x in self.events),
            "fetch_count":sum(x['action']=='reload' for x in self.events),
            "offload_count":sum(x['action']=='offload' for x in self.events),
            "discarded_pages":sum(x=='discarded' for x in self.archive.availability.values()),
            "unavailable_before_archive_pages":sum(x=='unavailable-before-archive' for x in self.archive.availability.values()),
            "cpu_copy_not_gpu_transfer":True,"recomputation_supported":False}
