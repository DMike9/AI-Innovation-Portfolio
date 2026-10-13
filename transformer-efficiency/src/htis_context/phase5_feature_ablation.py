"""Single-thread CPU experiment only: disable insertion feature reuse.

Restores the module hook after every call. This diagnostic is deliberately
excluded from concurrent serving; it does not change the validated source.
It summarizes physically available KV, never computes dense attention scores.
"""
from contextlib import contextmanager
from . import adapter as module
from .adapter import LlamaReferenceAdapter


@contextmanager
def recompute_features():
    original=module.update_features
    def uncached(keys,values,slots,page_size,previous,new_slots):
        return original(keys,values,slots,page_size,None,new_slots)
    module.update_features=uncached
    try:yield
    finally:module.update_features=original


class RecomputedFeatureAdapter(LlamaReferenceAdapter):
    def forward(self,*args,**kwargs):
        with recompute_features():return super().forward(*args,**kwargs)
