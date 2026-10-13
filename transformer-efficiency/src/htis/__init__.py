"""HTIS-KV: research-grade *simulation* of attention-aware KV retention.

No model accuracy, causal relevance, or GPU speedup is implied by these utilities.
"""

from .cache import CachePolicy, StreamingKVCache, TokenImportance, attention

__all__ = ["CachePolicy", "StreamingKVCache", "TokenImportance", "attention"]
