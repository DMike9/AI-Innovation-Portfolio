"""Exact execution identity and ordered source recipes, never semantic routing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ExecutionIdentity:
    model_id: str
    revision: str
    config_hash: str
    tokenizer_revision: str
    template_hash: str
    rope_hash: str
    dtype: str
    access_scope: str
    adapters: tuple[str, ...] = ()
    mask_semantics: str = "causal-slot-order+binary-key-padding-v1"
    cache_format: str = "B,KV-head,original-slot,head-dim-v1"
    implementation: str = "htis-context-llama-eager-v1"

    def __post_init__(self):
        object.__setattr__(self, "adapters", tuple(self.adapters))
        if not self.access_scope or not self.model_id:
            raise ValueError("Explicit model identity and access scope required")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Model revision must be an immutable 40-character commit SHA")
        if not re.fullmatch(r"[0-9a-f]{40}", self.tokenizer_revision):
            raise ValueError("Tokenizer revision must be pinned independently")

    @property
    def key(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class SourceRecipe:
    """Complete ordered ancestry, original RoPE positions, and key visibility."""
    tokens: tuple[tuple[int, ...], ...]
    positions: tuple[tuple[int, ...], ...]
    mask: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        for name in ("tokens", "positions", "mask"):
            rows = tuple(tuple(x) for x in getattr(self, name))
            if any(not isinstance(x, int) or isinstance(x, bool) for row in rows for x in row):
                raise ValueError("Recipe token, position and mask values must be integers")
            object.__setattr__(self, name, rows)
        if not self.tokens or not self.tokens[0]:
            raise ValueError("A source recipe must be nonempty")
        n = len(self.tokens[0])
        if len(self.positions) != len(self.tokens) or len(self.mask) != len(self.tokens):
            raise ValueError("Batch dimensions disagree")
        if any(len(row) != n for rows in (self.tokens, self.positions, self.mask) for row in rows):
            raise ValueError("Recipe dimensions disagree")
        if any(x < 0 for row in self.positions for x in row):
            raise ValueError("Original positions must be nonnegative")
        if any(x < 0 for row in self.tokens for x in row):
            raise ValueError("Token IDs must be nonnegative")
        if any(x not in (0, 1) for row in self.mask for x in row):
            raise ValueError("Only binary key-padding masks supported")

    @property
    def key(self) -> str:
        # Entire ordered prefix, not just the last chunk.
        return digest(asdict(self))

    @property
    def length(self) -> int:
        return len(self.tokens[0])


@dataclass(frozen=True)
class SemanticRoute:
    """A separate source-routing object; never accepted by KV restore APIs."""
    source_id: str
    embedding_revision: str
    embedding: tuple[float, ...]
