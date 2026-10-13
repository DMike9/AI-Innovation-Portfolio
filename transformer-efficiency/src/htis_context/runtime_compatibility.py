"""Explicit source migration for frozen Phase 3 scorers; never KV eligibility.

The reviewed manifest pins both old and new complete source hashes. Only
adapter/loader extensions are allowed; feature/scorer changes still fail closed.
The old code outside marked insertion blocks must reproduce the original hash.
"""
import hashlib
import json
from pathlib import Path
import re


def compatible_source(name, original_sha):
    if name not in {"adapter.py", "learned.py"}:
        return False
    directory = Path(__file__).parent
    manifest = directory / "runtime_compatibility.json"
    if not manifest.is_file():
        return False
    entry = json.loads(manifest.read_text())[name]
    raw = (directory / name).read_bytes()
    stripped = re.sub(rb"^[ \t]*# PHASE4 EXTENSION BEGIN\r?\n.*?^[ \t]*# PHASE4 EXTENSION END\r?\n", b"", raw, flags=re.M | re.S)
    return (entry["original_sha256"] == original_sha
            and hashlib.sha256(raw).hexdigest() == entry["extension_sha256"]
            and hashlib.sha256(stripped).hexdigest() == original_sha)
