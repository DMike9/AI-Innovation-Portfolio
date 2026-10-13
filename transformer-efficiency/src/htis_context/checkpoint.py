"""Pinned public checkpoint acquisition and strict local loading (no remote code)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import LlamaReferenceAdapter
from .provenance import ExecutionIdentity, digest


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download(config: dict, directory: Path) -> dict:
    from huggingface_hub import snapshot_download
    # No tokens, remote inference endpoints, or paid resources used.
    snapshot_download(config["model_id"], revision=config["revision"], local_dir=directory,
                      allow_patterns=["*.json", "*.txt", "README.md", "model.safetensors"])
    files = {p.name: file_sha256(p) for p in directory.iterdir() if p.is_file()
             and p.name != "htis-checkpoint.json"}
    manifest = {"model_id": config["model_id"], "revision": config["revision"],
                "license": config["license"], "files_sha256": files,
                "model_card": config["model_card"], "official_config": config["official_config"]}
    (directory / "htis-checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load(config: dict, directory: Path, *, scope: str = "local-public-research"):
    manifest_path = directory / "htis-checkpoint.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pinned pretrained weights unavailable: {manifest_path}; run download first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(manifest[k] != config[k] for k in ("model_id", "revision")):
        raise ValueError("Checkpoint revision/identity differs from pinned configuration")
    for name, sha in manifest["files_sha256"].items():
        if Path(name).name != name or file_sha256(directory / name) != sha:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    dtype = getattr(torch, config["dtype"])
    torch.set_num_threads(config["threads"])
    model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True,
                trust_remote_code=False, dtype=dtype, attn_implementation="eager").to(config["device"]).eval()
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    expected = {"max_position_embeddings": "context_length", "num_attention_heads": "query_heads",
                "num_key_value_heads": "kv_heads", "num_hidden_layers": "layers",
                "rope_theta": "rope_theta", "rope_scaling": "rope_scaling"}
    if any(getattr(model.config, field) != config[key] for field, key in expected.items()):
        raise ValueError("Official model configuration differs from reviewed model selection")
    if model.config.hidden_size // model.config.num_attention_heads != config["head_dim"]:
        raise ValueError("Unexpected physical head layout")
    identity = ExecutionIdentity(model_id=config["model_id"], revision=config["revision"],
                config_hash=digest(model.config.to_dict()), tokenizer_revision=config["revision"],
                template_hash=digest(tokenizer.chat_template),
                rope_hash=digest({"theta": model.config.rope_theta, "scaling": model.config.rope_scaling}),
                dtype=str(dtype), access_scope=scope,
                implementation=f"transformers-{transformers.__version__}/torch-{torch.__version__}/"
                               f"package-{digest({p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})}")
    return model, tokenizer, LlamaReferenceAdapter(model, identity), manifest
