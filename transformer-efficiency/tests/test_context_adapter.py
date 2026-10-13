"""Offline unit tests with SMALL RANDOM models; not pretrained model evidence."""
from dataclasses import replace
import os
from pathlib import Path
import json

import pytest

torch = pytest.importorskip("torch", reason="Install optional .[context] dependencies for adapter tests")
pytest.importorskip("transformers", reason="Install optional .[context] dependencies for adapter tests")
from transformers import LlamaConfig, LlamaForCausalLM

from htis_context.adapter import LlamaReferenceAdapter
from htis_context.provenance import ExecutionIdentity, SourceRecipe, digest
from htis_context.state import PrefixSnapshot


@pytest.fixture
def random_adapter():
    return make_random_adapter(2)


def make_random_adapter(kv_heads):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = LlamaConfig(vocab_size=43, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=kv_heads,
                        max_position_embeddings=128, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    identity = ExecutionIdentity("random-unit-fixture", "0" * 40, digest(config.to_dict()),
                                 "0" * 40, digest(None), digest({"theta": config.rope_theta, "scaling": None}),
                                 "torch.float32", "unit-test")
    return LlamaReferenceAdapter(model, identity)


@torch.no_grad()
def test_random_MHA_layout():
    adapter = make_random_adapter(4)
    ids = torch.tensor([[1, 2, 3, 4]])
    actual = adapter.forward(ids)
    expected = adapter.model(ids, use_cache=False).logits
    torch.testing.assert_close(actual.logits, expected, atol=1e-6, rtol=1e-5)
    assert actual.state.layers[0].keys.shape[1] == 4


@pytest.mark.parametrize("offset", [0, 7, 60])
@torch.inference_mode()
def test_random_unit_dense_and_cached_equivalence(random_adapter, offset):
    a = random_adapter
    ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    pos = torch.arange(5)[None].expand(2, -1) + offset
    expected = a.model(ids, position_ids=pos, use_cache=False).logits
    actual = a.forward(ids, position_ids=pos)
    torch.testing.assert_close(actual.logits, expected, atol=1e-6, rtol=1e-5)
    head = a.forward(ids[:, :3], position_ids=pos[:, :3])
    before = head.state.layers[0].keys.clone()
    tail = a.forward(ids[:, 3:], state=head.state, position_ids=pos[:, 3:])
    torch.testing.assert_close(tail.logits, expected[:, 3:], atol=1e-6, rtol=1e-5)
    assert tail.state.recipe.length == 5
    assert torch.equal(before, head.state.layers[0].keys)
    assert all(x.keys.shape == (2, 2, 5, 8) for x in tail.state.layers)
    assert tail.state.kv_bytes == 2 * 2 * 2 * 5 * 8 * 4 * 2


@pytest.mark.parametrize("mask", [[[0, 0, 1, 1, 1]], [[1, 1, 1, 0, 0]], [[1, 0, 1, 1, 1]]])
@torch.inference_mode()
def test_random_unit_padding_and_masks(random_adapter, mask):
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.tensor(mask)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    actual = random_adapter.forward(ids, position_ids=pos, attention_mask=mask)
    expected = random_adapter.model(ids, position_ids=pos, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual.logits[mask.bool()], expected[mask.bool()], atol=1e-6, rtol=1e-5)


def test_snapshot_restore_isolation_and_integrity(random_adapter):
    state = random_adapter.forward(torch.tensor([[1, 2, 3]])).state
    snapshot = PrefixSnapshot(state)
    a = snapshot.restore(state.identity, state.recipe)
    b = snapshot.restore(state.identity, state.recipe)
    a.layers[0].keys.zero_()
    assert torch.equal(b.layers[0].keys, state.layers[0].keys)
    with pytest.raises(ValueError):
        snapshot.restore(replace(state.identity, access_scope="another"), state.recipe)
    snapshot._state.layers[0].values.zero_()
    with pytest.raises(ValueError, match="integrity"):
        snapshot.restore(state.identity, state.recipe)


def test_safe_snapshot_restart_reload(random_adapter, tmp_path):
    a = random_adapter
    state = a.forward(torch.tensor([[1, 2, 3]]), position_ids=torch.tensor([[37, 38, 39]])).state
    snapshot = PrefixSnapshot(state)
    directory = tmp_path / "snapshot"
    snapshot.save(directory)
    with pytest.raises(FileExistsError):
        snapshot.save(directory)
    with pytest.raises(ValueError, match="scope"):
        PrefixSnapshot.load(directory, replace(state.identity, access_scope="other-tenant"), state.recipe)
    loaded = PrefixSnapshot.load(directory, state.identity, state.recipe)
    restored = loaded.restore(state.identity, state.recipe)
    torch.testing.assert_close(a.forward(torch.tensor([[4]]), state=restored, position_ids=torch.tensor([[40]])).logits,
                               a.forward(torch.tensor([[1, 2, 3, 4]]), position_ids=torch.tensor([[37, 38, 39, 40]])).logits[:, 3:],
                               atol=1e-6, rtol=1e-5)
    with (directory / "provenance.json").open(encoding="utf-8") as stream:
        record = json.load(stream)
    record["fingerprint"] = "tampered"
    (directory / "provenance.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        PrefixSnapshot.load(directory, state.identity, state.recipe)


def test_approximate_state_cannot_be_saved_as_exact(random_adapter):
    state = random_adapter.forward(torch.tensor([[1, 2, 3]])).state
    state.lineage = "approximate:intervention-1"
    with pytest.raises(ValueError, match="Approximate"):
        PrefixSnapshot(state)


@pytest.mark.parametrize("kwargs", [
    {"attention_mask": torch.tensor([[1, 0.5]])},
    {"attention_mask": torch.ones((1, 1, 2, 2))},
    {"position_ids": torch.tensor([[1, 128]])},
    {"position_ids": torch.tensor([[-1, 0]])},
    {"position_ids": torch.tensor([[1., 2.]])},
])
def test_unsupported_inputs_fail_explicitly(random_adapter, kwargs):
    with pytest.raises(ValueError):
        random_adapter.forward(torch.tensor([[1, 2]]), **kwargs)


def test_changed_past_mask_rejected(random_adapter):
    state = random_adapter.forward(torch.tensor([[1, 2]])).state
    with pytest.raises(ValueError, match="Earlier masks"):
        random_adapter.forward(torch.tensor([[3]]), state=state, attention_mask=torch.tensor([[0, 1, 1]]))


def test_weight_and_config_mutation_invalidate_execution(random_adapter):
    a = random_adapter
    state = a.forward(torch.tensor([[1, 2]])).state
    with torch.no_grad():
        a.model.model.layers[0].self_attn.q_proj.weight.add_(0.1)
    with pytest.raises(ValueError, match="stale"):
        a.forward(torch.tensor([[3]]), state=state)


def test_rope_mutation_invalidates_execution(random_adapter):
    a = random_adapter
    state = a.forward(torch.tensor([[1, 2]])).state
    a.model.config.rope_theta += 1
    with pytest.raises(ValueError, match="stale"):
        a.forward(torch.tensor([[3]]), state=state)


def test_revisions_must_be_immutable():
    with pytest.raises(ValueError, match="revision"):
        ExecutionIdentity("model", "main", "cfg", "0" * 40, "tpl", "rope", "float32", "scope")


@pytest.mark.pretrained
def test_pinned_pretrained_equivalence():
    if os.environ.get("HTIS_RUN_PRETRAINED") != "1":
        pytest.skip("Set HTIS_RUN_PRETRAINED=1 and download the pinned checkpoint; random unit tests are not model validation")
    from htis_context.checkpoint import load
    from htis_context.equivalence import check_equivalence
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/smollm2-135m.json").read_text())
    directory = Path(os.environ.get("HTIS_MODEL_DIR", str(root.parent / ".model-cache/smollm2-135m")))
    if not (directory / "htis-checkpoint.json").is_file():
        pytest.skip(f"Pinned pretrained checkpoint missing at {directory}")
    model, tokenizer, adapter, _ = load(config, directory)
    report = check_equivalence(model, tokenizer, adapter, config)
    assert report["passed"], [x for x in report["checks"] if not x["passed"]]
