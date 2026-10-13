from htis.experiment import run_suite, run_trial
from htis.synthetic import make_stream
import pytest


def test_reproducible_stream_and_no_future_targets():
    x = make_stream(60, 16, 123, "mixed")
    y = make_stream(60, 16, 123, "mixed")
    assert (x.keys == y.keys).all()
    assert (x.targets == y.targets).all()
    assert all(0 <= target <= t for t, target in enumerate(x.targets))


def test_trial_saves_theoretical_memory_and_reports_finite_error():
    trial = run_trial(scenario="mixed", seed=7, steps=48, head_dim=16,
                      budget=8, sinks=2, recent=2)
    assert set(trial) == {"recent", "sink_recent", "heavy_hitter", "hybrid"}
    for scores in trial.values():
        assert scores["tokens_retained"] == 8
        assert scores["final_kv_bytes_fp16"] == 2 * 8 * 16 * 2
        assert 0 < scores["attention_pair_ratio"] < 1
        assert 0 <= scores["target_recall"] <= 1
        assert 0 <= scores["normalized_output_mse"]
        assert 0 < scores["final_kv_reduction_fraction"] < 1


def test_suite_fixed_seed_is_deterministic():
    kw = dict(scenarios=("persistent",), seeds=(3, 4),
              steps=40, head_dim=8, budget=8, sinks=1, recent=2)
    assert run_suite(**kw) == run_suite(**kw)


def test_invalid_scenario_or_budget():
    with pytest.raises(ValueError):
        make_stream(scenario="not-real")
    with pytest.raises(ValueError):
        run_trial(scenario="mixed", seed=7, steps=10, budget=20)


def test_counterfactual_effect_equals_full_recomputation():
    import numpy as np
    from htis.cache import attention
    from htis.experiment import counterfactual_effects
    rng = np.random.default_rng(10)
    keys = rng.normal(size=(8, 4))
    values = rng.normal(size=(8, 4))
    out, weights = attention(rng.normal(size=4), keys, values)
    analytical = counterfactual_effects(weights, values, out)
    for i in range(8):
        kept = [j for j in range(8) if j != i]
        # Direct reconstruction from softmax weights is independent of logits.
        remaining = weights[kept] / weights[kept].sum()
        direct = remaining @ values[kept]
        assert np.isclose(analytical[i], np.linalg.norm(out - direct), atol=1e-10)


def test_ablation_rejects_overlapping_seeds_and_runs():
    from htis.ablation import ablate
    with pytest.raises(ValueError):
        ablate(validation_seeds=(1, 2), test_seeds=(2, 3))
    res = ablate(validation_seeds=(1,), test_seeds=(111,), weights=(0., 1.),
                 steps=40, head_dim=8, budget=8, sinks=1, recent=2)
    assert res["selected_weight"] in (0., 1.)
    assert len(res["held_out"]) == 4
