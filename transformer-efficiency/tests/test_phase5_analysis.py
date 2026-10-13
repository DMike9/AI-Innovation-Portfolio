"""Offline statistical pairing must not confound an ablation with checkpoint seed."""
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/analyze_phase5.py'
spec=importlib.util.spec_from_file_location('phase5_analysis_script',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def row(arm,seed,value,case='a'):
    return dict(arm=arm,seed=seed,mean_kl=value,case=case,group='doc/'+case,kind='official-short-qa')


def test_single_seed_ablation_pairs_only_same_trained_seed():
    result=module.matched_paired([row('htis',17,1),row('htis',29,100),row('ablation',17,2)],'ablation','htis','mean_kl')
    assert result['mean_delta']==1 and result['matched_case_seed_pairs']==[('a',17)]


def test_deterministic_reference_is_held_fixed_across_trained_seeds():
    result=module.matched_paired([row('recent',17,3),row('htis',17,1),row('htis',29,5)],'htis','recent','mean_kl')
    assert result['mean_delta']==0 and len(result['matched_case_seed_pairs'])==2


def test_stochastic_comparison_requires_shared_seed():
    with pytest.raises(ValueError,match='matching seed'):
        module.matched_paired([row('random',29,1),row('htis',17,2)],'htis','random','mean_kl')
