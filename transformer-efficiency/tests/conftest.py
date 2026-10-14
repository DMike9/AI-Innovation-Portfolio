"""Optional dependency guard for the frozen Phase 5B test module.

The numerical experiment/test source freeze remains unchanged. This collection
guard keeps the existing dependency-light CI job usable without Torch/context
extras, and has no effect on inference or on installed-context assertions.
"""
import pytest


class OptionalPhase5BModule(pytest.Module):
    def _getobj(self):
        pytest.importorskip('torch', reason='Phase 5B reference tests require optional Torch/context dependencies')
        pytest.importorskip('transformers', reason='Phase 5B reference tests require optional Transformers/context dependencies')
        return super()._getobj()


def pytest_pycollect_makemodule(module_path, parent):
    if module_path.name == 'test_phase5b.py':
        return OptionalPhase5BModule.from_parent(parent, path=module_path)
    return None
