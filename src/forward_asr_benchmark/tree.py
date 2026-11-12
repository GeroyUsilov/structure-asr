"""Compatibility alias; maintained implementation: ``structure_asr.simulation.tree``.

Importing this module returns ``structure_asr.simulation.tree`` itself, so saved configs, factory
strings and ``python -m forward_asr_benchmark.tree`` keep working.
"""
import sys

from structure_asr.simulation import tree as _impl

sys.modules[__name__] = _impl
