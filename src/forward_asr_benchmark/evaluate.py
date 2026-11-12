"""Compatibility alias; maintained implementation: ``structure_asr.simulation.evaluate``.

Importing this module returns ``structure_asr.simulation.evaluate`` itself, so saved configs, factory
strings and ``python -m forward_asr_benchmark.evaluate`` keep working.
"""
import sys

from structure_asr.simulation import evaluate as _impl

sys.modules[__name__] = _impl
