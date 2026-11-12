"""Compatibility alias; maintained implementation: ``structure_asr.simulation.config``.

Importing this module returns ``structure_asr.simulation.config`` itself, so saved configs, factory
strings and ``python -m forward_asr_benchmark.config`` keep working.
"""
import sys

from structure_asr.simulation import config as _impl

sys.modules[__name__] = _impl
