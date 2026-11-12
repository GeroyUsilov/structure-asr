"""Compatibility alias; maintained implementation: ``structure_asr.simulation.dca``.

Importing this module returns ``structure_asr.simulation.dca`` itself, so saved configs, factory
strings and ``python -m forward_asr_benchmark.dca`` keep working.
"""
import sys

from structure_asr.simulation import dca as _impl

sys.modules[__name__] = _impl
