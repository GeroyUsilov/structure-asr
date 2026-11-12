"""Compatibility alias; maintained implementation: ``structure_asr.simulation.scoring``.

Importing this module returns ``structure_asr.simulation.scoring`` itself, so saved configs, factory
strings and ``python -m forward_asr_benchmark.scoring`` keep working.
"""
import sys

from structure_asr.simulation import scoring as _impl

sys.modules[__name__] = _impl
