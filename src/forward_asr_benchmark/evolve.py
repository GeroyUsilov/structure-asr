"""Compatibility alias; maintained implementation: ``structure_asr.simulation.evolve``.

Importing this module returns ``structure_asr.simulation.evolve`` itself, so saved configs, factory
strings and ``python -m forward_asr_benchmark.evolve`` keep working.
"""
import sys

from structure_asr.simulation import evolve as _impl

sys.modules[__name__] = _impl
