"""Compatibility alias; maintained implementation: ``structure_asr.simulation.io``.

Importing this module returns ``structure_asr.simulation.io`` itself, so saved configs, factory
strings and ``python -m forward_asr_benchmark.io`` keep working.
"""
import sys

from structure_asr.simulation import io as _impl

sys.modules[__name__] = _impl
