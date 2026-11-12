"""Compatibility package; maintained implementation: ``structure_asr.simulation``.

The forward-evolution engine now lives inside ``structure_asr``. This package
keeps ``import forward_asr_benchmark`` and its submodules working for saved
configs and older scripts.
"""
from structure_asr.simulation import *  # noqa: F401,F403
from structure_asr.simulation import __all__  # noqa: F401
