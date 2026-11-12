"""Compatibility alias; maintained implementation: ``structure_asr.analysis.temperature_sweep``.

Importing this module returns ``structure_asr.analysis.temperature_sweep`` itself, so saved configs, factory
strings and ``python -m runpod_exact_mlm_ef_tu_sweep.analyze_temperature_sweep`` keep working.
"""
import sys

from structure_asr.analysis import temperature_sweep as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
