"""Compatibility alias; maintained implementation: ``structure_asr.simulation.cli``.

Importing this module returns ``structure_asr.simulation.cli`` itself, so saved configs, factory
strings and ``python -m forward_asr_benchmark.cli`` keep working.
"""
import sys

from structure_asr.simulation import cli as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
