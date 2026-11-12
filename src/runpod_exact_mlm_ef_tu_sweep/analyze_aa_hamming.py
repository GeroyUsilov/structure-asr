"""Compatibility alias; maintained implementation: ``structure_asr.analysis.hamming_aa``.

Importing this module returns ``structure_asr.analysis.hamming_aa`` itself, so saved configs, factory
strings and ``python -m runpod_exact_mlm_ef_tu_sweep.analyze_aa_hamming`` keep working.
"""
import sys

from structure_asr.analysis import hamming_aa as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
