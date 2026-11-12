"""Compatibility alias; maintained implementation: ``structure_asr.analysis.encode_truth_nodes``.

Importing this module returns ``structure_asr.analysis.encode_truth_nodes`` itself, so saved configs, factory
strings and ``python -m runpod_exact_mlm_ef_tu_sweep.convert_ground_truth_nodes_to_3di`` keep working.
"""
import sys

from structure_asr.analysis import encode_truth_nodes as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
