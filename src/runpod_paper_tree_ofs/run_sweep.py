"""Compatibility alias; maintained implementation: ``structure_asr.ofs.run_sweep``.

Importing this module returns ``structure_asr.ofs.run_sweep`` itself, so saved configs, factory
strings and ``python -m runpod_paper_tree_ofs.run_sweep`` keep working.
"""
import sys

from structure_asr.ofs import run_sweep as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
