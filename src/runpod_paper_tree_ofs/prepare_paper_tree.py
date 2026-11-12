"""Compatibility alias; maintained implementation: ``structure_asr.ofs.prepare_paper_tree``.

Importing this module returns ``structure_asr.ofs.prepare_paper_tree`` itself, so saved configs, factory
strings and ``python -m runpod_paper_tree_ofs.prepare_paper_tree`` keep working.
"""
import sys

from structure_asr.ofs import prepare_paper_tree as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
