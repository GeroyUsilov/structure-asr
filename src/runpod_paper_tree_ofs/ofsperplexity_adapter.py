"""Compatibility alias; maintained implementation: ``structure_asr.ofs.ofsperplexity_adapter``.

Importing this module returns ``structure_asr.ofs.ofsperplexity_adapter`` itself, so saved configs, factory
strings and ``python -m runpod_paper_tree_ofs.ofsperplexity_adapter`` keep working.
"""
import sys

from structure_asr.ofs import ofsperplexity_adapter as _impl

sys.modules[__name__] = _impl
