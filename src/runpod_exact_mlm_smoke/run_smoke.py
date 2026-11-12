"""Compatibility alias; maintained implementation: ``structure_asr.smoke.run_smoke``.

Importing this module returns ``structure_asr.smoke.run_smoke`` itself, so saved configs, factory
strings and ``python -m runpod_exact_mlm_smoke.run_smoke`` keep working.
"""
import sys

from structure_asr.smoke import run_smoke as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
