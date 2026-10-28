"""Bridge a user-selected OFS profile provider into the benchmark scorer."""

from __future__ import annotations

import importlib
from typing import Mapping

from forward_asr_benchmark import PseudoPerplexityScorer


def _load(specification: str):
    module_name, object_name = specification.split(":", 1)
    return getattr(importlib.import_module(module_name), object_name)


def make_ofs_scorer(parameters: Mapping):
    specification = str(parameters["provider_factory"])
    provider = _load(specification)(dict(parameters))
    if callable(provider):
        predictor = provider
    elif hasattr(provider, "predict_profile"):
        predictor = provider.predict_profile
    else:
        raise TypeError(f"{specification} must return a callable or object with predict_profile")
    return PseudoPerplexityScorer(
        profile_predictor=predictor,
        alphabet=tuple(str(parameters.get("alphabet", "ACDEFGHIKLMNPQRSTVWY"))),
        name="ofs",
    )

