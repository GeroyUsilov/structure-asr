"""TOML configuration and scorer plug-in loading."""

from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path
from typing import Any, Mapping

from .dca import DcaModel
from .scoring import CompositeScorer, ConstantScorer, DcaScorer, SequenceScorer


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        return tomllib.load(handle), config_path.parent


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_object(specification: str):
    if ":" not in specification:
        raise ValueError("Python factory must use 'module:object' notation")
    module_name, object_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def build_composite_scorer(
    scorer_configs: list[Mapping[str, Any]],
    dca_model: DcaModel | None,
    base: Path,
) -> CompositeScorer:
    configured: list[tuple[float, SequenceScorer]] = []
    for item in scorer_configs:
        kind = str(item["kind"])
        weight = float(item.get("weight", 1.0))
        if kind == "dca":
            if dca_model is None:
                raise ValueError("a dca scorer requires [dca].model_npz")
            scorer = DcaScorer(dca_model)
        elif kind == "constant":
            scorer = ConstantScorer(energy=float(item.get("energy", 0.0)))
        elif kind == "python":
            factory = load_object(str(item["factory"]))
            parameters = dict(item.get("parameters", {}))
            # Resolve only explicitly marked paths, avoiding guesses about strings.
            for key in list(parameters):
                if key.endswith("_path"):
                    parameters[key] = str(resolve_path(base, str(parameters[key])))
            scorer = factory(parameters)
            if not hasattr(scorer, "score") or not hasattr(scorer, "name"):
                raise TypeError(f"{item['factory']} did not return a SequenceScorer")
        else:
            raise ValueError(f"unknown scorer kind: {kind!r}")
        configured.append((weight, scorer))
    return CompositeScorer(configured)


def public_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-safe config copy with obvious credential fields removed."""
    payload = json.loads(json.dumps(config))
    secret_fragments = ("token", "secret", "password", "api_key")

    def scrub(value):
        if isinstance(value, dict):
            return {
                key: ("[REDACTED]" if any(fragment in key.lower() for fragment in secret_fragments) else scrub(child))
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [scrub(child) for child in value]
        return value

    return scrub(payload)
