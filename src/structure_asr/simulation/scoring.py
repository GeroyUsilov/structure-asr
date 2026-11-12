"""Composable sequence energies for evolutionary selection.

Every scorer returns an energy, so lower is better.  Metrics retain the native
quantities (pLDDT, pTM, pseudo-perplexity, and so on) for auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .dca import DcaModel


@dataclass(frozen=True)
class ScoreResult:
    energy: float
    metrics: Mapping[str, float] = field(default_factory=dict)


class SequenceScorer(Protocol):
    name: str

    def score(self, sequence: str) -> ScoreResult: ...


@dataclass
class ConstantScorer:
    """Zero-cost bookkeeping scorer for always-accept generator kernels."""

    energy: float = 0.0
    name: str = "constant"

    def score(self, sequence: str) -> ScoreResult:
        return ScoreResult(energy=float(self.energy), metrics={})


class CachedProfilePredictor:
    """Tiny LRU cache shared by an LM proposal and pseudoperplexity scorer.

    In a Markov chain the accepted proposal becomes the next current state, so
    caching two profiles normally reduces OFS inference to one new profile per
    attempted update even when the reverse proposal probability is evaluated.
    """

    def __init__(self, predictor: Callable[[str], np.ndarray], max_size: int = 2):
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.predictor = predictor
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __call__(self, sequence: str) -> np.ndarray:
        if sequence in self._cache:
            value = self._cache.pop(sequence)
            self._cache[sequence] = value
            return value
        value = np.asarray(self.predictor(sequence), dtype=float)
        self._cache[sequence] = value
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return value


@dataclass
class DcaScorer:
    model: DcaModel
    name: str = "dca"

    def score(self, sequence: str) -> ScoreResult:
        energy = self.model.energy(sequence)
        return ScoreResult(energy=energy, metrics={"dca_energy": energy})


@dataclass
class PseudoPerplexityScorer:
    """Score a sequence from a callable returning masked residue probabilities.

    ``profile_predictor(sequence)`` must return an (L, q) array whose columns
    follow ``alphabet``.  It can wrap exact ESM masking, the paper's OFS head,
    or any other encoder.  The energy is mean negative pseudo-log-likelihood;
    pseudo-perplexity is exp(energy).
    """

    profile_predictor: Callable[[str], np.ndarray]
    alphabet: Sequence[str] = tuple("ACDEFGHIKLMNPQRSTVWY")
    epsilon: float = 1e-12
    name: str = "lm"

    def score(self, sequence: str) -> ScoreResult:
        profile = np.asarray(self.profile_predictor(sequence), dtype=float)
        if profile.shape != (len(sequence), len(self.alphabet)):
            raise ValueError(
                f"LM profile shape {profile.shape} does not match "
                f"({len(sequence)}, {len(self.alphabet)})"
            )
        row_sums = profile.sum(axis=1, keepdims=True)
        if np.any(profile < 0) or np.any(row_sums <= 0):
            raise ValueError("LM profile must contain nonnegative, nonempty rows")
        profile = profile / row_sums
        index = {aa: i for i, aa in enumerate(self.alphabet)}
        try:
            observed = np.asarray([index[aa] for aa in sequence], dtype=int)
        except KeyError as exc:
            raise ValueError(f"sequence residue {exc.args[0]!r} is not in LM alphabet") from exc
        probabilities = np.maximum(profile[np.arange(len(sequence)), observed], self.epsilon)
        energy = float(-np.log(probabilities).mean())
        return ScoreResult(
            energy=energy,
            metrics={"pseudo_log_likelihood": -energy, "pseudo_perplexity": float(np.exp(energy))},
        )


@dataclass
class AlphaFoldUncertaintyScorer:
    """Turn AlphaFold/ColabFold output into a fold-preservation energy.

    ``predictor(sequence)`` returns native metrics.  Supported keys are
    ``mean_plddt`` (0..100), ``ptm`` (0..1), ``mean_pae`` (Angstrom), and
    ``reference_tm`` (0..1, comparing the predicted structure to the supplied
    ancestral fold).  For genuine fold preservation, reference_tm should be
    present; uncertainty alone is not proof of the same topology.
    """

    predictor: Callable[[str], Mapping[str, Any]]
    plddt_weight: float = 1.0
    ptm_weight: float = 1.0
    pae_weight: float = 0.25
    reference_tm_weight: float = 2.0
    max_pae: float = 31.75
    require_reference_tm: bool = True
    name: str = "alphafold"

    def score(self, sequence: str) -> ScoreResult:
        raw = self.predictor(sequence)
        metrics = {key: float(raw[key]) for key in ("mean_plddt", "ptm", "mean_pae", "reference_tm") if key in raw}
        if self.require_reference_tm and "reference_tm" not in metrics:
            raise ValueError("AlphaFold scorer requires reference_tm to enforce preservation of the ancestral fold")
        terms: list[float] = []
        if "mean_plddt" in metrics:
            terms.append(self.plddt_weight * (1.0 - np.clip(metrics["mean_plddt"] / 100.0, 0.0, 1.0)))
        if "ptm" in metrics:
            terms.append(self.ptm_weight * (1.0 - np.clip(metrics["ptm"], 0.0, 1.0)))
        if "mean_pae" in metrics:
            terms.append(self.pae_weight * np.clip(metrics["mean_pae"] / self.max_pae, 0.0, 1.0))
        if "reference_tm" in metrics:
            terms.append(self.reference_tm_weight * (1.0 - np.clip(metrics["reference_tm"], 0.0, 1.0)))
        if not terms:
            raise ValueError("AlphaFold predictor returned no supported metrics")
        return ScoreResult(energy=float(sum(terms)), metrics=metrics)


@dataclass
class CompositeScorer:
    scorers: Sequence[tuple[float, SequenceScorer]]
    name: str = "composite"

    def __post_init__(self) -> None:
        if not self.scorers or any(not np.isfinite(weight) or weight < 0 for weight, _ in self.scorers):
            raise ValueError("composite scorer needs nonnegative finite weights")

    def score(self, sequence: str) -> ScoreResult:
        energy = 0.0
        metrics: dict[str, float] = {}
        for weight, scorer in self.scorers:
            result = scorer.score(sequence)
            energy += weight * result.energy
            metrics[f"{scorer.name}.energy"] = float(result.energy)
            metrics.update({f"{scorer.name}.{key}": float(value) for key, value in result.metrics.items()})
        return ScoreResult(energy=float(energy), metrics=metrics)
