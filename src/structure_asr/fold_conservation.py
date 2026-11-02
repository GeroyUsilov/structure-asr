"""Structural selection for forward simulation: ESM2 proposes, 3Di decides.

A new, opt-in generator mode. The legacy generator (`exact_mlm_gibbs`) accepts
every masked-site draw, so nothing in it preserves the root's fold; past μ≥2 the
simulated tips encode to mostly coil (see
`experiments/2026-11-02_divergence`). Here a block of accepted draws is kept
only if the candidate's predicted 3Di still resembles a reference structure.

This changes the generative model, not just its implementation. Cells simulated
this way are a separate dataset from every historical μ/τ run: μ stops being a
pure attempt-rate multiplier, because rejected blocks consume attempts without
realizing substitutions. `FoldSettings` enters the cell fingerprint, so
checkpoints never mix across modes.

Using a structure-conserving generator to test a structure-aware reconstruction
method is circular if the conservation assumption is also what the method
exploits. Report the realized structural divergence of any such dataset
alongside its reconstruction accuracy, and never present these cells as
evidence that the fold is conserved in real protein families.

Cost: one extra 3Di prediction per block per lane, batched across lanes in the
same step as the ESM2 pass, so `block` is the cost lever: attempts/`block`
predictions per cell, against one per attempt if checked every time.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

CRITERIA = ("threshold", "metropolis")
REFERENCES = ("root", "parent")
PREDICTORS = ("greedy",)


@dataclass(frozen=True)
class FoldSettings:
    """How structural selection accepts or reverts a block of draws.

    ``reference``: ``root`` compares every candidate with the root's 3Di, which
    bounds total structural divergence across the whole tree; ``parent``
    compares with the 3Di of the edge's starting sequence, which bounds the
    change per edge and lets structure drift as depth accumulates.

    ``criterion``: ``threshold`` keeps a block when 3Di identity to the
    reference is at least ``min_identity``; ``metropolis`` accepts with
    probability min(1, exp(-``beta`` × (new 3Di distance − old 3Di distance))),
    a soft selection that needs one extra RNG draw per block.

    ``predictor``: only ``greedy``, the autoregressive ProstT5 translation that
    also produces the 3Di tips and truth, so selection and measurement use one
    predictor. ProstT5's encoder-plus-CNN head would be far cheaper and is the
    next cost lever if ``block`` is not enough, but it is a different predictor
    and is not implemented here.
    """

    reference: str = "root"
    criterion: str = "threshold"
    min_identity: float = 0.65
    beta: float = 20.0
    block: int = 10
    predictor: str = "greedy"

    def __post_init__(self) -> None:
        if self.reference not in REFERENCES:
            raise ValueError(f"[fold].reference must be one of {list(REFERENCES)}")
        if self.criterion not in CRITERIA:
            raise ValueError(f"[fold].criterion must be one of {list(CRITERIA)}")
        if self.predictor not in PREDICTORS:
            raise ValueError(f"[fold].predictor must be one of {list(PREDICTORS)}")
        if not 0 <= self.min_identity <= 1:
            raise ValueError("[fold].min_identity must be in [0, 1]")
        if self.beta < 0:
            raise ValueError("[fold].beta must be nonnegative")
        if self.block < 1:
            raise ValueError("[fold].block must be positive")

    @classmethod
    def from_toml(cls, section: dict) -> "FoldSettings":
        section = dict(section)
        unknown = set(section) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown [fold] keys: {sorted(unknown)}")
        for key in ("min_identity", "beta"):
            if key in section:
                section[key] = float(section[key])
        if "block" in section:
            section["block"] = int(section["block"])
        return cls(**section)

    def as_dict(self) -> dict:
        """Fingerprint and manifest payload: every field that changes output."""
        return asdict(self)


def three_di_identity(candidate: str, reference: str) -> float:
    """Fraction of positions whose 3Di state matches, over the shorter length."""
    if not candidate or not reference:
        raise ValueError("3Di identity needs two non-empty strings")
    if len(candidate) != len(reference):
        raise ValueError(
            f"3Di lengths differ ({len(candidate)} vs {len(reference)}); "
            "the predictor must return one state per residue"
        )
    matches = sum(one == two for one, two in zip(candidate, reference))
    return matches / len(reference)


class FoldFilter:
    """Batched structural acceptance shared by every lane of a simulation.

    ``encode`` maps a list of AA sequences to their predicted 3Di strings in one
    call. Predictions are cached by sequence, so a reference is encoded once and
    repeated candidates cost nothing.
    """

    def __init__(self, settings: FoldSettings, encode: Callable[[list[str]], Sequence[str]]):
        self.settings = settings
        self._encode = encode
        self._cache: dict[str, str] = {}
        self.predictions = 0

    def prepare(self, sequences: Iterable[str]) -> None:
        """Encode whatever is not cached yet, in one batched call."""
        wanted = list(dict.fromkeys(sequence for sequence in sequences if sequence not in self._cache))
        if not wanted:
            return
        predicted = list(self._encode(wanted))
        if len(predicted) != len(wanted):
            raise RuntimeError("fold encoder returned the wrong number of 3Di strings")
        self.predictions += len(wanted)
        for sequence, three_di in zip(wanted, predicted):
            self._cache[sequence] = str(three_di)

    def three_di(self, sequence: str) -> str:
        try:
            return self._cache[sequence]
        except KeyError as error:  # pragma: no cover - guards a caller mistake
            raise RuntimeError("fold filter was asked about an unencoded sequence") from error

    def identity(self, sequence: str, reference: str) -> float:
        return three_di_identity(self.three_di(sequence), self.three_di(reference))

    def accepts(
        self,
        candidate: str,
        block_start: str,
        reference: str,
        rng: np.random.Generator,
    ) -> tuple[bool, float]:
        """Decide one block. Returns (accepted, candidate identity to reference).

        ``metropolis`` draws from ``rng``; ``threshold`` does not, so a
        threshold run's RNG stream stays aligned with the legacy generator's.
        """
        identity = self.identity(candidate, reference)
        if self.settings.criterion == "threshold":
            return identity >= self.settings.min_identity, identity
        previous = self.identity(block_start, reference)
        # Distance = 1 - identity, so improving identity always accepts.
        log_acceptance = min(0.0, -self.settings.beta * (previous - identity))
        return bool(math.log(rng.random()) < log_acceptance), identity
