"""Small, explicit Potts/DCA model used by the simulator.

The portable NPZ representation has four arrays:

``alphabet`` (q,), ``fields`` (L, q), ``couplings`` (L, L, q, q), and an
optional scalar ``offset``.  Energies use the convention

    E(s) = offset - sum_i h_i(s_i) - sum_{i<j} J_ij(s_i, s_j).

Converters from adabmDCA/plmDCA should normalize their parameters into this
representation once, keeping the simulation core independent of a DCA package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum()))


@dataclass(frozen=True)
class DcaModel:
    alphabet: tuple[str, ...]
    fields: np.ndarray
    couplings: np.ndarray
    offset: float = 0.0

    def __post_init__(self) -> None:
        q = len(self.alphabet)
        if len(set(self.alphabet)) != q or any(len(token) != 1 for token in self.alphabet):
            raise ValueError("alphabet must contain unique one-character tokens")
        if self.fields.ndim != 2 or self.fields.shape[1] != q:
            raise ValueError("fields must have shape (L, q)")
        expected = (self.length, self.length, q, q)
        if self.couplings.shape != expected:
            raise ValueError(f"couplings must have shape {expected}")
        if not np.all(np.isfinite(self.fields)) or not np.all(np.isfinite(self.couplings)):
            raise ValueError("DCA parameters must be finite")

    @property
    def length(self) -> int:
        return int(self.fields.shape[0])

    @property
    def q(self) -> int:
        return len(self.alphabet)

    @property
    def index(self) -> dict[str, int]:
        return {token: i for i, token in enumerate(self.alphabet)}

    def encode(self, sequence: str) -> np.ndarray:
        if len(sequence) != self.length:
            raise ValueError(f"expected sequence length {self.length}, got {len(sequence)}")
        index = self.index
        try:
            return np.asarray([index[token] for token in sequence], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"sequence contains token outside DCA alphabet: {exc.args[0]!r}") from exc

    def energy(self, sequence: str) -> float:
        encoded = self.encode(sequence)
        value = float(self.offset - self.fields[np.arange(self.length), encoded].sum())
        for i in range(self.length):
            for j in range(i + 1, self.length):
                value -= float(self.couplings[i, j, encoded[i], encoded[j]])
        return value

    def conditional_logits(self, sequence: str, site: int) -> np.ndarray:
        """Return log P(s_site=a | s_without_site), up to normalization."""
        encoded = self.encode(sequence)
        if not 0 <= site < self.length:
            raise IndexError(site)
        logits = self.fields[site].astype(float, copy=True)
        for j in range(self.length):
            if j == site:
                continue
            if site < j:
                logits += self.couplings[site, j, :, encoded[j]]
            else:
                # J is stored in upper-triangle orientation: J[j, site](s_j, a).
                logits += self.couplings[j, site, encoded[j], :]
        return logits

    def conditional_log_probabilities(self, sequence: str, site: int) -> np.ndarray:
        logits = self.conditional_logits(sequence, site)
        return logits - _logsumexp(logits)

    def sample_conditional(self, sequence: str, site: int, rng: np.random.Generator) -> tuple[str, float]:
        log_probabilities = self.conditional_log_probabilities(sequence, site)
        probabilities = np.exp(log_probabilities)
        state = int(rng.choice(self.q, p=probabilities))
        return self.alphabet[state], float(log_probabilities[state])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            alphabet=np.asarray(self.alphabet),
            fields=self.fields,
            couplings=self.couplings,
            offset=np.asarray(self.offset),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DcaModel":
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                alphabet=tuple(str(x) for x in payload["alphabet"].tolist()),
                fields=np.asarray(payload["fields"], dtype=float),
                couplings=np.asarray(payload["couplings"], dtype=float),
                offset=float(payload["offset"]) if "offset" in payload else 0.0,
            )

    @classmethod
    def independent(cls, alphabet: Sequence[str], fields: np.ndarray) -> "DcaModel":
        fields = np.asarray(fields, dtype=float)
        length, q = fields.shape
        if q != len(alphabet):
            raise ValueError("field alphabet dimension does not match alphabet")
        return cls(
            alphabet=tuple(alphabet),
            fields=fields,
            couplings=np.zeros((length, length, q, q), dtype=float),
        )

