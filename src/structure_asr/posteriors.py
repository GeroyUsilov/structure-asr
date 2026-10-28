"""Shared IQ-TREE marginal loading and independent-site sequence scoring.

Sampling floors every entry and renormalizes each row. Historical evaluation
uses a different, explicitly documented censoring convention; do not conflate them.
"""
from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CANONICAL_STATES = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class NodePosterior:
    alphabet: tuple[str, ...]
    probabilities: np.ndarray
    log_probabilities: np.ndarray


def node_sort_key(name: str) -> tuple[str, int]:
    prefix = name.rstrip("0123456789")
    suffix = name[len(prefix) :]
    return prefix, int(suffix) if suffix else -1


def load_state(path: Path, probability_floor: float) -> OrderedDict[str, NodePosterior]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not lines:
        raise ValueError(f"no data rows in {path}")
    reader = csv.DictReader(lines, delimiter="\t")
    if reader.fieldnames is None:
        raise ValueError(f"missing header in {path}")
    probability_columns = [name for name in reader.fieldnames if name.startswith("p_")]
    alphabet = tuple(name[2:].upper() for name in probability_columns)
    if len(alphabet) != 20 or set(alphabet) != CANONICAL_STATES:
        raise ValueError(f"unexpected state alphabet in {path}: {alphabet}")

    raw_by_node: OrderedDict[str, list[tuple[int, list[float]]]] = OrderedDict()
    for row in reader:
        node = str(row["Node"])
        site = int(row["Site"])
        values = [float(row[column]) for column in probability_columns]
        raw_by_node.setdefault(node, []).append((site, values))

    result: OrderedDict[str, NodePosterior] = OrderedDict()
    for node, rows in raw_by_node.items():
        rows.sort(key=lambda item: item[0])
        expected_sites = list(range(1, len(rows) + 1))
        observed_sites = [site for site, _values in rows]
        if observed_sites != expected_sites:
            raise ValueError(f"non-contiguous sites for {node} in {path}")
        matrix = np.asarray([values for _site, values in rows], dtype=np.float64)
        if np.any(matrix < 0) or np.any(~np.isfinite(matrix)):
            raise ValueError(f"invalid posterior probability in {path}, {node}")
        matrix = np.maximum(matrix, probability_floor)
        matrix /= matrix.sum(axis=1, keepdims=True)
        result[node] = NodePosterior(
            alphabet=alphabet,
            probabilities=matrix,
            log_probabilities=np.log(matrix),
        )
    return result


def sequence_log_probability(posterior: NodePosterior, sequence: str) -> float:
    if len(sequence) != posterior.probabilities.shape[0]:
        raise ValueError("sequence and posterior lengths differ")
    index = {state: i for i, state in enumerate(posterior.alphabet)}
    try:
        columns = np.fromiter((index[state.upper()] for state in sequence), dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"sequence contains an invalid state: {error.args[0]}") from error
    return float(posterior.log_probabilities[np.arange(len(sequence)), columns].sum())


def aa_map_sequence(posterior: NodePosterior) -> str:
    indices = posterior.probabilities.argmax(axis=1)
    return "".join(posterior.alphabet[index] for index in indices)


def sample_sequence(posterior: NodePosterior, rng: np.random.Generator) -> str:
    states = [
        posterior.alphabet[int(rng.choice(len(posterior.alphabet), p=row))]
        for row in posterior.probabilities
    ]
    return "".join(states)
