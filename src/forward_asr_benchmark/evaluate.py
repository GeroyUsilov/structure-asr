"""Ground-truth sequence evaluation for reconstructed internal nodes."""

from __future__ import annotations

import csv
from pathlib import Path

from .io import read_fasta
from .scoring import SequenceScorer


def compare_reconstructions(
    truth_fasta: str | Path,
    reconstruction_fasta: str | Path,
) -> list[dict[str, int | float | str]]:
    truth = read_fasta(truth_fasta)
    reconstructed = read_fasta(reconstruction_fasta)
    rows: list[dict[str, int | float | str]] = []
    for node, sequence in reconstructed.items():
        if node not in truth:
            rows.append({"node": node, "status": "not_in_truth", "length": len(sequence)})
            continue
        target = truth[node]
        if len(sequence) != len(target):
            rows.append(
                {
                    "node": node,
                    "status": "length_mismatch",
                    "length": len(sequence),
                    "truth_length": len(target),
                }
            )
            continue
        distance = sum(a != b for a, b in zip(sequence, target))
        rows.append(
            {
                "node": node,
                "status": "ok",
                "length": len(sequence),
                "hamming_distance": distance,
                "normalized_hamming": distance / len(sequence) if sequence else 0.0,
                "sequence_identity": 1.0 - distance / len(sequence) if sequence else 1.0,
            }
        )
    return rows


def write_comparison(path: str | Path, rows: list[dict]) -> None:
    keys = ["node", "status", "length", "truth_length", "hamming_distance", "normalized_hamming", "sequence_identity"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def score_sequences(fasta: str | Path, scorer: SequenceScorer) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for identifier, sequence in read_fasta(fasta).items():
        result = scorer.score(sequence)
        rows.append({"sequence_id": identifier, "energy": result.energy, **result.metrics})
    return rows


def write_scores(path: str | Path, rows: list[dict[str, float | str]]) -> None:
    metric_names = sorted({key for row in rows for key in row} - {"sequence_id", "energy"})
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_id", "energy", *metric_names], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
