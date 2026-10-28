#!/usr/bin/env python3
"""Compute and plot within-dataset pairwise Hamming distances for AA FASTAs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from forward_asr_benchmark.io import read_fasta
from runpod_exact_mlm_ef_tu_sweep.analyze_3di_hamming import (
    hamming_matrices,
    write_matrix,
)


AA_ALPHABET_WITH_GAP = frozenset("ACDEFGHIKLMNPQRSTVWY-")


def load_aligned_aa(path: Path) -> tuple[list[str], np.ndarray]:
    records = read_fasta(path)
    identifiers = list(records)
    sequences = [records[identifier].upper() for identifier in identifiers]
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError(
            f"Hamming distance requires equal lengths within {path}: {sorted(lengths)}"
        )
    invalid = set().union(*(set(sequence) for sequence in sequences)) - AA_ALPHABET_WITH_GAP
    if invalid:
        raise ValueError(f"invalid amino-acid symbols in {path}: {sorted(invalid)}")
    return identifiers, np.asarray([list(sequence) for sequence in sequences], dtype="U1")


def analyze_dataset(name: str, fasta: Path, output: Path, bins: int) -> dict:
    identifiers, encoded = load_aligned_aa(fasta)
    mismatches, normalized = hamming_matrices(encoded)
    triangle = np.triu_indices(len(identifiers), k=1)
    pairwise_counts = mismatches[triangle]
    pairwise_normalized = normalized[triangle]
    output.mkdir(parents=True, exist_ok=True)
    write_matrix(output / "hamming_counts_matrix.tsv", identifiers, mismatches)
    write_matrix(output / "hamming_normalized_matrix.tsv", identifiers, normalized)
    with (output / "pairwise_hamming.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["sequence_a", "sequence_b", "mismatches", "normalized_hamming"]
        )
        for row, column, count, distance in zip(
            triangle[0], triangle[1], pairwise_counts, pairwise_normalized
        ):
            writer.writerow(
                [identifiers[row], identifiers[column], int(count), f"{distance:.9f}"]
            )

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.hist(
        pairwise_normalized,
        bins=np.linspace(0, 1, bins + 1),
        color="#35618f",
        alpha=0.72,
    )
    axis.set(
        xlabel="Normalized amino-acid Hamming distance",
        ylabel="Sequence pairs",
        title=name,
    )
    axis.set_xlim(0, 1)
    figure.tight_layout()
    figure.savefig(output / "hamming_histogram.png", dpi=200)
    figure.savefig(output / "hamming_histogram.pdf")
    plt.close(figure)

    size = max(7.0, min(14.0, 5.0 + len(identifiers) / 40.0))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(
        normalized, vmin=0, vmax=1, cmap="viridis", interpolation="nearest"
    )
    axis.set_title(f"{name}: normalized amino-acid Hamming distance")
    if len(identifiers) <= 60:
        axis.set_xticks(range(len(identifiers)), identifiers, rotation=90, fontsize=5)
        axis.set_yticks(range(len(identifiers)), identifiers, fontsize=5)
    else:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output / "hamming_matrix_heatmap.png", dpi=200)
    figure.savefig(output / "hamming_matrix_heatmap.pdf")
    plt.close(figure)

    summary = {
        "dataset": name,
        "source_fasta": str(fasta.resolve()),
        "sequence_count": len(identifiers),
        "aligned_length": int(encoded.shape[1]),
        "gap_count": int(np.count_nonzero(encoded == "-")),
        "pair_count": int(len(pairwise_normalized)),
        "normalized_hamming": {
            "mean": float(np.mean(pairwise_normalized)),
            "median": float(np.median(pairwise_normalized)),
            "min": float(np.min(pairwise_normalized)),
            "max": float(np.max(pairwise_normalized)),
        },
        "gap_policy": "gaps retained as an aligned state",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"summary": summary, "distances": pairwise_normalized}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-dir", type=Path, required=True)
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--rates", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )
    parser.add_argument("--bins", type=int, default=40)
    args = parser.parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")

    datasets = []
    for rate in args.rates:
        label = format(rate, "g").replace("-", "m").replace(".", "p")
        datasets.append(
            (
                f"EF-Tu mu={rate:g}",
                args.synthetic_dir / f"mu_{label}" / "leaves_aa.fasta",
            )
        )
    datasets.append(("QE1.9d AA", args.reference_fasta))
    missing = [str(path) for _name, path in datasets if not path.exists()]
    if missing:
        raise FileNotFoundError("missing AA FASTA(s): " + ", ".join(missing))

    args.outdir.mkdir(parents=True, exist_ok=True)
    analyses = []
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    edges = np.linspace(0, 1, args.bins + 1)
    simulation_colors = plt.colormaps["viridis"](
        np.linspace(0.06, 0.92, len(args.rates))
    )
    dataset_colors = [*simulation_colors, plt.colormaps["tab10"](3)]
    for dataset_index, (name, fasta) in enumerate(datasets):
        slug = name.lower().replace("=", "_").replace(".", "p").replace(" ", "_")
        result = analyze_dataset(name, fasta, args.outdir / slug, args.bins)
        analyses.append(result["summary"])
        axis.hist(
            result["distances"],
            bins=edges,
            histtype="stepfilled",
            color=dataset_colors[dataset_index],
            edgecolor=dataset_colors[dataset_index],
            linewidth=1.0,
            alpha=0.32,
            density=True,
            label=name,
        )
    axis.set(
        xlabel="Normalized amino-acid Hamming distance",
        ylabel="Density",
        title="Pairwise amino-acid distance distributions",
    )
    axis.set_xlim(0, 1)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.outdir / "all_datasets_aa_hamming_histograms.png", dpi=200)
    figure.savefig(args.outdir / "all_datasets_aa_hamming_histograms.pdf")
    plt.close(figure)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "distance": "sitewise Hamming distance within each aligned AA FASTA",
        "normalization": "mismatches divided by sequence length",
        "gap_policy": "gaps retained as an aligned state",
        "datasets": analyses,
    }
    (args.outdir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote AA Hamming analyses to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
