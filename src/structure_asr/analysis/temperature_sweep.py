#!/usr/bin/env python3
"""Prepare ASR FASTAs and AA/3Di pairwise-Hamming ECDFs for nine EF-Tu runs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from Bio import Phylo


ALPHABET_WITH_GAP = frozenset("ACDEFGHIKLMNPQRSTVWY-")
MUS = (1, 2, 3)
TAUS = (1.0, 1.5, 2.0)


def value_label(value: float) -> str:
    return format(value, "g").replace("-", "m").replace(".", "p")


def condition_slug(mu: int, tau: float) -> str:
    return f"mu{mu}_tau{value_label(tau)}"


def parse_fasta_text(text: str, source: str) -> OrderedDict[str, str]:
    records: OrderedDict[str, str] = OrderedDict()
    identifier: str | None = None
    chunks: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records[identifier] = "".join(chunks).upper()
            identifier = line[1:].split()[0]
            if not identifier:
                raise ValueError(f"empty FASTA identifier in {source}:{line_number}")
            if identifier in records:
                raise ValueError(f"duplicate FASTA identifier {identifier!r} in {source}")
            chunks = []
        else:
            if identifier is None:
                raise ValueError(f"sequence before FASTA header in {source}:{line_number}")
            chunks.append(line)
    if identifier is not None:
        if identifier in records:
            raise ValueError(f"duplicate FASTA identifier {identifier!r} in {source}")
        records[identifier] = "".join(chunks).upper()
    if not records:
        raise ValueError(f"no FASTA records in {source}")
    return records


def read_fasta(path: Path) -> OrderedDict[str, str]:
    return parse_fasta_text(path.read_text(encoding="utf-8"), str(path))


def write_fasta(path: Path, records: OrderedDict[str, str], width: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for identifier, sequence in records.items():
            handle.write(f">{identifier}\n")
            for start in range(0, len(sequence), width):
                handle.write(sequence[start : start + width] + "\n")


def tar_text(bundle: tarfile.TarFile, member: str) -> str:
    extracted = bundle.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(f"missing archive member: {member}")
    return extracted.read().decode("utf-8")


def validate_records(
    records: OrderedDict[str, str],
    *,
    source: str,
    expected_ids: set[str] | None = None,
    expected_count: int | None = None,
    expected_length: int | None = None,
) -> int:
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"{source} has {len(records)} records; expected {expected_count}")
    if expected_ids is not None and set(records) != expected_ids:
        missing = sorted(expected_ids - set(records))
        extra = sorted(set(records) - expected_ids)
        raise ValueError(f"{source} identifiers disagree with tree; missing={missing}, extra={extra}")
    lengths = {len(sequence) for sequence in records.values()}
    if len(lengths) != 1:
        raise ValueError(f"{source} is not aligned: lengths={sorted(lengths)}")
    length = next(iter(lengths))
    if expected_length is not None and length != expected_length:
        raise ValueError(f"{source} length is {length}; expected {expected_length}")
    invalid = set().union(*(set(sequence) for sequence in records.values())) - ALPHABET_WITH_GAP
    if invalid:
        raise ValueError(f"{source} contains invalid symbols: {sorted(invalid)}")
    return length


def pairwise_hamming(
    records: OrderedDict[str, str], output: Path
) -> tuple[np.ndarray, dict[str, float | int]]:
    identifiers = list(records)
    encoded = np.asarray([list(records[name]) for name in identifiers], dtype="U1")
    sequence_length = int(encoded.shape[1])
    distances: list[np.ndarray] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["sequence_a", "sequence_b", "mismatches", "aligned_length", "normalized_hamming"]
        )
        for index, identifier in enumerate(identifiers[:-1]):
            counts = np.count_nonzero(encoded[index + 1 :] != encoded[index], axis=1)
            normalized = counts.astype(float) / sequence_length
            distances.append(normalized)
            for other, count, distance in zip(identifiers[index + 1 :], counts, normalized):
                writer.writerow(
                    [identifier, other, int(count), sequence_length, f"{distance:.9f}"]
                )
    values = np.concatenate(distances)
    summary: dict[str, float | int] = {
        "sequence_count": len(identifiers),
        "aligned_length": sequence_length,
        "pair_count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }
    return values, summary


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    x = np.concatenate(([0.0], ordered, [1.0]))
    y = np.concatenate(([0.0], np.arange(1, len(ordered) + 1) / len(ordered), [1.0]))
    return x, y


def plot_ecdfs(
    distances: dict[str, dict[str, np.ndarray]], output: Path
) -> None:
    colors = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
    line_styles = {1.0: "-", 1.5: "--", 2.0: ":"}
    figure, axes = plt.subplots(1, 2, figsize=(15.2, 6.2), sharex=True, sharey=True)
    handles = []
    labels = []
    for axis, modality, title in zip(
        axes,
        ("aa", "3di"),
        ("Amino-acid distances", "Predicted 3Di distances"),
    ):
        for mu in MUS:
            for tau in TAUS:
                slug = condition_slug(mu, tau)
                x, y = ecdf(distances[modality][slug])
                (line,) = axis.step(
                    x,
                    y,
                    where="post",
                    color=colors[mu],
                    linestyle=line_styles[tau],
                    linewidth=1.9,
                    alpha=0.92,
                    label=rf"$\mu={mu}$, $\tau={tau:g}$",
                )
                if axis is axes[0]:
                    handles.append(line)
                    labels.append(rf"$\mu={mu}$, $\tau={tau:g}$")
        x, y = ecdf(distances[modality]["empirical"])
        (empirical_line,) = axis.step(
            x,
            y,
            where="post",
            color="#222222",
            linestyle="-",
            linewidth=2.8,
            label="QE1.9d empirical",
        )
        if axis is axes[0]:
            handles.append(empirical_line)
            labels.append("QE1.9d empirical")
        axis.set_title(title, fontsize=13, weight="semibold")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.005)
        axis.set_xlabel("Normalized pairwise Hamming distance")
        axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.65)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Cumulative fraction of sequence pairs")
    figure.suptitle("EF-Tu forward-evolution distance distributions", fontsize=15, y=0.98)
    # Matplotlib fills multi-column legends by column; reorder to display rows
    # in the same mu-major order used to draw the curves.
    legend_order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
    figure.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        handlelength=3.2,
        columnspacing=1.4,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.89, bottom=0.20, wspace=0.10)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_summary(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature-archive", type=Path, required=True)
    parser.add_argument("--tau1-dir", type=Path, required=True)
    parser.add_argument("--empirical-aa", type=Path, required=True)
    parser.add_argument("--empirical-3di", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    output = args.outdir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    asr_dir = output / "asr_inputs"
    distance_dir = output / "hamming_distances"

    canonical_tree = (args.tau1_dir / "mu_1" / "tree.nwk").read_text(encoding="utf-8")
    tree = Phylo.read(io.StringIO(canonical_tree), "newick")
    tip_ids = {str(clade.name) for clade in tree.get_terminals()}
    if len(tip_ids) != 39:
        raise ValueError(f"tree has {len(tip_ids)} unique tip identifiers; expected 39")
    (asr_dir / "ef_tu_tree.nwk").parent.mkdir(parents=True, exist_ok=True)
    (asr_dir / "ef_tu_tree.nwk").write_text(canonical_tree, encoding="utf-8")

    synthetic: dict[tuple[int, float], dict[str, OrderedDict[str, str]]] = {}
    sources: dict[tuple[int, float], str] = {}
    for mu in MUS:
        aa_path = args.tau1_dir / f"mu_{mu}" / "leaves_aa.fasta"
        three_di_path = args.tau1_dir / f"mu_{mu}" / "leaves_3di.fasta"
        tree_text = (args.tau1_dir / f"mu_{mu}" / "tree.nwk").read_text(encoding="utf-8")
        if tree_text != canonical_tree:
            raise ValueError(f"tau=1, mu={mu} tree differs from the canonical tree")
        synthetic[(mu, 1.0)] = {"aa": read_fasta(aa_path), "3di": read_fasta(three_di_path)}
        sources[(mu, 1.0)] = str(aa_path.parent.resolve())

    with tarfile.open(args.temperature_archive, "r:gz") as bundle:
        for tau, tau_dir in ((1.5, "tau_1p5"), (2.0, "tau_2")):
            for mu in MUS:
                prefix = f"{tau_dir}/results/mu_{mu}"
                tree_text = tar_text(bundle, f"{prefix}/simulation/tree.nwk")
                if tree_text != canonical_tree:
                    raise ValueError(f"tau={tau:g}, mu={mu} tree differs from the canonical tree")
                synthetic[(mu, tau)] = {
                    "aa": parse_fasta_text(
                        tar_text(bundle, f"{prefix}/simulation/leaves.fasta"),
                        f"{args.temperature_archive}:{prefix}/simulation/leaves.fasta",
                    ),
                    "3di": parse_fasta_text(
                        tar_text(bundle, f"{prefix}/leaves_3di.fasta"),
                        f"{args.temperature_archive}:{prefix}/leaves_3di.fasta",
                    ),
                }
                sources[(mu, tau)] = f"{args.temperature_archive.resolve()}:{prefix}"

    conditions = []
    for mu in MUS:
        for tau in TAUS:
            slug = condition_slug(mu, tau)
            records_by_modality = synthetic[(mu, tau)]
            aa_length = validate_records(
                records_by_modality["aa"],
                source=f"{slug} AA",
                expected_ids=tip_ids,
                expected_count=39,
                expected_length=396,
            )
            three_di_length = validate_records(
                records_by_modality["3di"],
                source=f"{slug} 3Di",
                expected_ids=tip_ids,
                expected_count=39,
                expected_length=396,
            )
            if list(records_by_modality["aa"]) != list(records_by_modality["3di"]):
                raise ValueError(f"{slug} AA and 3Di FASTA orders differ")
            aa_name = f"ef_tu_tips_{slug}_aa.fasta"
            three_di_name = f"ef_tu_tips_{slug}_3di.fasta"
            write_fasta(asr_dir / "aa" / aa_name, records_by_modality["aa"])
            write_fasta(asr_dir / "3di" / three_di_name, records_by_modality["3di"])
            conditions.append(
                {
                    "condition": slug,
                    "mu": mu,
                    "tau": format(tau, "g"),
                    "tips": 39,
                    "aa_length": aa_length,
                    "three_di_length": three_di_length,
                    "aa_fasta": f"aa/{aa_name}",
                    "three_di_fasta": f"3di/{three_di_name}",
                    "tree": "ef_tu_tree.nwk",
                    "source": sources[(mu, tau)],
                }
            )
    write_summary(asr_dir / "conditions.tsv", conditions)

    empirical = {
        "aa": read_fasta(args.empirical_aa),
        "3di": read_fasta(args.empirical_3di),
    }
    validate_records(empirical["aa"], source=str(args.empirical_aa))
    validate_records(empirical["3di"], source=str(args.empirical_3di))
    write_fasta(output / "empirical_inputs" / "QE1.9d_AA_SEQ.fasta", empirical["aa"])
    write_fasta(output / "empirical_inputs" / "QE1.9d_3Di.fasta", empirical["3di"])

    distance_values: dict[str, dict[str, np.ndarray]] = {"aa": {}, "3di": {}}
    summary_rows = []
    for modality in ("aa", "3di"):
        for mu in MUS:
            for tau in TAUS:
                slug = condition_slug(mu, tau)
                values, summary = pairwise_hamming(
                    synthetic[(mu, tau)][modality],
                    distance_dir / modality / f"{slug}.tsv",
                )
                distance_values[modality][slug] = values
                summary_rows.append(
                    {
                        "modality": modality,
                        "dataset": slug,
                        "mu": mu,
                        "tau": format(tau, "g"),
                        **summary,
                    }
                )
        values, summary = pairwise_hamming(
            empirical[modality], distance_dir / modality / "qe1p9d_empirical.tsv"
        )
        distance_values[modality]["empirical"] = values
        summary_rows.append(
            {
                "modality": modality,
                "dataset": "QE1.9d empirical",
                "mu": "",
                "tau": "",
                **summary,
            }
        )
    write_summary(distance_dir / "hamming_summary.tsv", summary_rows)
    plot_ecdfs(distance_values, output / "pairwise_hamming_ecdf_aa_3di")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_conditions": len(synthetic),
        "synthetic_tips_per_condition": 39,
        "synthetic_sequence_length": 396,
        "distance": "within-dataset normalized pairwise Hamming distance",
        "normalization": "mismatched aligned states divided by aligned length",
        "gap_policy": "alignment gaps retained as a state in empirical FASTAs",
        "temperature_archive": str(args.temperature_archive.resolve()),
        "tau1_directory": str(args.tau1_dir.resolve()),
        "empirical_aa": str(args.empirical_aa.resolve()),
        "empirical_3di": str(args.empirical_3di.resolve()),
        "outputs": {
            "asr_inputs": "asr_inputs/",
            "distance_tables": "hamming_distances/",
            "figure_png": "pairwise_hamming_ecdf_aa_3di.png",
            "figure_pdf": "pairwise_hamming_ecdf_aa_3di.pdf",
        },
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        """# EF-Tu mutation-scale and temperature analysis

`asr_inputs/aa/` contains nine amino-acid tip FASTAs, one for each combination
of `mu = 1, 2, 3` and `tau = 1, 1.5, 2`. Each FASTA has 39 records whose
identifiers exactly match the terminal labels in `asr_inputs/ef_tu_tree.nwk`.
`asr_inputs/3di/` contains the corresponding predicted 3Di tip FASTAs.
`asr_inputs/conditions.tsv` maps every condition to its files and source.

`hamming_distances/aa/` and `hamming_distances/3di/` contain all within-data-set
pairwise distances for the nine synthetic conditions and the QE1.9d empirical
data. Each row records the two identifiers, mismatch count, aligned length, and
normalized Hamming distance. `hamming_summary.tsv` provides distribution
summaries. Empirical gaps are retained as an aligned state.

`pairwise_hamming_ecdf_aa_3di.png` and `.pdf` show exact empirical cumulative
distribution functions side by side for amino-acid and predicted 3Di distance.
Color identifies `mu`, line style identifies `tau`, and black identifies the
QE1.9d empirical data.
""",
        encoding="utf-8",
    )
    print(f"Wrote temperature-sweep ASR inputs and Hamming analysis to {output}")


if __name__ == "__main__":
    main()
