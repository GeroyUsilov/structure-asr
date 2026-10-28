#!/usr/bin/env python3
"""Monte Carlo placement of true ancestral sequences under ASR marginals.

For every mapped internal node, this script treats the IQ-TREE ``.state``
sitewise marginal posteriors as an independent-site product distribution.
Sequences are sampled from that product, then both sampled and ground-truth
sequences are scored as the sum of sitewise log posterior probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AA_ORDER = tuple("ARNDCQEGHILKMFPSTWYV")
CONDITIONS = OrderedDict(
    (
        (
            "mu1_tau1p5",
            {
                "tau": 1.5,
                "aa_truth": "ground_truth_mu1_tau1p5_aa.fasta",
                "three_di_truth": "ground_truth_mu1_tau1p5_3di.fasta",
            },
        ),
        (
            "mu1_tau2",
            {
                "tau": 2.0,
                "aa_truth": "ground_truth_mu1_tau2_aa.fasta",
                "three_di_truth": "ground_truth_mu1_tau2_3di.fasta",
            },
        ),
    )
)
MODALITIES = ("aa", "3di")
REPRESENTATIVE_NODES = ("Node1", "Node29", "Node34")


def read_fasta(path: Path) -> OrderedDict[str, str]:
    records: OrderedDict[str, str] = OrderedDict()
    name = None
    parts: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records[name] = "".join(parts).upper()
                name = line[1:].split()[0]
                parts = []
            else:
                if name is None:
                    raise ValueError(f"Sequence before FASTA header in {path}")
                parts.append(line)
    if name is not None:
        records[name] = "".join(parts).upper()
    if not records:
        raise ValueError(f"No FASTA records in {path}")
    return records


def read_mapping(path: Path, condition: str, modality: str) -> OrderedDict[str, str]:
    result: OrderedDict[str, str] = OrderedDict()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["condition"] != condition or row["modality"] != modality:
                continue
            if row["matched"].lower() != "true":
                continue
            result[row["inferred_node"]] = row["ground_truth_node"]
    if not result:
        raise ValueError(f"No mapping for {condition}/{modality} in {path}")
    return result


def read_state(path: Path) -> tuple[tuple[str, ...], OrderedDict[str, np.ndarray]]:
    header = None
    rows: OrderedDict[str, list[list[float]]] = OrderedDict()
    sites: dict[str, list[int]] = {}
    with path.open(newline="") as handle:
        for raw in handle:
            if raw.startswith("#") or not raw.strip():
                continue
            fields = raw.rstrip("\n").split("\t")
            if header is None:
                header = fields
                continue
            node = fields[0]
            site = int(fields[1])
            rows.setdefault(node, []).append([float(x) for x in fields[3:]])
            sites.setdefault(node, []).append(site)
    if header is None:
        raise ValueError(f"No table header in {path}")
    alphabet = tuple(column[2:] for column in header[3:])
    if alphabet != AA_ORDER:
        raise ValueError(f"Unexpected alphabet in {path}: {alphabet}")
    arrays: OrderedDict[str, np.ndarray] = OrderedDict()
    for node, values in rows.items():
        if sites[node] != list(range(1, len(values) + 1)):
            raise ValueError(f"Non-consecutive sites for {node} in {path}")
        matrix = np.asarray(values, dtype=np.float64)
        row_sums = matrix.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError(f"Zero-sum probability row for {node} in {path}")
        arrays[node] = matrix / row_sums[:, None]
    return alphabet, arrays


def sample_log_scores(
    probabilities: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    probability_floor: float,
) -> np.ndarray:
    scores = np.zeros(n_samples, dtype=np.float64)
    log_probabilities = np.log(np.maximum(probabilities, probability_floor))
    for site in range(probabilities.shape[0]):
        states = rng.choice(probabilities.shape[1], size=n_samples, p=probabilities[site])
        scores += log_probabilities[site, states]
    return scores


def truth_log_score(
    sequence: str,
    probabilities: np.ndarray,
    alphabet: tuple[str, ...],
    probability_floor: float,
) -> tuple[float, int]:
    if len(sequence) != probabilities.shape[0]:
        raise ValueError(
            f"Truth length {len(sequence)} differs from state length {probabilities.shape[0]}"
        )
    index = {state: i for i, state in enumerate(alphabet)}
    total = 0.0
    zero_count = 0
    for site, state in enumerate(sequence):
        if state not in index:
            raise ValueError(f"Unsupported state {state!r} at site {site + 1}")
        value = probabilities[site, index[state]]
        if value <= 0:
            zero_count += 1
        total += math.log(max(value, probability_floor))
    return total, zero_count


def summarize_scores(samples: np.ndarray, truth: float) -> dict[str, float]:
    mean = float(np.mean(samples))
    sd = float(np.std(samples, ddof=1))
    lower = (1 + int(np.count_nonzero(samples <= truth))) / (len(samples) + 1)
    upper = (1 + int(np.count_nonzero(samples >= truth))) / (len(samples) + 1)
    return {
        "sample_mean": mean,
        "sample_sd": sd,
        "sample_q025": float(np.quantile(samples, 0.025)),
        "sample_q50": float(np.quantile(samples, 0.5)),
        "sample_q975": float(np.quantile(samples, 0.975)),
        "truth_percentile": 100.0 * lower,
        "lower_tail_p": lower,
        "upper_tail_p": upper,
        "two_sided_tail_p": min(1.0, 2.0 * min(lower, upper)),
        "truth_z": (truth - mean) / sd if sd > 0 else float("nan"),
    }


def histogram_payload(samples: np.ndarray, truth: float, bins: int = 44) -> dict:
    # Bin the Monte Carlo distribution itself.  A far-outlying truth score is
    # retained separately so a display can give the sampled bulk useful width.
    lo = float(np.min(samples))
    hi = float(np.max(samples))
    pad = max((hi - lo) * 0.02, 1e-8)
    counts, edges = np.histogram(samples, bins=bins, range=(lo - pad, hi + pad))
    return {
        "edges": [round(float(x), 8) for x in edges],
        "counts": [int(x) for x in counts],
        "truth": round(float(truth), 8),
    }


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def draw_histogram(ax, samples, truth, title, percentile, color):
    ax.hist(samples, bins=44, color=color, alpha=0.76, edgecolor="none")
    ax.axvline(truth, color="#a52125", linewidth=2.0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Mean log posterior probability per site")
    ax.set_ylabel("Monte Carlo samples")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.text(
        0.98,
        0.94,
        f"truth: {percentile:.2f}th percentile",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#a52125",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--aa-truth-dir", type=Path, required=True)
    parser.add_argument("--three-di-truth-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=731024)
    parser.add_argument("--probability-floor", type=float, default=5e-6)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    node_rows: list[dict] = []
    dataset_rows: list[dict] = []
    sample_rows: list[dict] = []
    plot_data: dict[tuple[str, str], dict] = {}
    rep_data: dict[tuple[str, str, str], dict] = {}
    inline = {"datasets": [], "representatives": []}

    for condition, info in CONDITIONS.items():
        for modality in MODALITIES:
            truth_name = info["aa_truth"] if modality == "aa" else info["three_di_truth"]
            truth_dir = args.aa_truth_dir if modality == "aa" else args.three_di_truth_dir
            truth_records = read_fasta(truth_dir / truth_name)
            mapping = read_mapping(args.mapping, condition, modality)
            state_path = args.state_dir / f"ef_tu_tips_{condition}_{modality}.fasta.state"
            alphabet, state_nodes = read_state(state_path)

            dataset_samples = np.zeros(args.samples, dtype=np.float64)
            dataset_truth = 0.0
            dataset_sites = 0
            dataset_zero_count = 0
            for inferred_node, truth_node in mapping.items():
                if inferred_node not in state_nodes:
                    raise KeyError(f"{inferred_node} missing from {state_path}")
                if truth_node not in truth_records:
                    raise KeyError(f"{truth_node} missing from {truth_dir / truth_name}")
                probabilities = state_nodes[inferred_node]
                sequence = truth_records[truth_node]
                truth_total, zero_count = truth_log_score(
                    sequence, probabilities, alphabet, args.probability_floor
                )
                samples_total = sample_log_scores(
                    probabilities, args.samples, rng, args.probability_floor
                )
                length = probabilities.shape[0]
                truth_mean = truth_total / length
                samples_mean = samples_total / length
                summary = summarize_scores(samples_mean, truth_mean)
                node_rows.append(
                    {
                        "condition": condition,
                        "tau": info["tau"],
                        "modality": modality,
                        "inferred_node": inferred_node,
                        "ground_truth_node": truth_node,
                        "sites": length,
                        "zero_probability_truth_sites": zero_count,
                        "truth_log_score": truth_total,
                        "truth_mean_log_score_per_site": truth_mean,
                        **summary,
                        "n_samples": args.samples,
                    }
                )
                dataset_samples += samples_total
                dataset_truth += truth_total
                dataset_sites += length
                dataset_zero_count += zero_count
                if inferred_node in REPRESENTATIVE_NODES:
                    rep_data[(condition, modality, inferred_node)] = {
                        "samples": samples_mean,
                        "truth": truth_mean,
                        "percentile": summary["truth_percentile"],
                        "truth_node": truth_node,
                    }
                    inline["representatives"].append(
                        {
                            "condition": condition,
                            "tau": info["tau"],
                            "modality": modality,
                            "node": inferred_node,
                            "ground_truth_node": truth_node,
                            "percentile": round(summary["truth_percentile"], 5),
                            "z": round(summary["truth_z"], 5),
                            **histogram_payload(samples_mean, truth_mean),
                        }
                    )
                    for i, score in enumerate(samples_mean):
                        sample_rows.append(
                            {
                                "condition": condition,
                                "tau": info["tau"],
                                "modality": modality,
                                "inferred_node": inferred_node,
                                "ground_truth_node": truth_node,
                                "sample": i + 1,
                                "mean_log_score_per_site": score,
                            }
                        )

            dataset_samples_mean = dataset_samples / dataset_sites
            dataset_truth_mean = dataset_truth / dataset_sites
            summary = summarize_scores(dataset_samples_mean, dataset_truth_mean)
            dataset_rows.append(
                {
                    "condition": condition,
                    "tau": info["tau"],
                    "modality": modality,
                    "nodes": len(mapping),
                    "sites": dataset_sites,
                    "zero_probability_truth_sites": dataset_zero_count,
                    "truth_log_score": dataset_truth,
                    "truth_mean_log_score_per_site": dataset_truth_mean,
                    **summary,
                    "n_samples": args.samples,
                }
            )
            plot_data[(condition, modality)] = {
                "samples": dataset_samples_mean,
                "truth": dataset_truth_mean,
                "percentile": summary["truth_percentile"],
            }
            inline["datasets"].append(
                {
                    "condition": condition,
                    "tau": info["tau"],
                    "modality": modality,
                    "nodes": len(mapping),
                    "sites": dataset_sites,
                    "percentile": round(summary["truth_percentile"], 5),
                    "z": round(summary["truth_z"], 5),
                    "two_sided_tail_p": round(summary["two_sided_tail_p"], 8),
                    **histogram_payload(dataset_samples_mean, dataset_truth_mean),
                }
            )

    node_fields = [
        "condition", "tau", "modality", "inferred_node", "ground_truth_node",
        "sites", "zero_probability_truth_sites", "truth_log_score",
        "truth_mean_log_score_per_site", "sample_mean", "sample_sd",
        "sample_q025", "sample_q50", "sample_q975", "truth_percentile",
        "lower_tail_p", "upper_tail_p", "two_sided_tail_p", "truth_z", "n_samples",
    ]
    dataset_fields = [
        "condition", "tau", "modality", "nodes", "sites",
        "zero_probability_truth_sites", "truth_log_score",
        "truth_mean_log_score_per_site", "sample_mean", "sample_sd",
        "sample_q025", "sample_q50", "sample_q975", "truth_percentile",
        "lower_tail_p", "upper_tail_p", "two_sided_tail_p", "truth_z", "n_samples",
    ]
    sample_fields = [
        "condition", "tau", "modality", "inferred_node", "ground_truth_node",
        "sample", "mean_log_score_per_site",
    ]
    write_tsv(args.outdir / "node_score_placement.tsv", node_rows, node_fields)
    write_tsv(args.outdir / "dataset_score_placement.tsv", dataset_rows, dataset_fields)
    write_tsv(args.outdir / "representative_node_samples.tsv", sample_rows, sample_fields)
    with (args.outdir / "histogram_data.json").open("w") as handle:
        json.dump(inline, handle, indent=2)

    colors = {"aa": "#2b6cb0", "3di": "#d97706"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for row_index, (condition, info) in enumerate(CONDITIONS.items()):
        for col_index, modality in enumerate(MODALITIES):
            datum = plot_data[(condition, modality)]
            draw_histogram(
                axes[row_index, col_index], datum["samples"], datum["truth"],
                f"tau = {info['tau']:g} · {modality.upper()} · 37 internal nodes",
                datum["percentile"], colors[modality],
            )
    fig.suptitle(
        "Ground-truth ancestral-set score under IQ-TREE marginal posteriors",
        fontsize=13,
    )
    fig.savefig(args.outdir / "dataset_score_histograms.png", dpi=220)
    fig.savefig(args.outdir / "dataset_score_histograms.pdf")
    plt.close(fig)

    for condition, info in CONDITIONS.items():
        fig, axes = plt.subplots(3, 2, figsize=(11, 9.4), constrained_layout=True)
        for row_index, node in enumerate(REPRESENTATIVE_NODES):
            for col_index, modality in enumerate(MODALITIES):
                datum = rep_data[(condition, modality, node)]
                draw_histogram(
                    axes[row_index, col_index], datum["samples"], datum["truth"],
                    f"{node} → {datum['truth_node']} · {modality.upper()}",
                    datum["percentile"], colors[modality],
                )
        fig.suptitle(
            f"Ground-truth node scores among posterior samples (mu = 1, tau = {info['tau']:g})",
            fontsize=13,
        )
        fig.savefig(args.outdir / f"representative_node_histograms_{condition}.png", dpi=220)
        fig.savefig(args.outdir / f"representative_node_histograms_{condition}.pdf")
        plt.close(fig)

    manifest = {
        "method": "independent-site sampling from IQ-TREE marginal posterior; score=sum_i log p_i(state_i)",
        "conditions": list(CONDITIONS),
        "modalities": list(MODALITIES),
        "representative_nodes": list(REPRESENTATIVE_NODES),
        "samples": args.samples,
        "seed": args.seed,
        "probability_floor_for_scoring": args.probability_floor,
        "sampling_uses_unfloored_normalized_printed_probabilities": True,
        "root_excluded": True,
        "mapped_internal_nodes_per_dataset": 37,
    }
    with (args.outdir / "analysis_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
