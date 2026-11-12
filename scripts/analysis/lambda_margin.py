"""Which λ makes the true ancestor score best under S_λ = ℓAA + λ·ℓ3Di?

Evaluation-only, run after inference (scientific rule 1). It reads ground truth
and therefore may not inform a sampler, a proposal, or a production choice of λ.
It is a diagnostic of the *objective*, not a tuning procedure.

The question needs care, because its literal form is degenerate. S_λ(truth) is

    ℓAA(truth) + λ·ℓ3Di(truth),

and ℓ3Di(truth) is a log probability, so it is negative; S_λ(truth) therefore
decreases monotonically in λ and is maximised at λ = 0 for every node. Absolute
score cannot pick a λ.

What is well posed is the truth's *margin* over the sequences the method would
otherwise return:

    M(λ) = S_λ(truth) − max over candidates c of S_λ(c)
         = min over c of [ (ℓAA(t) − ℓAA(c)) + λ·(ℓ3Di(t) − ℓ3Di(c)) ]

Each candidate contributes a line in λ, so M is a concave piecewise-linear
function with a genuine maximiser: the λ at which the models rank the true
ancestor as favourably as they ever will. M(λ) > 0 would mean the truth
outscores every candidate — i.e. the objective actually points at the right
sequence.

The candidate set is what this method really produced for that node: the AA-MAP
start (the sampler's start, and the λ=0 answer) plus the best-scoring sample
from each λ that was run. ℓ3Di is evaluated on the ProstT5 encoding of the
sequence, exactly as the sampler evaluates it, and AA and 3Di nodes are paired
through node_mapping.tsv rather than by node number (rule 2).

Usage: PYTHONPATH=src .venv/bin/python scripts/analysis/lambda_margin.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from structure_asr.posteriors import load_state, sequence_log_probability

GRIDS = (
    Path("results/runpod/tlsample/grid"),
    Path("results/runpod/divergence/grid"),
    Path("results/runpod/cal-1/grid"),
)
PROBABILITY_FLOOR = 5e-6
LAMBDA_GRID = np.concatenate([np.linspace(0, 2, 2001), np.linspace(2.01, 50, 4800)])


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    name = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            sequences[name] = ""
        elif name:
            sequences[name] += line
    return sequences


def node_pairs(grid: Path) -> dict[tuple[str, str], dict[str, str]]:
    """(condition, true node) -> {'aa': inferred AA node, '3di': inferred 3Di node}."""
    pairs: dict[tuple[str, str], dict[str, str]] = {}
    with open(grid / "evaluation" / "node_mapping.tsv") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["matched"] != "True":
                continue
            pairs.setdefault((row["condition"], row["ground_truth_node"]), {})[row["modality"]] = (
                row["inferred_node"]
            )
    return pairs


def analyse(grid: Path) -> list[dict]:
    pairs = node_pairs(grid)
    results = []
    for cell in sorted(p for p in grid.iterdir() if (p / "sample").is_dir()):
        aa_posterior = load_state(cell / "asr" / "aa.state", PROBABILITY_FLOOR)
        di_posterior = load_state(cell / "asr" / "3di.state", PROBABILITY_FLOOR)
        truth_aa = read_fasta(cell / "simulation" / "ground_truth_nodes.fasta")
        truth_3di = read_fasta(cell / "truth" / "ground_truth_nodes_3di.fasta")

        lambda_dirs = sorted((cell / "sample").iterdir())
        if not lambda_dirs:
            continue
        candidates: dict[str, list[tuple[str, str, str]]] = {}
        start = lambda_dirs[0]
        start_aa = read_fasta(start / "aa_map_start_aa.fasta")
        start_3di = read_fasta(start / "aa_map_start_3di.fasta")
        for node, sequence in start_aa.items():
            candidates.setdefault(node, []).append(("AA-MAP start", sequence, start_3di[node]))
        for directory in lambda_dirs:
            best_aa = read_fasta(directory / "joint_asr_best_aa.fasta")
            best_3di = read_fasta(directory / "joint_asr_best_3di.fasta")
            label = f"best sample, {directory.name.replace('lambda_', 'λ=').replace('p', '.')}"
            for node, sequence in best_aa.items():
                candidates.setdefault(node, []).append((label, sequence, best_3di[node]))

        for (condition, true_node), modalities in pairs.items():
            if condition != cell.name or "aa" not in modalities or "3di" not in modalities:
                continue
            aa_node, di_node = modalities["aa"], modalities["3di"]
            if aa_node not in candidates:
                continue
            truth_ll_aa = sequence_log_probability(aa_posterior[aa_node], truth_aa[true_node])
            truth_ll_3di = sequence_log_probability(di_posterior[di_node], truth_3di[true_node])

            lines = []
            for label, sequence, structure in candidates[aa_node]:
                intercept = truth_ll_aa - sequence_log_probability(aa_posterior[aa_node], sequence)
                slope = truth_ll_3di - sequence_log_probability(di_posterior[di_node], structure)
                lines.append((label, intercept, slope))

            margin = np.min(
                np.stack([intercept + slope * LAMBDA_GRID for _, intercept, slope in lines]), axis=0
            )
            best_index = int(np.argmax(margin))
            binding = min(lines, key=lambda line: line[1] + line[2] * LAMBDA_GRID[best_index])
            start_errors = sum(
                one != two for one, two in zip(candidates[aa_node][0][1], truth_aa[true_node])
            )
            positive = np.flatnonzero(margin > 0)
            # The margin keeps rising at the top of the grid when every candidate
            # has a worse 3Di score than the truth. That is the λ→∞ limit, where
            # the objective ignores AA entirely; it reflects a candidate set that
            # holds no 3Di-optimal sequence, not a usable optimum.
            unbounded = bool(min(slope for _, _, slope in lines) > 0)
            results.append(
                {
                    "cell": cell.name,
                    "node": aa_node,
                    "aa_errors": start_errors,
                    "truth_ll_aa": truth_ll_aa,
                    "truth_ll_3di": truth_ll_3di,
                    "best_lambda": float(LAMBDA_GRID[best_index]),
                    "best_margin": float(margin[best_index]),
                    "margin_at_zero": float(margin[0]),
                    "crossing": float(LAMBDA_GRID[positive[0]]) if positive.size else None,
                    "binding": binding[0],
                    "ever_wins": bool(positive.size),
                    "unbounded": unbounded,
                }
            )
    return results


def main() -> None:
    rows: list[dict] = []
    for grid in GRIDS:
        if (grid / "evaluation" / "node_mapping.tsv").exists():
            rows.extend(analyse(grid))

    print(f"{'cell':>28} {'node':>7} {'AAerr':>5} {'λ*':>8} {'M(0)':>9} {'λ truth wins':>13}  binding candidate at λ*")
    for row in rows:
        star = "∞" if row["unbounded"] else f"{row['best_lambda']:.3f}"
        crossing = f"λ > {row['crossing']:.2f}" if row["crossing"] is not None else "never"
        print(
            f"{row['cell'][:28]:>28} {row['node']:>7} {row['aa_errors']:>5} {star:>8} "
            f"{row['margin_at_zero']:>9.1f} {crossing:>13}  {row['binding']}"
        )

    bounded = [r for r in rows if not r["unbounded"]]
    unbounded = [r for r in rows if r["unbounded"]]
    print()
    print(f"{len(rows)} nodes analysed.")
    print(f"  {len(bounded)} have a finite λ*: the margin peaks and then falls.")
    zeros = [r for r in bounded if r["best_lambda"] == 0]
    interior = [r for r in bounded if r["best_lambda"] > 0]
    print(f"    λ* = 0 (structure never helps the truth's standing): {len(zeros)}")
    if interior:
        values = sorted(r["best_lambda"] for r in interior)
        print(f"    λ* > 0: {len(interior)}, range {values[0]:.3f} – {values[-1]:.3f}, "
              f"median {values[len(values) // 2]:.3f}")
    print(f"    of these {len(bounded)}, the truth outscores every candidate at some λ in "
          f"{sum(1 for r in bounded if r['ever_wins'])}.")
    print(f"  {len(unbounded)} have no finite λ*: every candidate's 3Di scores worse than the truth's,")
    print("    so the margin rises without bound and is maximised only as λ→∞, where the")
    print("    objective ignores the AA channel. Those crossings are listed above but are an")
    print("    artifact of a candidate set containing no 3Di-optimal sequence.")


if __name__ == "__main__":
    main()
