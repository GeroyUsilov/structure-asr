r"""Which λ maximises the probability of the true ancestral node under S_λ?

Evaluation-only, post-inference (scientific rule 1): a sanity check on the
objective, not a way to pick λ for a run.

S_λ = ℓAA + λ·ℓ3Di is an unnormalised score. Read as a distribution over a
site's (amino acid, 3Di state) pair it is

    P_λ(a, d) ∝ P_AA(a) · P_3Di(d)^λ ,

so the per-site normaliser factorises as Z_i(λ) = (Σ_a P_AA(a))·(Σ_d P_3Di(d)^λ)
= Σ_d P_3Di,i(d)^λ. The amino-acid factor is already normalised and contributes
no λ dependence, so the normalised log probability of the true ancestor is

    log P_λ(truth) = Σ_i log P_AA,i(a*_i)  +  Σ_i [ λ·log P_3Di,i(d*_i) − log Σ_d P_3Di,i(d)^λ ]
                     \_____ constant in λ _____/   \_________ all the λ dependence _________/

λ* is therefore a property of the 3Di posterior alone: the tempering exponent
under which the true 3Di states are most likely. That makes it a calibration
check. λ* ≈ 1 means the 3Di posterior is calibrated and deserves unit weight;
λ* < 1 means it is overconfident and should be down-weighted; λ* > 1 means it is
underconfident. The unnormalised score itself is useless for this — it falls
monotonically in λ because ℓ3Di < 0, and is maximised at λ=0 for every node.

AA and 3Di nodes are paired by topology through node_mapping.tsv (rule 2).

Usage: PYTHONPATH=src .venv/bin/python scripts/analysis/lambda_for_truth.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from structure_asr.posteriors import load_state

GRIDS = (
    Path("results/runpod/tlsample/grid"),
    Path("results/runpod/divergence/grid"),
    Path("results/runpod/cal-1/grid"),
)
PROBABILITY_FLOOR = 5e-6
LAMBDA_GRID = np.concatenate([np.linspace(0.0, 4.0, 4001), np.linspace(4.01, 40.0, 3600)])


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


def tempered_log_likelihood(posterior, sequence: str) -> np.ndarray:
    """Σ_i [ λ·log p(true state) − log Σ_d p_d^λ ], evaluated on LAMBDA_GRID."""
    log_probabilities = posterior.log_probabilities  # (sites, states)
    index = {state: i for i, state in enumerate(posterior.alphabet)}
    columns = np.fromiter((index[state.upper()] for state in sequence), dtype=np.int64)
    true_log_p = log_probabilities[np.arange(len(sequence)), columns]  # (sites,)

    # log Σ_d p_d^λ per site, stable: logsumexp over λ·log p_d.
    scaled = log_probabilities[:, None, :] * LAMBDA_GRID[None, :, None]  # (sites, λ, states)
    ceiling = scaled.max(axis=2)
    log_partition = ceiling + np.log(np.exp(scaled - ceiling[:, :, None]).sum(axis=2))
    return (true_log_p[:, None] * LAMBDA_GRID[None, :] - log_partition).sum(axis=0)


def node_pairs(grid: Path) -> dict[tuple[str, str], dict[str, str]]:
    pairs: dict[tuple[str, str], dict[str, str]] = {}
    with open(grid / "evaluation" / "node_mapping.tsv") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["matched"] == "True":
                pairs.setdefault((row["condition"], row["ground_truth_node"]), {})[
                    row["modality"]
                ] = row["inferred_node"]
    return pairs


def main() -> None:
    rows = []
    for grid in GRIDS:
        if not (grid / "evaluation" / "node_mapping.tsv").exists():
            continue
        pairs = node_pairs(grid)
        for cell in sorted(p for p in grid.iterdir() if (p / "asr").is_dir()):
            di_posterior = load_state(cell / "asr" / "3di.state", PROBABILITY_FLOOR)
            truth_3di = read_fasta(cell / "truth" / "ground_truth_nodes_3di.fasta")
            per_node = []
            for (condition, true_node), modalities in pairs.items():
                if condition != cell.name or "3di" not in modalities:
                    continue
                di_node = modalities["3di"]
                if di_node not in di_posterior or true_node not in truth_3di:
                    continue
                curve = tempered_log_likelihood(di_posterior[di_node], truth_3di[true_node])
                best = int(np.argmax(curve))
                per_node.append(
                    {
                        "cell": cell.name,
                        "node": di_node,
                        "lambda_star": float(LAMBDA_GRID[best]),
                        "capped": best == len(LAMBDA_GRID) - 1,
                        "gain_over_lambda1": float(curve[best] - curve[np.argmin(abs(LAMBDA_GRID - 1))]),
                    }
                )
            if per_node:
                rows.extend(per_node)
                values = sorted(r["lambda_star"] for r in per_node)
                print(
                    f"{cell.name:>28}  n={len(per_node):>2}  "
                    f"λ* median {values[len(values) // 2]:.3f}  "
                    f"range {values[0]:.3f} – {values[-1]:.3f}"
                )

    values = sorted(r["lambda_star"] for r in rows)
    print()
    print(f"{len(rows)} true ancestral nodes over {len({r['cell'] for r in rows})} cells.")
    print(f"  λ* median {values[len(values) // 2]:.3f}, range {values[0]:.3f} – {values[-1]:.3f}")
    print(f"  λ* < 1 (3Di posterior overconfident): {sum(1 for v in values if v < 1)}")
    print(f"  λ* > 1 (underconfident):              {sum(1 for v in values if v > 1)}")
    if any(r["capped"] for r in rows):
        print(f"  hit the top of the grid: {sum(1 for r in rows if r['capped'])}")


if __name__ == "__main__":
    main()
