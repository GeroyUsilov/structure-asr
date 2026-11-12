"""Where do the sampler's accepted moves go, relative to the truth?

Evaluation-only, run after inference: it reads ground truth, so nothing here may
inform a sampler, a proposal or a choice of lambda (scientific rule 1). It
answers the question the summed error counts leave open — at lambda >= 0.5 the
chains do beat their start under S_lambda, so the moves are real; are they
right?

For each node it classifies every site where the best-scoring sample differs
from the AA-MAP start into wrong->right, right->wrong and wrong->wrong, pairing
inferred with true nodes through evaluation/node_mapping.tsv (rule 2, never by
node number).

Usage: PYTHONPATH=src .venv/bin/python experiments/.../move_audit.py <grid outdir>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

LAMBDAS = ("0", "0p02", "0p1", "0p5", "1")


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


def aa_node_mapping(outdir: Path) -> dict[tuple[str, str], str]:
    """(condition, inferred node) -> true node, for matched AA nodes only."""
    mapping = {}
    with open(outdir / "evaluation" / "node_mapping.tsv") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["modality"] == "aa" and row["matched"] == "True":
                mapping[(row["condition"], row["inferred_node"])] = row["ground_truth_node"]
    return mapping


def main(outdir: Path) -> None:
    mapping = aa_node_mapping(outdir)
    header = f"{'cell':>28} {'lambda':>6} {'node':>7} {'moved':>6} {'w->r':>5} {'r->w':>5} {'w->w':>5} {'net':>5}"
    print(header)
    totals: dict[str, list[int]] = {}
    for cell in sorted(p for p in outdir.iterdir() if (p / "sample").is_dir()):
        truth = read_fasta(cell / "simulation" / "ground_truth_nodes.fasta")
        for lam in LAMBDAS:
            base = cell / "sample" / f"lambda_{lam}"
            if not base.is_dir():
                continue
            start = read_fasta(base / "aa_map_start_aa.fasta")
            best = read_fasta(base / "best_aa.fasta")
            for key in sorted(best):
                node = key.split("_chain")[0]
                true_node = mapping.get((cell.name, node))
                if true_node is None:
                    continue
                true_sequence = truth[true_node]
                right = wrong = neither = 0
                for was, now, want in zip(start[node], best[key], true_sequence):
                    if was == now:
                        continue
                    if now == want:
                        right += 1
                    elif was == want:
                        wrong += 1
                    else:
                        neither += 1
                moved = right + wrong + neither
                print(
                    f"{cell.name:>28} {lam:>6} {node:>7} {moved:>6} "
                    f"{right:>5} {wrong:>5} {neither:>5} {wrong - right:>+5}"
                )
                tally = totals.setdefault(lam, [0, 0, 0])
                tally[0] += right
                tally[1] += wrong
                tally[2] += neither
    print()
    print(f"{'lambda':>6} {'w->r':>5} {'r->w':>5} {'w->w':>5}  right/moved")
    for lam, (right, wrong, neither) in totals.items():
        moved = right + wrong + neither
        share = f"{right / moved:.2f}" if moved else "n/a"
        print(f"{lam:>6} {right:>5} {wrong:>5} {neither:>5}  {share}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/runpod/tlsample/grid"))
