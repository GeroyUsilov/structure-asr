"""Re-evaluate the bundled, completed EF-Tu experiments without GPU inference.

Topology is matched separately for AA and 3Di reconstructions. The simulated
degree-two root has no inferred counterpart and is excluded. This module reads
ground truth only after inference; it is never imported by either sampler.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from Bio import Phylo

from structure_asr.simulation.tree import load_tree
from structure_asr.evaluation import read_fasta, score_dataset, topology_mapping, write_tsv

SITE_COLUMNS = [
    "condition", "tau", "modality", "inferred_node", "ground_truth_node", "site",
    "ground_truth_state", "map_state", "p_ground_truth", "p_map",
    "ground_truth_is_map", "printed_zero", "ln_p_ground_truth_censored",
]
CONDITIONS = (("mu1_tau1p5", 1.5), ("mu1_tau2", 2.0))


def errors(sequence: str, truth: str) -> int:
    """Aligned Hamming count; never silently truncate unequal sequences."""
    if len(sequence) != len(truth):
        raise ValueError("unaligned or unequal sequence lengths")
    return sum(a.upper() != b.upper() for a, b in zip(sequence, truth))


def evaluate(example_root: Path, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    tree = load_tree(example_root / "reference/AB_eftu_rerooted.tre")
    dataset_rows, node_rows, mapping_rows, joint_rows = [], [], [], []
    with (outdir / "site_probabilities.tsv").open("w", newline="") as handle:
        site_writer = csv.DictWriter(handle, fieldnames=SITE_COLUMNS, delimiter="\t")
        site_writer.writeheader()
        for condition, tau in CONDITIONS:
            base = example_root / condition
            mappings, truths = {}, {}
            for modality in ("aa", "3di"):
                inferred = Phylo.read(base / f"{modality}.treefile", "newick")
                mapping = topology_mapping(tree, inferred)
                if len(mapping) != 37 or not all(row["matched"] for row in mapping):
                    raise ValueError(f"expected 37 matched internal nodes: {condition} {modality}")
                mappings[modality] = {row["inferred_node"]: row["ground_truth_node"] for row in mapping}
                truths[modality] = read_fasta(base / f"truth_{modality}.fasta")
                nodes, dataset = score_dataset(
                    condition=condition, tau=tau, modality=modality,
                    state_path=base / f"{modality}.state", truth=truths[modality],
                    mapping_rows=mapping, site_writer=site_writer,
                )
                node_rows.extend(nodes)
                dataset_rows.append(dataset)
                mapping_rows.extend({"condition": condition, "tau": tau,
                                     "modality": modality, **row} for row in mapping)

            # Stored starts and encodings avoid calling ProstT5 during evaluation.
            checkpoint = json.loads((base / "legacy_results/checkpoint.json").read_text())
            manifest = json.loads((base / "legacy_results/run_manifest.json").read_text())
            if manifest["status"] != "complete":
                raise ValueError(f"incomplete inference: {condition}")
            for chain in checkpoint["chains"]:
                node = chain["node"]
                if mappings["aa"][node] != mappings["3di"][node]:
                    raise ValueError(f"AA/3Di inferred labels refer to different vertices: {node}")
                row = {"condition": condition, "node": node, "chain": chain["chain"]}
                for modality, key in (("aa", "seq"), ("3di", "struct")):
                    truth = truths[modality][mappings[modality][node]]
                    for stage in ("start", "best", "current"):
                        row[f"{modality}_{stage}_errors"] = errors(chain[f"{stage}_{key}"], truth)
                joint_rows.append(row)

    write_tsv(outdir / "dataset_summary.tsv", dataset_rows)
    write_tsv(outdir / "node_summary.tsv", node_rows)
    write_tsv(outdir / "node_mapping.tsv", mapping_rows)
    write_tsv(outdir / "legacy_joint_comparison.tsv", joint_rows)
    summary = {
        "conditions": len(CONDITIONS), "matched_nodes_per_condition_modality": 37,
        "legacy_reconstructions": len(joint_rows),
        "legacy_aa_errors_start": sum(row["aa_start_errors"] for row in joint_rows),
        "legacy_aa_errors_best": sum(row["aa_best_errors"] for row in joint_rows),
        "note": "Best means highest stored model score, not highest ground-truth identity.",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-root", type=Path, default=Path("examples/ef_tu"))
    parser.add_argument("--outdir", type=Path, default=Path("results/reproduced"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.example_root, args.outdir), indent=2))


if __name__ == "__main__":
    main()
