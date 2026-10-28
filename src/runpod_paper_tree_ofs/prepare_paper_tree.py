#!/usr/bin/env python3
"""Rebuild the beta-lactamase topology used by the phyloDCA paper.

The authors' public repository excludes its generated Newick via .gitignore,
but publishes the exact input MSA and tree-building code.  This script applies
the documented FastTree command and post-processing to that pinned MSA, writes
the complete reconstructed tree, and extracts a deterministic small subtree for
GPU-heavy OFS pilot simulations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from Bio import Phylo, SeqIO
from Bio.Phylo.BaseTree import Tree


PHYLODCA_REPOSITORY = "https://github.com/AlyZei/phyloDCA_public.git"
PHYLODCA_COMMIT = "798728c5115d1ed7366c7be2e38e07ad4c2090f4"
MSA_RELATIVE_PATH = "data_betaLac/extant_msa/PF13354_noinsert_max19gaps_nodupl_noclose_BetaLact.faa"
EXPECTED_MSA_COUNT = 18_334
EXPECTED_ALIGNMENT_LENGTH = 202
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_fasttree() -> str:
    for candidate in ("FastTree", "fasttree", "FastTreeMP"):
        executable = shutil.which(candidate)
        if executable:
            return executable
    raise RuntimeError("FastTree executable not found; install the 'fasttree' system package")


def fasttree_version(executable: str) -> str:
    completed = subprocess.run(
        [executable], input=b"", stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    text = (completed.stderr or completed.stdout).decode("utf-8", errors="replace")
    return text.splitlines()[0] if text.splitlines() else "unknown"


def build_fasttree(msa: Path, output: Path, executable: str) -> None:
    with output.open("wb") as handle:
        completed = subprocess.run(
            [executable, "-gamma", "-wag", "-pseudo", str(msa)],
            stdout=handle,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))


def collapse_short_branches(tree: Tree, threshold: float) -> int:
    collapsed = 0
    for clade in list(tree.find_clades(order="postorder")):
        if clade is tree.root:
            continue
        length = float(clade.branch_length or 0.0)
        if length < threshold:
            tree.collapse(clade)
            collapsed += 1
    return collapsed


def collapse_unary(clade) -> int:
    collapsed = 0
    rewritten = []
    for child in clade.clades:
        collapsed += collapse_unary(child)
        while not child.is_terminal() and len(child.clades) == 1:
            grandchild = child.clades[0]
            grandchild.branch_length = float(grandchild.branch_length or 0.0) + float(
                child.branch_length or 0.0
            )
            child = grandchild
            collapsed += 1
        rewritten.append(child)
    clade.clades = rewritten
    return collapsed


def mean_root_to_leaf(tree: Tree) -> float:
    depths = tree.depths()
    values = [float(depths[leaf]) for leaf in tree.get_terminals()]
    if not values or sum(values) <= 0:
        raise ValueError("tree has no positive root-to-leaf distances")
    return sum(values) / len(values)


def normalize_mean_root_to_leaf(tree: Tree) -> float:
    mean = mean_root_to_leaf(tree)
    for clade in tree.find_clades():
        if clade is not tree.root:
            clade.branch_length = float(clade.branch_length or 0.0) / mean
    tree.root.branch_length = 0.0
    return mean


def assign_internal_names(tree: Tree) -> None:
    tree.root.name = "Root"
    counter = 0
    for clade in tree.find_clades(order="preorder"):
        if clade is tree.root or clade.is_terminal():
            continue
        counter += 1
        clade.name = f"Internal_{counter:05d}"


def deterministic_subtree(tree: Tree, target_leaves: int) -> Tree:
    candidates = []
    for clade in tree.get_nonterminals(order="preorder"):
        count = len(clade.get_terminals())
        if count < 2:
            continue
        signature = "|".join(sorted(str(leaf.name) for leaf in clade.get_terminals()))
        candidates.append((abs(count - target_leaves), 0 if count >= target_leaves else 1, signature, clade))
    if not candidates:
        raise ValueError("tree contains no eligible internal subtree")
    selected = min(candidates, key=lambda item: item[:3])[3]
    subtree = Tree(root=copy.deepcopy(selected), rooted=True)
    subtree.root.branch_length = 0.0
    normalize_mean_root_to_leaf(subtree)
    assign_internal_names(subtree)
    return subtree


def write_tree(tree: Tree, path: Path) -> None:
    Phylo.write(tree, str(path), "newick")


def extract_root(msa: Path, one_based_index: int, output: Path) -> dict:
    records = list(SeqIO.parse(str(msa), "fasta"))
    lengths = {len(record.seq) for record in records}
    if len(records) != EXPECTED_MSA_COUNT or lengths != {EXPECTED_ALIGNMENT_LENGTH}:
        raise ValueError(
            f"unexpected paper MSA dimensions: {len(records)} records, lengths={sorted(lengths)}"
        )
    if not 1 <= one_based_index <= len(records):
        raise IndexError(one_based_index)
    record = records[one_based_index - 1]
    aligned = str(record.seq).upper().replace(".", "-")
    sequence = aligned.replace("-", "")
    invalid = sorted(set(sequence) - CANONICAL_AA)
    if invalid:
        raise ValueError(f"selected root contains noncanonical residues: {invalid}")
    output.write_text(f">wt{one_based_index}|{record.id}|ungapped\n{sequence}\n", encoding="utf-8")
    return {
        "one_based_index": one_based_index,
        "source_id": record.id,
        "aligned_length": len(aligned),
        "ungapped_length": len(sequence),
        "gap_count": aligned.count("-"),
        "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
    }


def tree_stats(tree: Tree) -> dict:
    return {
        "tips": len(tree.get_terminals()),
        "internal_nodes": len(tree.get_nonterminals()),
        "mean_root_to_leaf": mean_root_to_leaf(tree),
        "total_branch_length": float(tree.total_branch_length()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--root-index", type=int, default=17_282)
    parser.add_argument("--subtree-leaves", type=int, default=8)
    parser.add_argument("--collapse-threshold", type=float, default=1e-6)
    args = parser.parse_args()

    msa = args.repo_dir / MSA_RELATIVE_PATH
    if not msa.exists() or msa.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Git-LFS paper MSA is missing or still a pointer: {msa}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    executable = resolve_fasttree()
    raw_tree = args.outdir / "paper_beta_lactamase.fasttree_raw.nwk"
    build_fasttree(msa, raw_tree, executable)
    tree = Phylo.read(str(raw_tree), "newick")
    collapsed_short = collapse_short_branches(tree, args.collapse_threshold)
    collapsed_unary = collapse_unary(tree.root)
    tree.root_at_midpoint()
    pre_normalization_mean = normalize_mean_root_to_leaf(tree)
    assign_internal_names(tree)
    full_tree = args.outdir / "paper_beta_lactamase_full.normalized.nwk"
    write_tree(tree, full_tree)

    subtree = deterministic_subtree(tree, args.subtree_leaves)
    actual_subtree_tips = len(subtree.get_terminals())
    subtree_path = args.outdir / f"paper_beta_lactamase_subtree{actual_subtree_tips}.normalized.nwk"
    write_tree(subtree, subtree_path)
    root_path = args.outdir / f"root_wt{args.root_index}_ungapped.fasta"
    root_info = extract_root(msa, args.root_index, root_path)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "paper-equivalent reconstruction; authors did not publish the generated Newick",
        "phyloDCA_repository": PHYLODCA_REPOSITORY,
        "phyloDCA_commit": PHYLODCA_COMMIT,
        "source_msa": MSA_RELATIVE_PATH,
        "source_msa_sha256": sha256(msa),
        "fasttree_command": [executable, "-gamma", "-wag", "-pseudo", str(msa)],
        "fasttree_version": fasttree_version(executable),
        "collapse_threshold": args.collapse_threshold,
        "collapsed_short_branches": collapsed_short,
        "collapsed_unary_nodes": collapsed_unary,
        "pre_normalization_mean_root_to_leaf": pre_normalization_mean,
        "normalization": "divide all branch lengths by mean root-to-leaf distance",
        "full_tree": {"path": full_tree.name, **tree_stats(tree)},
        "pilot_subtree": {"path": subtree_path.name, **tree_stats(subtree)},
        "root": {"path": root_path.name, **root_info},
    }
    (args.outdir / "paper_tree_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
