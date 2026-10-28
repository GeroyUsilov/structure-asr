#!/usr/bin/env python3
"""Launch the three-rate OFS pseudo-Gibbs sweep in one GPU process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from Bio import Phylo

from forward_asr_benchmark.cli import simulate
from forward_asr_benchmark.io import read_fasta


def planned_attempts(tree, sequence_length: int, rate: float) -> int:
    return sum(
        int(round(rate * sequence_length * float(clade.branch_length or 0.0)))
        for clade in tree.find_clades()
        if clade is not tree.root
    )


def write_config(
    path: Path,
    tree: Path,
    root: Path,
    output: Path,
    rate: float,
    seed: int,
    args,
) -> None:
    content = f'''[simulation]
tree = {json.dumps(str(tree))}
root_fasta = {json.dumps(str(root))}
output_dir = {json.dumps(str(output))}
mutation_scale = {rate}
temperature = 1.0
seed = {seed}
kernel = "ofs_pseudogibbs"
proposal = "ofs"
proposal_temperature = {args.proposal_temperature}

[[scorers]]
kind = "python"
factory = "runpod_paper_tree_ofs.ofs_adapter:make_ofs_scorer"
weight = 1.0

[scorers.parameters]
provider_factory = {json.dumps(args.provider_factory)}
model_kind = {json.dumps(args.model_kind)}
model_path = {json.dumps(args.model_path)}
head_path = {json.dumps(args.head_path)}
device = {json.dumps(args.device)}
alphabet = "ACDEFGHIKLMNPQRSTVWY"
'''
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--rates", nargs="+", type=float, default=[1.0, 10.0, 100.0])
    parser.add_argument("--seed", type=int, default=731024)
    parser.add_argument("--max-attempts", type=int, default=250_000)
    parser.add_argument("--allow-large-run", action="store_true")
    parser.add_argument("--proposal-temperature", type=float, default=1.0)
    parser.add_argument(
        "--provider-factory",
        default="runpod_paper_tree_ofs.ofsperplexity_adapter:make_profile_predictor",
    )
    parser.add_argument("--model-kind", default=os.environ.get("OFS_MODEL_KIND", "esmc"))
    parser.add_argument("--model-path", default=os.environ.get("OFS_MODEL_PATH", ""))
    parser.add_argument("--head-path", default=os.environ.get("OFS_HEAD_PATH", ""))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = json.loads((args.input_dir / "paper_tree_manifest.json").read_text())
    if args.tree is None:
        args.tree = args.input_dir / manifest["pilot_subtree"]["path"]
    root = args.input_dir / manifest["root"]["path"]
    sequence = next(iter(read_fasta(root).values()))
    tree = Phylo.read(str(args.tree), "newick")
    args.outdir.mkdir(parents=True, exist_ok=True)
    config_dir = args.outdir / "configs"
    config_dir.mkdir(exist_ok=True)

    plan = []
    for rate in args.rates:
        attempts = planned_attempts(tree, len(sequence), rate)
        plan.append({"mutation_scale": rate, "planned_attempts": attempts})
        if attempts > args.max_attempts and not args.allow_large_run:
            raise RuntimeError(
                f"mu={rate:g} plans {attempts:,} OFS calls, over --max-attempts "
                f"{args.max_attempts:,}; use a smaller subtree or --allow-large-run"
            )
    print(json.dumps({"tree": str(args.tree), "root_length": len(sequence), "jobs": plan}, indent=2))
    if not args.model_path or not args.head_path:
        raise RuntimeError("Set OFS_MODEL_PATH and OFS_HEAD_PATH before launching the sweep")

    for index, rate in enumerate(args.rates):
        label = str(rate).replace(".", "p")
        run_output = args.outdir / f"mu_{label}"
        if (run_output / "manifest.json").exists():
            print(f"Skipping completed {run_output}")
            continue
        config_path = config_dir / f"mu_{label}.toml"
        write_config(config_path, args.tree.resolve(), root.resolve(), run_output.resolve(), rate, args.seed + index, args)
        simulate(str(config_path))


if __name__ == "__main__":
    main()

