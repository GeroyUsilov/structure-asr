#!/usr/bin/env python3
"""Configure, run, and validate the exact ESM2 generator smoke test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from Bio import Phylo

from forward_asr_benchmark.cli import simulate
from forward_asr_benchmark.io import read_fasta


ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


def write_config(
    path: Path,
    *,
    output: Path,
    timings: Path,
    model_name: str,
    device: str,
    dtype: str,
) -> None:
    bundle = Path(__file__).resolve().parent
    content = f'''[simulation]
tree = {json.dumps(str(bundle / "tiny_tree.nwk"))}
root_fasta = {json.dumps(str(bundle / "root.fasta"))}
output_dir = {json.dumps(str(output))}
mutation_scale = 1.0
temperature = 1.0
seed = 731024
kernel = "exact_mlm_gibbs"
proposal = "exact_mlm"
proposal_temperature = 1.0
alphabet = "ACDEFGHIKLMNPQRSTVWY"

[[scorers]]
kind = "constant"
weight = 1.0

[exact_mlm]
factory = "runpod_exact_mlm_smoke.esm2_adapter:make_site_predictor"

[exact_mlm.parameters]
model_name = {json.dumps(model_name)}
device = {json.dumps(device)}
dtype = {json.dumps(dtype)}
timings_path = {json.dumps(str(timings))}
'''
    path.write_text(content, encoding="utf-8")


def timing_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "total_seconds": sum(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": ordered[p95_index],
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
    }


def validate(
    output: Path,
    timings: Path,
    expected_length: int,
    expected_updates: int,
) -> dict:
    nodes = read_fasta(output / "ground_truth_nodes.fasta")
    leaves = read_fasta(output / "leaves.fasta")
    if len(nodes) != 5 or len(leaves) != 3:
        raise RuntimeError(f"unexpected node counts: {len(nodes)} nodes, {len(leaves)} leaves")
    if any(len(sequence) != expected_length for sequence in nodes.values()):
        raise RuntimeError("the exact ESM2 generator changed sequence length")
    if any(set(sequence) - ALPHABET for sequence in nodes.values()):
        raise RuntimeError("a generated sequence contains a noncanonical residue")
    with (output / "events.tsv").open(encoding="utf-8", newline="") as handle:
        events = list(csv.DictReader(handle, delimiter="\t"))
    if len(events) != expected_updates:
        raise RuntimeError(f"observed {len(events)} updates, expected {expected_updates}")
    if any(row["accepted"].lower() != "true" for row in events):
        raise RuntimeError("an exact masked-LM Gibbs update was rejected")
    with (output / "node_scores.tsv").open(encoding="utf-8", newline="") as handle:
        scores = list(csv.DictReader(handle, delimiter="\t"))
    if len(scores) != len(nodes) or any(
        not math.isfinite(float(row["energy"])) for row in scores
    ):
        raise RuntimeError("node score output is invalid")
    with timings.open(encoding="utf-8", newline="") as handle:
        timing_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(timing_rows) != expected_updates:
        raise RuntimeError(
            f"observed {len(timing_rows)} timing rows, expected {expected_updates}"
        )
    if [int(row["pass"]) for row in timing_rows] != list(
        range(1, expected_updates + 1)
    ):
        raise RuntimeError("model-pass timing indices are invalid")
    forward_times = [float(row["forward_seconds"]) for row in timing_rows]
    end_to_end_times = [float(row["end_to_end_seconds"]) for row in timing_rows]
    if any(not math.isfinite(value) or value <= 0 for value in forward_times + end_to_end_times):
        raise RuntimeError("model-pass timings must be positive and finite")
    return {
        "status": "passed",
        "model_forward_passes": len(events),
        "updates": len(events),
        "realized_residue_changes": sum(
            row["old_state"] != row["new_state"] for row in events
        ),
        "nodes": len(nodes),
        "leaves": len(leaves),
        "sequence_length": expected_length,
        "output": str(output),
        "timings_tsv": str(timings),
        "forward_timing": timing_summary(forward_times),
        "end_to_end_timing": timing_summary(end_to_end_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--work-root", type=Path, default=Path("/workspace/exact_mlm_smoke"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    root_sequence = next(iter(read_fasta(bundle / "root.fasta").values()))
    tree = Phylo.read(str(bundle / "tiny_tree.nwk"), "newick")
    expected_updates = sum(
        round(len(root_sequence) * float(clade.branch_length or 0.0))
        for clade in tree.find_clades()
        if clade is not tree.root
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_parent = args.work_root / "runs" / run_id
    suffix = 1
    while run_parent.exists():
        run_parent = args.work_root / "runs" / f"{run_id}_{suffix}"
        suffix += 1
    run_parent.mkdir(parents=True)
    output = run_parent / "simulation"
    config = run_parent / "smoke_config.toml"
    timings = run_parent / "model_pass_timings.tsv"
    write_config(
        config,
        output=output,
        timings=timings,
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
    )
    simulate(str(config))
    summary = validate(output, timings, len(root_sequence), expected_updates)
    (run_parent / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("EXACT ESM2 GENERATOR SMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
