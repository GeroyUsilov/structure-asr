#!/usr/bin/env python3
"""Run exact single-site ESM2 evolution on EF-Tu at requested rate scales."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from Bio import SeqIO

from forward_asr_benchmark.cli import simulate
from forward_asr_benchmark.io import read_fasta, write_fasta
from forward_asr_benchmark.tree import load_tree


ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
SOURCE_URL = "https://luthey-schulten.chemistry.illinois.edu/tutorials/ef-tu/eftu.tar.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rate_label(rate: float) -> str:
    return format(rate, "g").replace("-", "m").replace(".", "p")


def exact_planned_attempts(tree, sequence_length: int, rate: float, seed: int) -> int:
    """Reproduce the simulator's independent stochastic rounding per branch."""
    edges = [
        (parent, child)
        for parent in tree.find_clades(order="preorder")
        for child in parent.clades
    ]
    edge_seeds = np.random.SeedSequence(seed).spawn(len(edges))
    total = 0
    for (_parent, child), edge_seed in zip(edges, edge_seeds):
        value = rate * sequence_length * float(child.branch_length or 0.0)
        base = math.floor(value)
        rng = np.random.default_rng(edge_seed)
        total += base + int(rng.random() < value - base)
    return total


def choose_root(alignment: Path, accession: str) -> tuple[str, str]:
    matches = [
        record
        for record in SeqIO.parse(str(alignment), "fasta")
        if accession == record.id or accession in record.id.split("|")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"root accession {accession!r} matched {len(matches)} alignment records"
        )
    record = matches[0]
    sequence = str(record.seq).replace("-", "").replace(".", "").upper()
    invalid = set(sequence) - ALPHABET
    if invalid:
        raise RuntimeError(f"root contains noncanonical residues: {sorted(invalid)}")
    return record.id, sequence


def write_config(
    path: Path,
    *,
    tree: Path,
    root: Path,
    output: Path,
    timings: Path,
    checkpoint: Path,
    rate: float,
    planned_attempts: int,
    seed: int,
    model_name: str,
    device: str,
    dtype: str,
    proposal_temperature: float,
) -> None:
    content = f'''[simulation]
tree = {json.dumps(str(tree))}
root_fasta = {json.dumps(str(root))}
output_dir = {json.dumps(str(output))}
mutation_scale = {rate}
temperature = 1.0
seed = {seed}
kernel = "exact_mlm_gibbs"
proposal = "exact_mlm"
proposal_temperature = {proposal_temperature}
alphabet = "ACDEFGHIKLMNPQRSTVWY"
checkpoint_path = {json.dumps(str(checkpoint))}
checkpoint_every_branches = 1

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
resume_timings = true
progress_total = {planned_attempts}
progress_description = {json.dumps(f"mu={rate:g} ESM2")}
'''
    path.write_text(content, encoding="utf-8")


def reconcile_timings_with_checkpoint(checkpoint: Path, timings: Path) -> int:
    """Discard calls from an interrupted, not-yet-checkpointed branch."""
    expected_header = (
        "pass\tsequence_length\tsite_zero_based\tforward_seconds\t"
        "end_to_end_seconds"
    )
    completed_events = 0
    if checkpoint.exists():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        completed_events = len(payload.get("events", []))
    if not timings.exists():
        return completed_events
    lines = timings.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != expected_header:
        raise RuntimeError(f"invalid timing file header: {timings}")
    if len(lines) - 1 < completed_events:
        raise RuntimeError(
            f"checkpoint has {completed_events} events but timing file has {len(lines) - 1} rows"
        )
    timings.write_text(
        "\n".join(lines[: completed_events + 1]) + "\n", encoding="utf-8"
    )
    return completed_events


def timing_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise RuntimeError("cannot summarize an empty timing series")
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


def summarize_completed_run(
    *,
    rate: float,
    planned_attempts: int,
    run_root: Path,
) -> dict:
    output = run_root / "simulation"
    timings = run_root / "model_pass_timings.tsv"
    with (output / "events.tsv").open(encoding="utf-8", newline="") as handle:
        events = list(csv.DictReader(handle, delimiter="\t"))
    with timings.open(encoding="utf-8", newline="") as handle:
        timing_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(events) != planned_attempts:
        raise RuntimeError(
            f"mu={rate:g} produced {len(events)} events; planned {planned_attempts}"
        )
    if len(timing_rows) != len(events):
        raise RuntimeError(
            f"mu={rate:g} has {len(timing_rows)} timing rows for {len(events)} events"
        )
    if any(row["accepted"].lower() != "true" for row in events):
        raise RuntimeError(f"mu={rate:g} contains a rejected exact-MLM update")
    forward = [float(row["forward_seconds"]) for row in timing_rows]
    end_to_end = [float(row["end_to_end_seconds"]) for row in timing_rows]
    if any(not math.isfinite(value) or value <= 0 for value in forward + end_to_end):
        raise RuntimeError(f"mu={rate:g} contains an invalid model timing")
    nodes = read_fasta(output / "ground_truth_nodes.fasta")
    leaves = read_fasta(output / "leaves.fasta")
    return {
        "mutation_scale": rate,
        "planned_attempts": planned_attempts,
        "observed_attempts": len(events),
        "realized_residue_changes": sum(
            row["old_state"] != row["new_state"] for row in events
        ),
        "node_count": len(nodes),
        "leaf_count": len(leaves),
        "forward_timing": timing_summary(forward),
        "end_to_end_timing": timing_summary(end_to_end),
        "run_directory": str(run_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--root-accession", default="163840912")
    parser.add_argument("--rates", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--seed", type=int, default=731024)
    parser.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--proposal-temperature", type=float, default=1.0)
    parser.add_argument("--max-attempts-per-run", type=int, default=100_000)
    parser.add_argument("--seconds-per-call", type=float, default=0.04)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    if any(rate <= 0 for rate in args.rates):
        raise ValueError("all mutation scales must be positive")
    if args.proposal_temperature <= 0 or args.seconds_per_call <= 0:
        raise ValueError("temperatures and timing estimates must be positive")

    tree_path = args.tree.resolve()
    alignment_path = args.alignment.resolve()
    tree = load_tree(tree_path)
    source_record, root_sequence = choose_root(alignment_path, args.root_accession)
    args.outdir.mkdir(parents=True, exist_ok=True)
    root_path = args.outdir / "root_sequence.fasta"
    write_fasta(root_path, {f"EF_Tu_root_{args.root_accession}": root_sequence})

    branch_lengths = [
        float(clade.branch_length or 0.0)
        for clade in tree.find_clades()
        if clade is not tree.root
    ]
    plan = []
    for rate in args.rates:
        attempts = exact_planned_attempts(tree, len(root_sequence), rate, args.seed)
        if attempts > args.max_attempts_per_run:
            raise RuntimeError(
                f"mu={rate:g} plans {attempts:,} model calls, exceeding "
                f"--max-attempts-per-run={args.max_attempts_per_run:,}"
            )
        plan.append(
            {
                "mutation_scale": rate,
                "planned_attempts": attempts,
                "estimated_forward_seconds": attempts * args.seconds_per_call,
            }
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "exact random-site single-mask ESM2 Gibbs generator",
        "model_name": args.model_name,
        "dtype": args.dtype,
        "device": args.device,
        "seed_shared_across_rates": args.seed,
        "proposal_temperature": args.proposal_temperature,
        "rate_definition": "updates on branch b = stochastic_round(mu * sequence_length * b)",
        "source_url": SOURCE_URL,
        "tree": {
            "path": str(tree_path),
            "sha256": sha256(tree_path),
            "tips": len(tree.get_terminals()),
            "nodes": len(list(tree.find_clades())),
            "branches": len(branch_lengths),
            "total_branch_length": sum(branch_lengths),
        },
        "alignment": {
            "path": str(alignment_path),
            "sha256": sha256(alignment_path),
            "selected_record": source_record,
        },
        "root": {
            "path": str(root_path),
            "accession": args.root_accession,
            "length": len(root_sequence),
            "sha256": hashlib.sha256(root_sequence.encode()).hexdigest(),
        },
        "timing_estimate_seconds_per_call": args.seconds_per_call,
        "jobs": plan,
    }
    (args.outdir / "sweep_plan.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if args.plan_only:
        return

    completed = []
    for job in plan:
        rate = float(job["mutation_scale"])
        run_root = args.outdir / f"mu_{rate_label(rate)}"
        output = run_root / "simulation"
        timings = run_root / "model_pass_timings.tsv"
        checkpoint = run_root / "checkpoint.json"
        config = run_root / "config.toml"
        run_root.mkdir(parents=True, exist_ok=True)
        if (output / "manifest.json").exists() and timings.exists():
            print(f"Validating completed mu={rate:g} run", flush=True)
        else:
            if output.exists():
                raise RuntimeError(
                    f"partial output directory blocks restart: {output}; move it aside and rerun"
                )
            write_config(
                config,
                tree=tree_path,
                root=root_path.resolve(),
                output=output.resolve(),
                timings=timings.resolve(),
                checkpoint=checkpoint.resolve(),
                rate=rate,
                planned_attempts=int(job["planned_attempts"]),
                seed=args.seed,
                model_name=args.model_name,
                device=args.device,
                dtype=args.dtype,
                proposal_temperature=args.proposal_temperature,
            )
            resumed_calls = reconcile_timings_with_checkpoint(checkpoint, timings)
            if resumed_calls:
                print(
                    f"Resuming mu={rate:g} after {resumed_calls:,} checkpointed calls",
                    flush=True,
                )
            print(
                f"Running mu={rate:g}: {int(job['planned_attempts']):,} exact ESM2 calls",
                flush=True,
            )
            simulate(str(config))
        completed.append(
            summarize_completed_run(
                rate=rate,
                planned_attempts=int(job["planned_attempts"]),
                run_root=run_root,
            )
        )
        (args.outdir / "sweep_summary.json").write_text(
            json.dumps(
                {
                    "status": "complete" if len(completed) == len(plan) else "in_progress",
                    "completed_runs": completed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"{len(plan)}-RATE EF-TU EXACT ESM2 SWEEP PASSED", flush=True)


if __name__ == "__main__":
    main()
