"""Stable benchmark input/output formats."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .evolve import EvolutionResult
from .tree import tree_to_newick


def read_fasta(path: str | Path) -> dict[str, str]:
    records: dict[str, str] = {}
    identifier: str | None = None
    parts: list[str] = []
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records[identifier] = "".join(parts).upper()
            identifier = line[1:].split()[0]
            if not identifier or identifier in records:
                raise ValueError(f"invalid or duplicate FASTA identifier at line {lineno}")
            parts = []
        elif identifier is None:
            raise ValueError(f"sequence data before FASTA header at line {lineno}")
        else:
            parts.append(line)
    if identifier is not None:
        records[identifier] = "".join(parts).upper()
    if not records:
        raise ValueError(f"no FASTA records in {path}")
    return records


def write_fasta(path: str | Path, records: Mapping[str, str], width: int = 80) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for identifier, sequence in records.items():
            handle.write(f">{identifier}\n")
            for start in range(0, len(sequence), width):
                handle.write(sequence[start : start + width] + "\n")


def write_run(output_dir: str | Path, tree, result: EvolutionResult, manifest: Mapping) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_fasta(output / "ground_truth_nodes.fasta", result.sequences)
    terminal_names = {str(clade.name) for clade in tree.get_terminals()}
    write_fasta(
        output / "leaves.fasta",
        {name: sequence for name, sequence in result.sequences.items() if name in terminal_names},
    )
    (output / "tree.nwk").write_text(tree_to_newick(tree) + "\n", encoding="utf-8")

    with (output / "events.tsv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(asdict(result.events[0]).keys()) if result.events else [
            "parent", "child", "attempt", "site", "old_state", "new_state", "accepted",
            "log_acceptance", "energy_before", "energy_after",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(asdict(event) for event in result.events)

    metric_names = sorted({key for score in result.scores.values() for key in score.metrics})
    with (output / "node_scores.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node", "energy", *metric_names], delimiter="\t")
        writer.writeheader()
        for node, score in result.scores.items():
            writer.writerow({"node": node, "energy": score.energy, **score.metrics})

    payload = dict(manifest)
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    payload["outputs"] = {
        "ground_truth": "ground_truth_nodes.fasta",
        "leaves": "leaves.fasta",
        "tree": "tree.nwk",
        "events": "events.tsv",
        "node_scores": "node_scores.tsv",
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

