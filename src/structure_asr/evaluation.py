#!/usr/bin/env python3
"""Score known synthetic node states under IQ-TREE ASR posteriors."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tarfile
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from Bio import Phylo

from structure_asr.simulation.tree import load_tree


ROUNDING_FLOOR = 5e-6
CONDITIONS = ((1.0, "tau1"), (1.5, "tau1p5"), (2.0, "tau2"))
MODALITIES = ("aa", "3di")


def parse_fasta_text(text: str, source: str) -> OrderedDict[str, str]:
    records: OrderedDict[str, str] = OrderedDict()
    identifier: str | None = None
    chunks: list[str] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records[identifier] = "".join(chunks).upper()
            identifier = line[1:].split()[0]
            if not identifier:
                raise ValueError(f"empty FASTA identifier in {source}:{line_number}")
            if identifier in records:
                raise ValueError(f"duplicate FASTA identifier {identifier!r} in {source}")
            chunks = []
        else:
            if identifier is None:
                raise ValueError(f"sequence before FASTA header in {source}:{line_number}")
            chunks.append(line)
    if identifier is not None:
        if identifier in records:
            raise ValueError(f"duplicate FASTA identifier {identifier!r} in {source}")
        records[identifier] = "".join(chunks).upper()
    if not records:
        raise ValueError(f"no FASTA records in {source}")
    return records


def read_fasta(path: Path) -> OrderedDict[str, str]:
    return parse_fasta_text(path.read_text(encoding="utf-8"), str(path))


def write_fasta(path: Path, records: OrderedDict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for identifier, sequence in records.items():
            handle.write(f">{identifier}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def tar_text(bundle: tarfile.TarFile, member: str) -> str:
    stream = bundle.extractfile(member)
    if stream is None:
        raise FileNotFoundError(f"missing archive member: {member}")
    return stream.read().decode("utf-8")


def build_graph(tree) -> tuple[dict[object, list[object]], set[object]]:
    adjacency: dict[object, list[object]] = defaultdict(list)
    terminals = set(tree.get_terminals())
    for parent in tree.find_clades(order="preorder"):
        adjacency.setdefault(parent, [])
        for child in parent.clades:
            adjacency[parent].append(child)
            adjacency[child].append(parent)
    return dict(adjacency), terminals


def suppress_degree_two_root(
    adjacency: dict[object, list[object]], root: object
) -> dict[object, list[object]]:
    graph = {node: list(neighbors) for node, neighbors in adjacency.items()}
    neighbors = graph[root]
    if len(neighbors) != 2:
        raise ValueError(f"expected a degree-two true root, found degree {len(neighbors)}")
    left, right = neighbors
    graph[left] = [right if node is root else node for node in graph[left]]
    graph[right] = [left if node is root else node for node in graph[right]]
    del graph[root]
    return graph


def component_tips(
    adjacency: dict[object, list[object]],
    terminals: set[object],
    start: object,
    blocked: object,
) -> tuple[str, ...]:
    stack = [start]
    seen = {blocked}
    tips: list[str] = []
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node in terminals:
            tips.append(str(node.name))
        stack.extend(adjacency[node])
    return tuple(sorted(tips))


def vertex_signatures(
    tree,
    *,
    suppress_root: bool,
) -> dict[tuple[tuple[str, ...], ...], str]:
    adjacency, terminals = build_graph(tree)
    if suppress_root:
        adjacency = suppress_degree_two_root(adjacency, tree.root)
    signatures: dict[tuple[tuple[str, ...], ...], str] = {}
    for node, neighbors in adjacency.items():
        if node in terminals:
            continue
        if len(neighbors) != 3:
            raise ValueError(
                f"internal node {node.name!r} has degree {len(neighbors)} after unrooting"
            )
        parts = tuple(
            sorted(
                component_tips(adjacency, terminals, neighbor, node)
                for neighbor in neighbors
            )
        )
        if parts in signatures:
            raise ValueError("duplicate internal-node topology signature")
        signatures[parts] = str(node.name)
    return signatures


def topology_mapping(true_tree, inferred_tree) -> list[dict[str, object]]:
    true_signatures = vertex_signatures(true_tree, suppress_root=True)
    inferred_signatures = vertex_signatures(inferred_tree, suppress_root=False)
    rows = []
    for signature, inferred_node in sorted(
        inferred_signatures.items(), key=lambda item: int(item[1].removeprefix("Node"))
    ):
        truth_node = true_signatures.get(signature)
        rows.append(
            {
                "inferred_node": inferred_node,
                "ground_truth_node": truth_node or "",
                "matched": truth_node is not None,
                "incident_clade_sizes": ",".join(str(len(part)) for part in signature),
            }
        )
    return rows


def read_state(path: Path) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    if reader.fieldnames is None:
        raise ValueError(f"missing header in {path}")
    probability_columns = [name for name in reader.fieldnames if name.startswith("p_")]
    if len(probability_columns) != 20:
        raise ValueError(f"{path} has {len(probability_columns)} probability columns")
    by_node: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in reader:
        by_node[str(row["Node"])].append(row)
    for node, rows in by_node.items():
        rows.sort(key=lambda row: int(row["Site"]))
        sites = [int(row["Site"]) for row in rows]
        if sites != list(range(1, len(rows) + 1)):
            raise ValueError(f"nonconsecutive sites for {node} in {path}")
        for row in rows:
            total = sum(float(row[column]) for column in probability_columns)
            if abs(total - 1.0) > 1.1e-4:
                raise ValueError(f"posterior row sum {total} outside rounding tolerance in {path}")
    return probability_columns, dict(by_node)


def score_dataset(
    *,
    condition: str,
    tau: float,
    modality: str,
    state_path: Path,
    truth: OrderedDict[str, str],
    mapping_rows: list[dict[str, object]],
    site_writer: csv.DictWriter,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    probability_columns, state_by_node = read_state(state_path)
    alphabet = {column.removeprefix("p_") for column in probability_columns}
    node_rows: list[dict[str, object]] = []
    all_probabilities: list[float] = []
    all_log_probabilities: list[float] = []
    all_correct: list[bool] = []
    zero_count = 0
    matched = [row for row in mapping_rows if row["matched"]]
    for mapping in matched:
        inferred_node = str(mapping["inferred_node"])
        truth_node = str(mapping["ground_truth_node"])
        sequence = truth[truth_node]
        state_rows = state_by_node[inferred_node]
        if len(state_rows) != len(sequence):
            raise ValueError(
                f"{condition} {modality} {inferred_node}: state length {len(state_rows)} "
                f"!= truth length {len(sequence)}"
            )
        node_probabilities: list[float] = []
        node_logs: list[float] = []
        node_correct: list[bool] = []
        node_zeros = 0
        for site_index, (truth_state, state_row) in enumerate(zip(sequence, state_rows), 1):
            if truth_state not in alphabet:
                raise ValueError(
                    f"truth state {truth_state!r} absent from {state_path.name} alphabet"
                )
            probability = float(state_row[f"p_{truth_state}"])
            map_state = str(state_row["State"])
            if map_state not in alphabet:
                # IQ-TREE 3.1.4 prints "-" for some 3Di sites under +F models;
                # where it prints a state, it is always the argmax.
                map_state = max(probability_columns, key=lambda column: float(state_row[column]))
                map_state = map_state.removeprefix("p_")
            map_probability = float(state_row[f"p_{map_state}"])
            printed_zero = probability == 0.0
            censored = max(probability, ROUNDING_FLOOR)
            log_probability = math.log(censored)
            correct = truth_state == map_state
            site_writer.writerow(
                {
                    "condition": condition,
                    "tau": format(tau, "g"),
                    "modality": modality,
                    "inferred_node": inferred_node,
                    "ground_truth_node": truth_node,
                    "site": site_index,
                    "ground_truth_state": truth_state,
                    "map_state": map_state,
                    "p_ground_truth": f"{probability:.5f}",
                    "p_map": f"{map_probability:.5f}",
                    "ground_truth_is_map": str(correct).lower(),
                    "printed_zero": str(printed_zero).lower(),
                    "ln_p_ground_truth_censored": f"{log_probability:.12g}",
                }
            )
            node_probabilities.append(probability)
            node_logs.append(log_probability)
            node_correct.append(correct)
            node_zeros += int(printed_zero)
        node_rows.append(
            {
                "condition": condition,
                "tau": tau,
                "modality": modality,
                "inferred_node": inferred_node,
                "ground_truth_node": truth_node,
                "sites": len(sequence),
                "mean_p_ground_truth": sum(node_probabilities) / len(node_probabilities),
                "map_accuracy": sum(node_correct) / len(node_correct),
                "printed_zero_sites": node_zeros,
                "log_probability_censored": sum(node_logs),
                "log10_probability_censored": sum(node_logs) / math.log(10),
                "mean_nll_censored": -sum(node_logs) / len(node_logs),
                "geometric_mean_p_censored": math.exp(sum(node_logs) / len(node_logs)),
            }
        )
        all_probabilities.extend(node_probabilities)
        all_log_probabilities.extend(node_logs)
        all_correct.extend(node_correct)
        zero_count += node_zeros
    if set(state_by_node) != {str(row["inferred_node"]) for row in mapping_rows}:
        raise ValueError(f"state/tree node mismatch in {state_path}")
    dataset_row = {
        "condition": condition,
        "tau": tau,
        "modality": modality,
        "status": "complete",
        "matched_nodes": len(matched),
        "unmatched_nodes": len(mapping_rows) - len(matched),
        "evaluated_sites": len(all_probabilities),
        "mean_p_ground_truth": sum(all_probabilities) / len(all_probabilities),
        "map_accuracy": sum(all_correct) / len(all_correct),
        "printed_zero_sites": zero_count,
        "mean_nll_censored": -sum(all_log_probabilities) / len(all_log_probabilities),
        "geometric_mean_p_censored": math.exp(
            sum(all_log_probabilities) / len(all_log_probabilities)
        ),
        "state_file": str(state_path.resolve()),
    }
    return node_rows, dataset_row


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"cannot infer columns for empty table {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--temperature-archive", type=Path, required=True)
    parser.add_argument("--true-tree", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--tau1-ground-truth-aa", type=Path)
    parser.add_argument("--ground-truth-3di-dir", type=Path)
    args = parser.parse_args()

    output = args.outdir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    true_tree = load_tree(args.true_tree)

    aa_truth: dict[float, OrderedDict[str, str]] = {}
    if args.tau1_ground_truth_aa is not None:
        aa_truth[1.0] = read_fasta(args.tau1_ground_truth_aa)
    with tarfile.open(args.temperature_archive, "r:gz") as bundle:
        for tau, label in ((1.5, "tau_1p5"), (2.0, "tau_2")):
            member = f"{label}/results/mu_1/simulation/ground_truth_nodes.fasta"
            aa_truth[tau] = parse_fasta_text(
                tar_text(bundle, member), f"{args.temperature_archive}:{member}"
            )

    ground_truth_dir = output / "ground_truth_inputs"
    for tau, records in aa_truth.items():
        write_fasta(
            ground_truth_dir / f"ground_truth_mu1_tau{format(tau, 'g').replace('.', 'p')}_aa.fasta",
            records,
        )

    three_di_truth: dict[float, OrderedDict[str, str]] = {}
    if args.ground_truth_3di_dir is not None:
        for tau, _label in CONDITIONS:
            path = args.ground_truth_3di_dir / (
                f"ground_truth_mu1_tau{format(tau, 'g').replace('.', 'p')}_3di.fasta"
            )
            if path.exists():
                three_di_truth[tau] = read_fasta(path)

    combined_pending: OrderedDict[str, str] = OrderedDict()
    for tau, records in aa_truth.items():
        if tau in three_di_truth:
            continue
        tau_label = format(tau, "g").replace(".", "p")
        for node, sequence in records.items():
            combined_pending[f"mu1_tau{tau_label}__{node}"] = sequence
    if combined_pending:
        write_fasta(ground_truth_dir / "pending_aa_to_3di.fasta", combined_pending)

    site_columns = [
        "condition",
        "tau",
        "modality",
        "inferred_node",
        "ground_truth_node",
        "site",
        "ground_truth_state",
        "map_state",
        "p_ground_truth",
        "p_map",
        "ground_truth_is_map",
        "printed_zero",
        "ln_p_ground_truth_censored",
    ]
    node_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    mapping_rows_all: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    site_path = output / "site_probabilities.tsv"
    with site_path.open("w", encoding="utf-8", newline="") as site_handle:
        site_writer = csv.DictWriter(site_handle, fieldnames=site_columns, delimiter="\t")
        site_writer.writeheader()
        for tau, label in CONDITIONS:
            condition = f"mu1_{label}"
            for modality in MODALITIES:
                state_path = args.state_dir / f"ef_tu_tips_mu1_{label}_{modality}.fasta.state"
                tree_path = args.state_dir / f"ef_tu_tips_mu1_{label}_{modality}.fasta.treefile"
                if not state_path.exists() or not tree_path.exists():
                    missing_rows.append(
                        {
                            "condition": condition,
                            "tau": tau,
                            "modality": modality,
                            "missing_input": "IQ-TREE state/treefile",
                            "required_path_or_action": str(state_path),
                        }
                    )
                    continue
                inferred_tree = Phylo.read(str(tree_path), "newick")
                mappings = topology_mapping(true_tree, inferred_tree)
                for row in mappings:
                    mapping_rows_all.append(
                        {"condition": condition, "tau": tau, "modality": modality, **row}
                    )
                truth = aa_truth.get(tau) if modality == "aa" else three_di_truth.get(tau)
                if truth is None:
                    missing_kind = (
                        "ground-truth AA internal-node FASTA"
                        if modality == "aa"
                        else "ground-truth 3Di internal-node FASTA"
                    )
                    missing_rows.append(
                        {
                            "condition": condition,
                            "tau": tau,
                            "modality": modality,
                            "missing_input": missing_kind,
                            "required_path_or_action": (
                                "recover/rerun tau=1 exact ESM2 simulation"
                                if modality == "aa"
                                else "encode all ground-truth AA nodes with ProstT5 <AA2fold>"
                            ),
                        }
                    )
                    dataset_rows.append(
                        {
                            "condition": condition,
                            "tau": tau,
                            "modality": modality,
                            "status": "missing_truth",
                            "matched_nodes": sum(bool(row["matched"]) for row in mappings),
                            "unmatched_nodes": sum(not bool(row["matched"]) for row in mappings),
                            "evaluated_sites": 0,
                            "mean_p_ground_truth": "",
                            "map_accuracy": "",
                            "printed_zero_sites": "",
                            "mean_nll_censored": "",
                            "geometric_mean_p_censored": "",
                            "state_file": str(state_path.resolve()),
                        }
                    )
                    continue
                expected_nodes = {str(clade.name) for clade in true_tree.find_clades()}
                if not expected_nodes.issubset(truth):
                    missing = sorted(expected_nodes - set(truth))
                    raise ValueError(f"truth FASTA for {condition} lacks nodes: {missing}")
                scored_nodes, dataset_row = score_dataset(
                    condition=condition,
                    tau=tau,
                    modality=modality,
                    state_path=state_path,
                    truth=truth,
                    mapping_rows=mappings,
                    site_writer=site_writer,
                )
                node_rows.extend(scored_nodes)
                dataset_rows.append(dataset_row)

    write_tsv(output / "node_probability_summary.tsv", node_rows)
    write_tsv(output / "dataset_probability_summary.tsv", dataset_rows)
    write_tsv(output / "node_mapping.tsv", mapping_rows_all)
    write_tsv(
        output / "missing_inputs.tsv",
        missing_rows,
        ["condition", "tau", "modality", "missing_input", "required_path_or_action"],
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "topology-mapped evaluation of ground-truth states under IQ-TREE site posteriors",
        "node_mapping": (
            "unroot the known degree-two root, then match each degree-three internal vertex "
            "by its three incident tip partitions"
        ),
        "probability_resolution": "IQ-TREE .state probabilities are printed to five decimals",
        "zero_policy": (
            "printed 0.00000 values are retained in p_ground_truth and replaced by 5e-6 only "
            "for censored log-probability/NLL summaries"
        ),
        "rounding_floor": ROUNDING_FLOOR,
        "root_policy": "37 unrooted internal nodes evaluated; the suppressed true root has no IQ-TREE state",
        "complete_datasets": [
            row["condition"] + ":" + row["modality"]
            for row in dataset_rows
            if row["status"] == "complete"
        ],
        "missing_datasets": [
            row["condition"] + ":" + row["modality"]
            for row in dataset_rows
            if row["status"] != "complete"
        ],
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
