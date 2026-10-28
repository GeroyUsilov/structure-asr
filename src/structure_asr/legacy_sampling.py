#!/usr/bin/env python3
"""Batched joint AA/3Di MCMC for IQ-TREE ancestral posterior tables.

The default ``legacy`` acceptance rule reproduces the existing
``protein_mcmc_utils.run_forward_chain`` procedure.  ``hastings`` adds the
proposal-density correction required when AA-posterior proposals are treated
as a formal Metropolis-Hastings kernel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

from forward_asr_benchmark.io import write_fasta
from structure_asr.prostt5 import encode_batch, load_prostt5
from structure_asr.posteriors import (
    NodePosterior, node_sort_key, load_state, sequence_log_probability,
    aa_map_sequence, sample_sequence,
)


CANONICAL_STATES = frozenset("ACDEFGHIKLMNPQRSTVWY")
TRACE_COLUMNS = [
    "step",
    "node",
    "chain",
    "accepted",
    "mutations",
    "log_acceptance",
    "proposal_log_q_forward",
    "proposal_log_q_reverse",
    "candidate_log_p_aa",
    "candidate_log_p_3di",
    "candidate_log_p_joint",
    "current_log_p_aa",
    "current_log_p_3di",
    "current_log_p_joint",
    "best_log_p_joint",
]


@dataclass
class ChainState:
    node: str
    chain: int
    rng: np.random.Generator
    start_seq: str
    start_struct: str
    start_log_p_aa: float
    start_log_p_3di: float
    start_log_p_joint: float
    current_seq: str
    current_struct: str
    current_log_p_aa: float
    current_log_p_3di: float
    current_log_p_joint: float
    best_seq: str
    best_struct: str
    best_log_p_aa: float
    best_log_p_3di: float
    best_log_p_joint: float
    accepted: int = 0


def propose_sequence(
    state: ChainState,
    posterior: NodePosterior,
    n_muts: int,
) -> tuple[str, str, float, float]:
    sequence = list(state.current_seq)
    sites = state.rng.choice(len(sequence), size=n_muts, replace=False)
    forward = 0.0
    reverse = 0.0
    mutations: list[str] = []
    alphabet_index = {token: i for i, token in enumerate(posterior.alphabet)}
    for raw_site in sites:
        site = int(raw_site)
        old = sequence[site]
        new_index = int(
            state.rng.choice(len(posterior.alphabet), p=posterior.probabilities[site])
        )
        new = posterior.alphabet[new_index]
        forward += float(posterior.log_probabilities[site, new_index])
        reverse += float(posterior.log_probabilities[site, alphabet_index[old]])
        sequence[site] = new
        mutations.append(f"{site + 1}:{old}>{new}")
    return "".join(sequence), ";".join(mutations), forward, reverse


def encode_sequences(
    sequences: list[str],
    tokenizer,
    model,
    device,
    torch_module,
    model_batch_size: int,
    max_chunk_aa: int,
) -> list[str]:
    structures: list[str] = []
    for start in range(0, len(sequences), model_batch_size):
        structures.extend(
            encode_batch(
                sequences[start : start + model_batch_size],
                tokenizer,
                model,
                device,
                torch_module,
                max_chunk_aa,
            )
        )
    return structures


def chain_to_json(state: ChainState) -> dict[str, object]:
    return {
        "node": state.node,
        "chain": state.chain,
        "rng_state": state.rng.bit_generator.state,
        "start_seq": state.start_seq,
        "start_struct": state.start_struct,
        "start_log_p_aa": state.start_log_p_aa,
        "start_log_p_3di": state.start_log_p_3di,
        "start_log_p_joint": state.start_log_p_joint,
        "current_seq": state.current_seq,
        "current_struct": state.current_struct,
        "current_log_p_aa": state.current_log_p_aa,
        "current_log_p_3di": state.current_log_p_3di,
        "current_log_p_joint": state.current_log_p_joint,
        "best_seq": state.best_seq,
        "best_struct": state.best_struct,
        "best_log_p_aa": state.best_log_p_aa,
        "best_log_p_3di": state.best_log_p_3di,
        "best_log_p_joint": state.best_log_p_joint,
        "accepted": state.accepted,
    }


def chain_from_json(payload: dict[str, object]) -> ChainState:
    rng = np.random.default_rng()
    rng.bit_generator.state = payload.pop("rng_state")  # type: ignore[assignment]
    return ChainState(rng=rng, **payload)  # type: ignore[arg-type]


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_trace(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_COLUMNS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def trim_trace(path: Path, completed_step: int) -> None:
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if int(row["step"]) <= completed_step]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def config_payload(args: argparse.Namespace, nodes: list[str]) -> dict[str, object]:
    return {
        "condition": args.condition,
        "nodes": nodes,
        "n_iter": args.n_iter,
        "chains_per_node": args.chains_per_node,
        "mcmc_temperature": args.mcmc_temperature,
        "n_muts": args.n_muts,
        "alpha": args.alpha,
        "seed": args.seed,
        "start_mode": args.start_mode,
        "acceptance_mode": args.acceptance_mode,
        "probability_floor": args.probability_floor,
        "model": args.model,
        "max_chunk_aa": args.max_chunk_aa,
    }


def write_outputs(
    output: Path,
    states: list[ChainState],
    config: dict[str, object],
    resolved_model: str,
    step: int,
) -> None:
    ordered = sorted(states, key=lambda item: (node_sort_key(item.node), item.chain))
    chain_id = lambda state: f"{state.node}_chain{state.chain:02d}"
    write_fasta(output / "final_aa.fasta", {chain_id(s): s.current_seq for s in ordered})
    write_fasta(output / "final_3di.fasta", {chain_id(s): s.current_struct for s in ordered})
    write_fasta(output / "best_aa.fasta", {chain_id(s): s.best_seq for s in ordered})
    write_fasta(output / "best_3di.fasta", {chain_id(s): s.best_struct for s in ordered})

    best_by_node: OrderedDict[str, ChainState] = OrderedDict()
    for state in ordered:
        prior = best_by_node.get(state.node)
        if prior is None or state.best_log_p_joint > prior.best_log_p_joint:
            best_by_node[state.node] = state
    write_fasta(
        output / "joint_asr_best_aa.fasta",
        {node: state.best_seq for node, state in best_by_node.items()},
    )
    write_fasta(
        output / "joint_asr_best_3di.fasta",
        {node: state.best_struct for node, state in best_by_node.items()},
    )

    with (output / "chain_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        columns = [
            "node", "chain", "steps", "accepted", "acceptance_rate",
            "start_log_p_aa", "start_log_p_3di", "start_log_p_joint",
            "final_log_p_aa", "final_log_p_3di", "final_log_p_joint",
            "best_log_p_aa", "best_log_p_3di", "best_log_p_joint",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for state in ordered:
            writer.writerow(
                {
                    "node": state.node,
                    "chain": state.chain,
                    "steps": step,
                    "accepted": state.accepted,
                    "acceptance_rate": state.accepted / step if step else 0.0,
                    "start_log_p_aa": state.start_log_p_aa,
                    "start_log_p_3di": state.start_log_p_3di,
                    "start_log_p_joint": state.start_log_p_joint,
                    "final_log_p_aa": state.current_log_p_aa,
                    "final_log_p_3di": state.current_log_p_3di,
                    "final_log_p_joint": state.current_log_p_joint,
                    "best_log_p_aa": state.best_log_p_aa,
                    "best_log_p_3di": state.best_log_p_3di,
                    "best_log_p_joint": state.best_log_p_joint,
                }
            )
    atomic_write_json(
        output / "run_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if step == int(config["n_iter"]) else "partial",
            "method": "joint AA/3Di sequence-space MCMC",
            "objective": "alpha*log_P_AA(sequence) + (1-alpha)*log_P_3Di(ProstT5(sequence))",
            "prostt5_method": "deterministic greedy <AA2fold>",
            "resolved_model_path": resolved_model,
            "completed_steps": step,
            "configuration": config,
            "outputs": {
                "primary_aa": "joint_asr_best_aa.fasta",
                "primary_3di": "joint_asr_best_3di.fasta",
                "chain_summary": "chain_summary.tsv",
                "trace": "trace.tsv",
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aa-state", type=Path, required=True)
    parser.add_argument("--three-di-state", type=Path, required=True)
    parser.add_argument("--nodes-file", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--model", default="Rostlab/ProstT5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--model-batch-size", type=int, default=8)
    parser.add_argument("--max-chunk-aa", type=int, default=500)
    parser.add_argument("--n-iter", type=int, default=3000)
    parser.add_argument("--chains-per-node", type=int, default=1)
    parser.add_argument("--mcmc-temperature", type=float, default=2.0)
    parser.add_argument("--n-muts", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=731024)
    parser.add_argument("--start-mode", choices=["aa_map", "aa_sample"], default="aa_map")
    parser.add_argument("--acceptance-mode", choices=["legacy", "hastings"], default="legacy")
    parser.add_argument("--probability-floor", type=float, default=5e-6)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.n_iter < 1 or args.chains_per_node < 1 or args.model_batch_size < 1:
        raise ValueError("iteration, chain, and batch counts must be positive")
    if args.n_muts < 1 or args.mcmc_temperature <= 0 or args.checkpoint_every < 1:
        raise ValueError("mutation count, temperature, and checkpoint interval must be positive")
    if not 0 <= args.alpha <= 1 or not 0 < args.probability_floor < 1:
        raise ValueError("alpha or probability floor is outside its valid range")

    aa = load_state(args.aa_state, args.probability_floor)
    three_di = load_state(args.three_di_state, args.probability_floor)
    nodes = [line.strip() for line in args.nodes_file.read_text().splitlines() if line.strip()]
    if len(nodes) != len(set(nodes)) or not nodes:
        raise ValueError("nodes file is empty or contains duplicates")
    for node in nodes:
        if node not in aa or node not in three_di:
            raise ValueError(f"{node} is absent from an input state file")
        if aa[node].probabilities.shape[0] != three_di[node].probabilities.shape[0]:
            raise ValueError(f"AA and 3Di lengths differ for {node}")
        if args.n_muts > aa[node].probabilities.shape[0]:
            raise ValueError(f"n_muts exceeds sequence length for {node}")

    if args.validate_only:
        print(
            json.dumps(
                {
                    "condition": args.condition,
                    "nodes": nodes,
                    "sites_per_node": {
                        node: int(aa[node].probabilities.shape[0]) for node in nodes
                    },
                    "configuration": config_payload(args, nodes),
                    "status": "inputs_valid",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    output = args.outdir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    trace_path = output / "trace.tsv"
    config = config_payload(args, nodes)

    torch_module, tokenizer, model, device, resolved_model = load_prostt5(
        args.model, args.device, args.dtype
    )
    completed_step = 0
    states: list[ChainState]
    if args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["configuration"] != config:
            raise ValueError("checkpoint configuration does not match this run")
        completed_step = int(checkpoint["completed_step"])
        states = [chain_from_json(dict(item)) for item in checkpoint["chains"]]
        trim_trace(trace_path, completed_step)
        print(f"Resuming after MCMC step {completed_step:,}", flush=True)
    else:
        if checkpoint_path.exists() or trace_path.exists():
            raise RuntimeError("existing output requires --resume or a new output directory")
        states = []
        starts: list[str] = []
        metadata: list[tuple[str, int, np.random.Generator]] = []
        for node in nodes:
            node_number = node_sort_key(node)[1]
            for chain_index in range(1, args.chains_per_node + 1):
                rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, node_number, chain_index])
                )
                start = (
                    aa_map_sequence(aa[node])
                    if args.start_mode == "aa_map"
                    else sample_sequence(aa[node], rng)
                )
                starts.append(start)
                metadata.append((node, chain_index, rng))
        print(f"Encoding {len(starts)} starting sequences", flush=True)
        start_structures = encode_sequences(
            starts,
            tokenizer,
            model,
            device,
            torch_module,
            args.model_batch_size,
            args.max_chunk_aa,
        )
        for start, structure, (node, chain_index, rng) in zip(
            starts, start_structures, metadata
        ):
            lp_aa = sequence_log_probability(aa[node], start)
            lp_3di = sequence_log_probability(three_di[node], structure)
            lp_joint = args.alpha * lp_aa + (1 - args.alpha) * lp_3di
            states.append(
                ChainState(
                    node=node,
                    chain=chain_index,
                    rng=rng,
                    start_seq=start,
                    start_struct=structure,
                    start_log_p_aa=lp_aa,
                    start_log_p_3di=lp_3di,
                    start_log_p_joint=lp_joint,
                    current_seq=start,
                    current_struct=structure,
                    current_log_p_aa=lp_aa,
                    current_log_p_3di=lp_3di,
                    current_log_p_joint=lp_joint,
                    best_seq=start,
                    best_struct=structure,
                    best_log_p_aa=lp_aa,
                    best_log_p_3di=lp_3di,
                    best_log_p_joint=lp_joint,
                )
            )

    trace_buffer: list[dict[str, object]] = []
    progress = tqdm(
        range(completed_step + 1, args.n_iter + 1),
        total=args.n_iter,
        initial=completed_step,
        desc=f"Joint ASR {args.condition}",
        unit="step",
    )
    for step in progress:
        proposals = []
        changed_indices = []
        changed_sequences = []
        for index, state in enumerate(states):
            sequence, mutations, log_q_forward, log_q_reverse = propose_sequence(
                state, aa[state.node], args.n_muts
            )
            proposals.append(
                {
                    "sequence": sequence,
                    "mutations": mutations,
                    "log_q_forward": log_q_forward,
                    "log_q_reverse": log_q_reverse,
                }
            )
            if sequence != state.current_seq:
                changed_indices.append(index)
                changed_sequences.append(sequence)
        changed_structures = encode_sequences(
            changed_sequences,
            tokenizer,
            model,
            device,
            torch_module,
            args.model_batch_size,
            args.max_chunk_aa,
        ) if changed_sequences else []
        structure_by_index = dict(zip(changed_indices, changed_structures))

        for index, (state, proposal) in enumerate(zip(states, proposals)):
            candidate_seq = str(proposal["sequence"])
            candidate_struct = structure_by_index.get(index, state.current_struct)
            candidate_aa = sequence_log_probability(aa[state.node], candidate_seq)
            candidate_3di = sequence_log_probability(three_di[state.node], candidate_struct)
            candidate_joint = args.alpha * candidate_aa + (1 - args.alpha) * candidate_3di
            log_acceptance = (candidate_joint - state.current_log_p_joint) / args.mcmc_temperature
            if args.acceptance_mode == "hastings":
                log_acceptance += float(proposal["log_q_reverse"]) - float(proposal["log_q_forward"])
            log_acceptance = min(0.0, log_acceptance)
            accepted = bool(math.log(max(state.rng.random(), np.finfo(float).tiny)) < log_acceptance)
            if accepted:
                state.current_seq = candidate_seq
                state.current_struct = candidate_struct
                state.current_log_p_aa = candidate_aa
                state.current_log_p_3di = candidate_3di
                state.current_log_p_joint = candidate_joint
                state.accepted += 1
                if candidate_joint > state.best_log_p_joint:
                    state.best_seq = candidate_seq
                    state.best_struct = candidate_struct
                    state.best_log_p_aa = candidate_aa
                    state.best_log_p_3di = candidate_3di
                    state.best_log_p_joint = candidate_joint
            trace_buffer.append(
                {
                    "step": step,
                    "node": state.node,
                    "chain": state.chain,
                    "accepted": accepted,
                    "mutations": proposal["mutations"],
                    "log_acceptance": log_acceptance,
                    "proposal_log_q_forward": proposal["log_q_forward"],
                    "proposal_log_q_reverse": proposal["log_q_reverse"],
                    "candidate_log_p_aa": candidate_aa,
                    "candidate_log_p_3di": candidate_3di,
                    "candidate_log_p_joint": candidate_joint,
                    "current_log_p_aa": state.current_log_p_aa,
                    "current_log_p_3di": state.current_log_p_3di,
                    "current_log_p_joint": state.current_log_p_joint,
                    "best_log_p_joint": state.best_log_p_joint,
                }
            )

        if step % args.checkpoint_every == 0 or step == args.n_iter:
            append_trace(trace_path, trace_buffer)
            trace_buffer.clear()
            atomic_write_json(
                checkpoint_path,
                {
                    "configuration": config,
                    "completed_step": step,
                    "chains": [chain_to_json(state) for state in states],
                },
            )
            write_outputs(output, states, config, str(resolved_model), step)
        progress.set_postfix(
            acceptance=f"{sum(state.accepted for state in states)/(step*len(states)):.3f}"
        )
    progress.close()
    print(f"Completed {args.condition}: {output}", flush=True)


if __name__ == "__main__":
    main()
