#!/usr/bin/env python3
"""Uncertainty-weighted joint AA/3Di Metropolis-Hastings sampling.

Sites are proposed from fixed weights derived from AA-posterior Gini
uncertainty.  A new residue is drawn from the AA posterior conditional on
being different from the current residue, eliminating no-op proposals.  The
asymmetric residue proposal is handled with an explicit Hastings correction.
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
    "site_uncertainty",
    "site_weight",
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


def uncertainty_site_weights(
    posterior: NodePosterior,
    uncertainty_mixture: float,
    power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Gini uncertainties and a full-support proposal distribution."""
    uncertainties = 1.0 - np.square(posterior.probabilities).sum(axis=1)
    focused = np.power(np.maximum(uncertainties, 0.0), power)
    if focused.sum() <= 0:
        focused = np.ones_like(focused)
    focused /= focused.sum()
    uniform = np.full_like(focused, 1.0 / len(focused))
    weights = (1.0 - uncertainty_mixture) * uniform + uncertainty_mixture * focused
    weights /= weights.sum()
    return uncertainties, weights


def propose_sequence(
    state: ChainState,
    posterior: NodePosterior,
    site_weights: np.ndarray,
) -> tuple[str, str, float, float, int]:
    sequence = list(state.current_seq)
    alphabet_index = {token: i for i, token in enumerate(posterior.alphabet)}
    site = int(state.rng.choice(len(sequence), p=site_weights))
    old = sequence[site]
    old_index = alphabet_index[old]
    row = posterior.probabilities[site]
    proposal = row.copy()
    proposal[old_index] = 0.0
    proposal /= proposal.sum()
    new_index = int(state.rng.choice(len(posterior.alphabet), p=proposal))
    new = posterior.alphabet[new_index]
    if new == old:
        raise RuntimeError("conditional residue proposal produced a no-op")
    sequence[site] = new

    # q(x->y) = w_i p_i(new)/(1-p_i(old)); w_i cancels in the
    # reverse/forward ratio but is included here for auditable trace values.
    log_weight = math.log(float(site_weights[site]))
    forward = (
        log_weight
        + float(posterior.log_probabilities[site, new_index])
        - math.log1p(-float(row[old_index]))
    )
    reverse = (
        log_weight
        + float(posterior.log_probabilities[site, old_index])
        - math.log1p(-float(row[new_index]))
    )
    return "".join(sequence), f"{site + 1}:{old}>{new}", forward, reverse, site


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


def write_site_proposals(
    path: Path,
    nodes: list[str],
    aa: OrderedDict[str, NodePosterior],
    site_proposals: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "node", "site", "aa_map_state", "aa_max_probability",
            "gini_uncertainty", "proposal_site_weight",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for node in nodes:
            posterior = aa[node]
            uncertainties, weights = site_proposals[node]
            for site, (row, uncertainty, weight) in enumerate(
                zip(posterior.probabilities, uncertainties, weights), start=1
            ):
                best = int(np.argmax(row))
                writer.writerow(
                    {
                        "node": node,
                        "site": site,
                        "aa_map_state": posterior.alphabet[best],
                        "aa_max_probability": float(row[best]),
                        "gini_uncertainty": float(uncertainty),
                        "proposal_site_weight": float(weight),
                    }
                )


def config_payload(args: argparse.Namespace, nodes: list[str]) -> dict[str, object]:
    return {
        "condition": args.condition,
        "nodes": nodes,
        "n_iter": args.n_iter,
        "chains_per_node": args.chains_per_node,
        "mcmc_temperature": args.mcmc_temperature,
        "n_muts": 1,
        "structural_lambda": args.structural_lambda,
        "site_uncertainty": "gini: 1-sum_a p_i(a)^2",
        "site_uncertainty_mixture": args.site_uncertainty_mixture,
        "site_weight_power": args.site_weight_power,
        "seed": args.seed,
        "start_mode": args.start_mode,
        "acceptance_mode": "metropolis_hastings",
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
    start_by_node: OrderedDict[str, ChainState] = OrderedDict()
    for state in ordered:
        start_by_node.setdefault(state.node, state)
        prior = best_by_node.get(state.node)
        if prior is None or state.best_log_p_joint > prior.best_log_p_joint:
            best_by_node[state.node] = state
    write_fasta(
        output / "aa_map_start_aa.fasta",
        {node: state.start_seq for node, state in start_by_node.items()},
    )
    write_fasta(
        output / "aa_map_start_3di.fasta",
        {node: state.start_struct for node, state in start_by_node.items()},
    )
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
            "method": "uncertainty-weighted joint AA/3Di Metropolis-Hastings",
            "objective": "log_P_AA(sequence) + structural_lambda*log_P_3Di(ProstT5(sequence))",
            "proposal": "fixed AA-Gini uncertainty site weights; non-self AA-posterior residue proposal",
            "prostt5_method": "deterministic greedy <AA2fold>",
            "resolved_model_path": resolved_model,
            "completed_steps": step,
            "configuration": config,
            "outputs": {
                "primary_aa": "joint_asr_best_aa.fasta",
                "primary_3di": "joint_asr_best_3di.fasta",
                "aa_map_start_aa": "aa_map_start_aa.fasta",
                "aa_map_start_3di": "aa_map_start_3di.fasta",
                "chain_summary": "chain_summary.tsv",
                "trace": "trace.tsv",
            },
        },
    )


class SamplerRun:
    """One sampler configuration's chains, checkpoint, trace and outputs.

    The step logic is shared by ``main`` (one configuration) and
    ``structure_asr.grid sample`` (many configurations whose ProstT5 calls are
    batched together). Each chain owns its RNG, so interleaving configurations
    does not change any chain's draws.
    """

    def __init__(
        self,
        *,
        output: Path,
        config: dict[str, object],
        nodes: list[str],
        aa: OrderedDict[str, NodePosterior],
        three_di: OrderedDict[str, NodePosterior],
        site_proposals: dict[str, tuple[np.ndarray, np.ndarray]],
        checkpoint_every: int,
    ) -> None:
        self.output = output
        self.config = config
        self.nodes = nodes
        self.aa = aa
        self.three_di = three_di
        self.site_proposals = site_proposals
        self.checkpoint_every = checkpoint_every
        self.n_iter = int(config["n_iter"])
        self.structural_lambda = float(config["structural_lambda"])
        self.mcmc_temperature = float(config["mcmc_temperature"])
        self.checkpoint_path = output / "checkpoint.json"
        self.trace_path = output / "trace.tsv"
        self.completed_step = 0
        self.states: list[ChainState] = []
        self.trace_buffer: list[dict[str, object]] = []
        self._starts: list[tuple[str, int, np.random.Generator, str]] = []
        self._proposals: list[dict[str, object]] = []
        self._changed_indices: list[int] = []

    @property
    def finished(self) -> bool:
        return self.completed_step >= self.n_iter

    def write_site_proposals(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        write_site_proposals(
            self.output / "site_proposal_weights.tsv", self.nodes, self.aa, self.site_proposals
        )

    def prepare(self, *, resume: bool) -> list[tuple[str, int, np.random.Generator, str]] | None:
        """Resume from a checkpoint (returns None) or return starts needing encoding."""
        if resume and self.checkpoint_path.exists():
            checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint["configuration"] != self.config:
                raise ValueError("checkpoint configuration does not match this run")
            self.completed_step = int(checkpoint["completed_step"])
            self.states = [chain_from_json(dict(item)) for item in checkpoint["chains"]]
            trim_trace(self.trace_path, self.completed_step)
            print(f"Resuming after MCMC step {self.completed_step:,}", flush=True)
            return None
        if self.checkpoint_path.exists() or self.trace_path.exists():
            raise RuntimeError("existing output requires --resume or a new output directory")
        seed = int(self.config["seed"])
        self._starts = []
        for node in self.nodes:
            node_number = node_sort_key(node)[1]
            for chain_index in range(1, int(self.config["chains_per_node"]) + 1):
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, node_number, chain_index])
                )
                start = (
                    aa_map_sequence(self.aa[node])
                    if self.config["start_mode"] == "aa_map"
                    else sample_sequence(self.aa[node], rng)
                )
                self._starts.append((node, chain_index, rng, start))
        return self._starts

    def start(self, start_structures: list[str]) -> None:
        for (node, chain_index, rng, start), structure in zip(self._starts, start_structures):
            lp_aa = sequence_log_probability(self.aa[node], start)
            lp_3di = sequence_log_probability(self.three_di[node], structure)
            lp_joint = lp_aa + self.structural_lambda * lp_3di
            self.states.append(
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
        self._starts = []

    def propose(self) -> list[str]:
        """Draw one proposal per chain; return the sequences needing encoding."""
        self._proposals = []
        self._changed_indices = []
        changed_sequences = []
        for index, state in enumerate(self.states):
            uncertainties, site_weights = self.site_proposals[state.node]
            sequence, mutations, log_q_forward, log_q_reverse, site = propose_sequence(
                state, self.aa[state.node], site_weights
            )
            self._proposals.append(
                {
                    "sequence": sequence,
                    "mutations": mutations,
                    "log_q_forward": log_q_forward,
                    "log_q_reverse": log_q_reverse,
                    "site": site,
                    "site_uncertainty": float(uncertainties[site]),
                    "site_weight": float(site_weights[site]),
                }
            )
            if sequence != state.current_seq:
                self._changed_indices.append(index)
                changed_sequences.append(sequence)
        return changed_sequences

    def accept(self, changed_structures: list[str]) -> None:
        step = self.completed_step + 1
        structure_by_index = dict(zip(self._changed_indices, changed_structures))
        for index, (state, proposal) in enumerate(zip(self.states, self._proposals)):
            candidate_seq = str(proposal["sequence"])
            candidate_struct = structure_by_index.get(index, state.current_struct)
            candidate_aa = sequence_log_probability(self.aa[state.node], candidate_seq)
            candidate_3di = sequence_log_probability(self.three_di[state.node], candidate_struct)
            candidate_joint = candidate_aa + self.structural_lambda * candidate_3di
            log_acceptance = (candidate_joint - state.current_log_p_joint) / self.mcmc_temperature
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
            self.trace_buffer.append(
                {
                    "step": step,
                    "node": state.node,
                    "chain": state.chain,
                    "accepted": accepted,
                    "mutations": proposal["mutations"],
                    "site_uncertainty": proposal["site_uncertainty"],
                    "site_weight": proposal["site_weight"],
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
        self.completed_step = step

    def checkpoint(self, resolved_model: str) -> None:
        step = self.completed_step
        if step % self.checkpoint_every == 0 or step == self.n_iter:
            append_trace(self.trace_path, self.trace_buffer)
            self.trace_buffer.clear()
            atomic_write_json(
                self.checkpoint_path,
                {
                    "configuration": self.config,
                    "completed_step": step,
                    "chains": [chain_to_json(state) for state in self.states],
                },
            )
            write_outputs(self.output, self.states, self.config, resolved_model, step)


def advance_runs(
    runs: list[SamplerRun],
    encode,
    resolved_model: str,
    *,
    resume: bool,
    desc: str,
) -> None:
    """Start or resume ``runs`` and step them in lockstep to completion.

    Each step makes one ``encode(sequences)`` call holding every unfinished
    run's changed proposals, in run order. Runs that finish earlier drop out.
    """

    def encode_split(groups: list[list[str]]) -> list[list[str]]:
        flat = [sequence for group in groups for sequence in group]
        structures = encode(flat) if flat else []
        if len(structures) != len(flat):
            raise RuntimeError("encoder returned a different number of structures")
        split, offset = [], 0
        for group in groups:
            split.append(structures[offset : offset + len(group)])
            offset += len(group)
        return split

    starting = []
    for run in runs:
        starts = run.prepare(resume=resume)
        if starts is not None:
            starting.append((run, [start for _node, _chain, _rng, start in starts]))
    if starting:
        print(f"Encoding {sum(len(group) for _run, group in starting)} starting sequences", flush=True)
        for (run, _group), structures in zip(
            starting, encode_split([group for _run, group in starting])
        ):
            run.start(structures)

    progress = tqdm(
        total=max(run.n_iter for run in runs),
        initial=min(run.completed_step for run in runs),
        desc=desc,
        unit="step",
    )
    while True:
        active = [run for run in runs if not run.finished]
        if not active:
            break
        for run, structures in zip(active, encode_split([run.propose() for run in active])):
            run.accept(structures)
            run.checkpoint(resolved_model)
        accepted = sum(state.accepted for run in active for state in run.states)
        proposed = sum(run.completed_step * len(run.states) for run in active)
        progress.update(1)
        progress.set_postfix(acceptance=f"{accepted / proposed:.3f}")
    progress.close()


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
    parser.add_argument("--structural-lambda", type=float, default=0.02)
    parser.add_argument("--site-uncertainty-mixture", type=float, default=0.9)
    parser.add_argument("--site-weight-power", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=731024)
    parser.add_argument("--start-mode", choices=["aa_map", "aa_sample"], default="aa_map")
    parser.add_argument("--probability-floor", type=float, default=5e-6)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.n_iter < 1 or args.chains_per_node < 1 or args.model_batch_size < 1:
        raise ValueError("iteration, chain, and batch counts must be positive")
    if args.mcmc_temperature <= 0 or args.checkpoint_every < 1:
        raise ValueError("temperature and checkpoint interval must be positive")
    if args.structural_lambda < 0 or not 0 < args.probability_floor < 1:
        raise ValueError("structural lambda or probability floor is outside its valid range")
    if not 0 <= args.site_uncertainty_mixture <= 1 or args.site_weight_power <= 0:
        raise ValueError("site-weight parameters are outside their valid ranges")

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
    site_proposals = {
        node: uncertainty_site_weights(
            aa[node], args.site_uncertainty_mixture, args.site_weight_power
        )
        for node in nodes
    }

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

    run = SamplerRun(
        output=args.outdir.resolve(),
        config=config_payload(args, nodes),
        nodes=nodes,
        aa=aa,
        three_di=three_di,
        site_proposals=site_proposals,
        checkpoint_every=args.checkpoint_every,
    )
    run.write_site_proposals()

    torch_module, tokenizer, model, device, resolved_model = load_prostt5(
        args.model, args.device, args.dtype
    )

    def encode(sequences: list[str]) -> list[str]:
        return encode_sequences(
            sequences,
            tokenizer,
            model,
            device,
            torch_module,
            args.model_batch_size,
            args.max_chunk_aa,
        )

    advance_runs(
        [run], encode, str(resolved_model), resume=args.resume, desc=f"Joint ASR {args.condition}"
    )
    print(f"Completed {args.condition}: {run.output}", flush=True)


if __name__ == "__main__":
    main()
