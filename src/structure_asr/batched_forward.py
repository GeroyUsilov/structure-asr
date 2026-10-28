"""Batched exact masked-LM forward evolution across many independent cells.

``ForwardSimulator`` spawns one RNG stream per preorder edge, so an edge's
trajectory depends only on its parent sequence, its own stream and the model's
site profiles. This module advances many such edges ("lanes"), drawn from many
simulation cells, together and evaluates their masked sites in one model pass.
Each lane makes exactly the RNG calls of ``MaskedSiteConditionalProposal`` in
the same order. With a deterministic predictor the output is identical to the
serial simulator; on a GPU, batched floating-point reductions can differ from
batch-1 calls (see METHODS.md).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from forward_asr_benchmark.evolve import (
    EvolutionResult,
    ForwardSimulator,
    MaskedSiteConditionalProposal,
    MutationEvent,
)
from forward_asr_benchmark.scoring import CompositeScorer, ConstantScorer, ScoreResult
from forward_asr_benchmark.tree import tree_to_newick

BatchPredictor = Callable[[list[str], list[int]], np.ndarray]
CHECKPOINT_SCHEMA = 2


def _unused_predictor(_sequence: str, _site: int) -> np.ndarray:
    raise RuntimeError("batched lanes supply profiles explicitly")


def historical_scorer() -> CompositeScorer:
    """The bookkeeping scorer written by ``structure_asr.forward`` configs."""
    return CompositeScorer([(1.0, ConstantScorer())])


@dataclass
class Cell:
    """One forward simulation: tree, root, rate, generator temperature and seed."""

    label: str
    tree: object
    root_sequence: str
    mutation_scale: float
    proposal_temperature: float
    seed: int
    configuration: dict = field(default_factory=dict)
    scorer: object = field(default_factory=historical_scorer)

    def __post_init__(self) -> None:
        if self.mutation_scale < 0:
            raise ValueError("mutation_scale must be nonnegative")
        self.proposal = MaskedSiteConditionalProposal(
            _unused_predictor, proposal_temperature=self.proposal_temperature
        )
        self.edges = [
            (parent, child)
            for parent in self.tree.find_clades(order="preorder")
            for child in parent.clades
        ]
        self.edge_seeds = np.random.SeedSequence(self.seed).spawn(len(self.edges))
        self.parent_edge = {
            str(child.name): index for index, (_parent, child) in enumerate(self.edges)
        }
        self.downstream = _downstream_branch_lengths(self.tree)

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "configuration": self.configuration,
                "mutation_scale": self.mutation_scale,
                "proposal_temperature": self.proposal_temperature,
                "seed": self.seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        material = payload + "\n" + tree_to_newick(self.tree) + "\n" + self.root_sequence
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _downstream_branch_lengths(tree) -> dict[str, float]:
    """Longest branch-length path from each node to a tip (scheduling priority)."""
    result: dict[str, float] = {}
    for clade in tree.find_clades(order="postorder"):
        result[str(clade.name)] = max(
            (float(child.branch_length or 0.0) + result[str(child.name)] for child in clade.clades),
            default=0.0,
        )
    return result


@dataclass
class CellProgress:
    sequences: dict[str, str]
    scores: dict[str, ScoreResult]
    events_by_edge: dict[int, list[MutationEvent]]

    @property
    def completed_edges(self) -> set[int]:
        return set(self.events_by_edge)


@dataclass
class _Lane:
    cell: int
    edge: int
    rng: np.random.Generator
    sequence: str
    score: ScoreResult
    attempts: int
    events: list[MutationEvent] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.attempts - len(self.events)


def new_progress(cell: Cell) -> CellProgress:
    root = str(cell.tree.root.name)
    return CellProgress(
        sequences={root: cell.root_sequence},
        scores={root: cell.scorer.score(cell.root_sequence)},
        events_by_edge={},
    )


def finished_result(cell: Cell, progress: CellProgress) -> EvolutionResult:
    """Order nodes and events exactly as the serial preorder simulator does."""
    if len(progress.events_by_edge) != len(cell.edges):
        raise RuntimeError(f"{cell.label}: simulation is incomplete")
    root = str(cell.tree.root.name)
    names = [root] + [str(child.name) for _parent, child in cell.edges]
    return EvolutionResult(
        sequences={name: progress.sequences[name] for name in names},
        scores={name: progress.scores[name] for name in names},
        events=[
            event
            for index in range(len(cell.edges))
            for event in progress.events_by_edge[index]
        ],
    )


def simulate_cells(
    cells: Sequence[Cell],
    predictor: BatchPredictor,
    *,
    max_batch: int = 64,
    progress: Sequence[CellProgress | None] | None = None,
    edge_completed: Callable[[int, CellProgress], None] | None = None,
    batch_completed: Callable[[int], None] | None = None,
) -> list[CellProgress]:
    """Advance every incomplete edge of every cell, batching model passes.

    ``progress`` holds resumed per-cell state (completed edges only; an
    interrupted edge restarts from its own fresh RNG stream, as in the serial
    checkpoint). ``edge_completed(cell_index, state)`` is called after each edge.
    """
    if max_batch < 1:
        raise ValueError("max_batch must be positive")
    states = [
        (progress[index] if progress is not None and progress[index] is not None else new_progress(cell))
        for index, cell in enumerate(cells)
    ]
    lanes: list[_Lane] = []
    started: list[set[int]] = [set(state.completed_edges) for state in states]

    def activate_ready() -> None:
        changed = True
        while changed:
            changed = False
            for cell_index, (cell, state) in enumerate(zip(cells, states)):
                for edge_index, (parent, child) in enumerate(cell.edges):
                    if edge_index in started[cell_index]:
                        continue
                    parent_name = str(parent.name)
                    if parent_name not in state.sequences:
                        continue
                    started[cell_index].add(edge_index)
                    sequence = state.sequences[parent_name]
                    rng = np.random.default_rng(cell.edge_seeds[edge_index])
                    attempts = ForwardSimulator._stochastic_round(
                        cell.mutation_scale * len(sequence) * float(child.branch_length or 0.0),
                        rng,
                    )
                    lane = _Lane(
                        cell_index, edge_index, rng, sequence, state.scores[parent_name], attempts
                    )
                    if attempts == 0:
                        finish(lane)
                        changed = True
                    else:
                        lanes.append(lane)

    def finish(lane: _Lane) -> None:
        cell, state = cells[lane.cell], states[lane.cell]
        child = str(cell.edges[lane.edge][1].name)
        state.sequences[child] = lane.sequence
        state.scores[child] = lane.score
        state.events_by_edge[lane.edge] = lane.events
        if edge_completed is not None:
            edge_completed(lane.cell, state)

    def priority(lane: _Lane) -> float:
        cell = cells[lane.cell]
        child = str(cell.edges[lane.edge][1].name)
        future = cell.mutation_scale * len(lane.sequence) * cell.downstream[child]
        return lane.remaining + future

    activate_ready()
    while lanes:
        lanes.sort(key=lambda lane: (-priority(lane), lane.cell, lane.edge))
        length = len(lanes[0].sequence)
        batch = [lane for lane in lanes if len(lane.sequence) == length][:max_batch]
        sites = [
            cells[lane.cell].proposal.choose_site(lane.sequence, lane.rng) for lane in batch
        ]
        profiles = np.asarray(predictor([lane.sequence for lane in batch], sites), dtype=float)
        if profiles.shape[0] != len(batch):
            raise RuntimeError("batched predictor returned the wrong number of profiles")
        if batch_completed is not None:
            batch_completed(len(batch))
        finished = []
        for lane, site, profile in zip(batch, sites, profiles):
            cell = cells[lane.cell]
            move = cell.proposal.complete(lane.sequence, site, profile, lane.rng)
            proposed_score = cell.scorer.score(move.sequence)
            parent, child = cell.edges[lane.edge]
            lane.events.append(
                MutationEvent(
                    parent=str(parent.name),
                    child=str(child.name),
                    attempt=len(lane.events) + 1,
                    site=move.site,
                    old_state=move.old_state,
                    new_state=move.new_state,
                    accepted=True,
                    log_acceptance=0.0,
                    energy_before=lane.score.energy,
                    energy_after=proposed_score.energy,
                )
            )
            lane.sequence = move.sequence
            lane.score = proposed_score
            if lane.remaining == 0:
                finished.append(lane)
        if finished:
            for lane in finished:
                lanes.remove(lane)
                finish(lane)
            activate_ready()
    return states


def write_checkpoint(path: Path, cell: Cell, state: CellProgress) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "fingerprint": cell.fingerprint(),
        "total_edges": len(cell.edges),
        "completed_edges": sorted(state.completed_edges),
        "sequences": state.sequences,
        "scores": {
            node: {"energy": score.energy, "metrics": dict(score.metrics)}
            for node, score in state.scores.items()
        },
        "events_by_edge": {
            str(index): [asdict(event) for event in events]
            for index, events in sorted(state.events_by_edge.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_checkpoint(path: Path, cell: Cell) -> CellProgress | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"unsupported batched checkpoint schema in {path}")
    if payload.get("fingerprint") != cell.fingerprint():
        raise RuntimeError(f"checkpoint does not match this cell's config/tree/root: {path}")
    return CellProgress(
        sequences={str(key): str(value) for key, value in payload["sequences"].items()},
        scores={
            str(key): ScoreResult(
                energy=float(value["energy"]),
                metrics={str(k): float(v) for k, v in value.get("metrics", {}).items()},
            )
            for key, value in payload["scores"].items()
        },
        events_by_edge={
            int(index): [MutationEvent(**event) for event in events]
            for index, events in payload["events_by_edge"].items()
        },
    )
