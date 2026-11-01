#!/usr/bin/env python3
"""Multi-regime forward-simulation grid: plan, simulate, encode, reconstruct, evaluate.

One TOML file defines a grid of (mu, tau, seed) cells on a fixed tree and root.
Stages are separate commands so each runs where it belongs:

  plan      CPU   expand the grid and count exact model calls
  forward   GPU   batched masked-site ESM2 evolution across all cells
  encode    GPU   ProstT5 AA->3Di for every simulated node (tips are inference input)
  asr       CPU   IQ-TREE marginal ASR on the fixed topology (AA and 3Di)
  sample    GPU   joint AA/3Di MH sampling for every cell x structural lambda,
                  with all chains' ProstT5 calls batched together
  evaluate  CPU   post-inference topology-mapped comparison with truth
  status    any   per-cell stage completion

Inference stages never read ground truth; ``evaluate`` is the only reader.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
import tomllib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from Bio import Phylo

from forward_asr_benchmark.io import read_fasta, write_fasta, write_run
from forward_asr_benchmark.tree import load_tree
from structure_asr import iqtree, sampling
from structure_asr.batched_forward import (
    Cell,
    finished_result,
    load_checkpoint,
    simulate_cells,
    write_checkpoint,
)
from structure_asr.evaluation import read_fasta as read_ordered_fasta
from structure_asr.evaluation import score_dataset, topology_mapping, vertex_signatures, write_tsv
from structure_asr.forward import choose_root, exact_planned_attempts, rate_label, sha256
from structure_asr.posteriors import load_state, node_sort_key

THREE_DI_METHOD = "deterministic greedy <AA2fold> generation"
SITE_COLUMNS = [
    "condition", "tau", "modality", "inferred_node", "ground_truth_node", "site",
    "ground_truth_state", "map_state", "p_ground_truth", "p_map",
    "ground_truth_is_map", "printed_zero", "ln_p_ground_truth_censored",
]


@dataclass(frozen=True)
class CellSpec:
    index: int
    mutation_scale: float
    proposal_temperature: float
    seed: int

    @property
    def label(self) -> str:
        return (
            f"mu_{rate_label(self.mutation_scale)}__tau_{rate_label(self.proposal_temperature)}"
            f"__seed_{self.seed}"
        )


@dataclass(frozen=True)
class SampleSettings:
    """``[sample]``: one sampler run per (cell, structural lambda).

    Defaults match ``python -m structure_asr.sampling``. ``nodes`` names AA-tree
    (``asr/aa.treefile``) internal nodes, or is ``"all"``; the 3Di posterior for
    each is found by topology. ``seed`` defaults to the cell seed, so the runs of
    one cell share starts and RNG streams and differ only in lambda.
    """

    structural_lambdas: tuple[float, ...]
    nodes: tuple[str, ...] | None
    n_iter: int = 3000
    chains_per_node: int = 1
    mcmc_temperature: float = 2.0
    site_uncertainty_mixture: float = 0.9
    site_weight_power: float = 1.0
    seed: int | None = None
    start_mode: str = "aa_map"
    probability_floor: float = 5e-6
    checkpoint_every: int = 10
    max_chunk_aa: int = 500

    @classmethod
    def from_toml(cls, section: dict) -> "SampleSettings":
        section = dict(section)
        unknown = set(section) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown [sample] keys: {sorted(unknown)}")
        if "structural_lambdas" not in section or "nodes" not in section:
            raise ValueError("[sample] requires structural_lambdas and nodes")
        lambdas = tuple(float(value) for value in section.pop("structural_lambdas"))
        nodes = section.pop("nodes")
        if nodes == "all":
            nodes = None
        elif isinstance(nodes, list) and nodes:
            nodes = tuple(str(node) for node in nodes)
            if len(set(nodes)) != len(nodes):
                raise ValueError("duplicate [sample].nodes")
        else:
            raise ValueError('[sample].nodes must be a non-empty list or "all"')
        if not lambdas or any(value < 0 for value in lambdas):
            raise ValueError("[sample].structural_lambdas must be non-empty and >= 0")
        if len({rate_label(value) for value in lambdas}) != len(lambdas):
            raise ValueError("duplicate [sample].structural_lambdas")
        settings = cls(structural_lambdas=lambdas, nodes=nodes, **section)
        if settings.n_iter < 1 or settings.chains_per_node < 1 or settings.checkpoint_every < 1:
            raise ValueError("[sample] iteration, chain and checkpoint counts must be positive")
        if settings.mcmc_temperature <= 0 or not 0 < settings.probability_floor < 1:
            raise ValueError("[sample] temperature or probability floor is outside its valid range")
        if not 0 <= settings.site_uncertainty_mixture <= 1 or settings.site_weight_power <= 0:
            raise ValueError("[sample] site-weight parameters are outside their valid ranges")
        if settings.start_mode not in ("aa_map", "aa_sample"):
            raise ValueError("[sample].start_mode must be aa_map or aa_sample")
        return settings


@dataclass
class Grid:
    name: str
    config_path: Path
    outdir: Path
    tree_path: Path
    alignment_path: Path
    root_accession: str
    model_name: str
    dtype: str
    max_attempts_per_cell: int
    cells: list[CellSpec]
    asr: dict
    sample: SampleSettings | None = field(default=None)

    def __post_init__(self) -> None:
        self.tree = load_tree(self.tree_path)
        self.root_record, self.root_sequence = choose_root(self.alignment_path, self.root_accession)

    def cell_dir(self, spec: CellSpec) -> Path:
        return self.outdir / spec.label

    def shard(self, shard: str | None) -> list[CellSpec]:
        if shard is None:
            return list(self.cells)
        index, count = (int(part) for part in shard.split("/"))
        if count < 1 or not 0 <= index < count:
            raise ValueError("--shard must be i/n with 0 <= i < n")
        return [spec for spec in self.cells if spec.index % count == index]

    def configuration(self, spec: CellSpec) -> dict:
        """Everything that defines a cell's simulated output (checkpoint identity)."""
        return {
            "kernel": "exact_mlm_gibbs",
            "proposal": "exact_mlm",
            "model_name": self.model_name,
            "dtype": self.dtype,
            "tree_sha256": sha256(self.tree_path),
            "root_accession": self.root_accession,
            "alphabet": "ACDEFGHIKLMNPQRSTVWY",
        }

    def make_cell(self, spec: CellSpec) -> Cell:
        return Cell(
            label=spec.label,
            tree=load_tree(self.tree_path),
            root_sequence=self.root_sequence,
            mutation_scale=spec.mutation_scale,
            proposal_temperature=spec.proposal_temperature,
            seed=spec.seed,
            configuration=self.configuration(spec),
        )

    def planned_attempts(self, spec: CellSpec) -> int:
        return exact_planned_attempts(
            self.tree, len(self.root_sequence), spec.mutation_scale, spec.seed
        )


def load_grid(path: Path, outdir: Path | None = None) -> Grid:
    config_path = path.resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    base = config_path.parent
    section = config["grid"]

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (base / candidate).resolve()

    rates = [float(value) for value in section["rates"]]
    temperatures = [float(value) for value in section["proposal_temperatures"]]
    seeds = [int(value) for value in section["seeds"]]
    if any(value <= 0 for value in rates + temperatures):
        raise ValueError("rates and proposal temperatures must be positive")
    for values, name in ((rates, "rates"), (temperatures, "proposal_temperatures"), (seeds, "seeds")):
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate values in grid.{name}")
    specs = [
        CellSpec(index, rate, temperature, seed)
        for index, (rate, temperature, seed) in enumerate(
            (rate, temperature, seed)
            for rate in rates
            for temperature in temperatures
            for seed in seeds
        )
    ]
    asr = dict(config.get("asr", {}))
    if "three_di_matrix" in asr:
        asr["three_di_matrix"] = resolve(str(asr["three_di_matrix"]))
    name = str(section["name"])
    return Grid(
        name=name,
        config_path=config_path,
        outdir=(outdir or resolve(str(section.get("outdir", f"../results/grids/{name}")))).resolve(),
        tree_path=resolve(str(section["tree"])),
        alignment_path=resolve(str(section["alignment"])),
        root_accession=str(section.get("root_accession", "163840912")),
        model_name=str(section.get("model_name", "facebook/esm2_t33_650M_UR50D")),
        dtype=str(section.get("dtype", "float16")),
        max_attempts_per_cell=int(section.get("max_attempts_per_cell", 100_000)),
        cells=specs,
        asr=asr,
        sample=SampleSettings.from_toml(config["sample"]) if "sample" in config else None,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


# --------------------------------------------------------------------- plan


def git_revision() -> str:
    """Commit of this checkout, suffixed ``-dirty`` for uncommitted changes; "" if unknown."""
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return commit + ("-dirty" if dirty else "")


def plan(grid: Grid) -> dict:
    length = len(grid.root_sequence)
    deepest = max(
        sum(float(clade.branch_length or 0.0) for clade in grid.tree.get_path(tip))
        for tip in grid.tree.get_terminals()
    )
    cells = []
    for spec in grid.cells:
        attempts = grid.planned_attempts(spec)
        if attempts > grid.max_attempts_per_cell:
            raise RuntimeError(
                f"{spec.label} plans {attempts:,} model calls, exceeding "
                f"max_attempts_per_cell={grid.max_attempts_per_cell:,}"
            )
        cells.append(
            {
                "label": spec.label,
                "mutation_scale": spec.mutation_scale,
                "proposal_temperature": spec.proposal_temperature,
                "seed": spec.seed,
                "planned_attempts": attempts,
                "approx_sequential_steps": math.ceil(spec.mutation_scale * length * deepest),
            }
        )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": grid.name,
        "config": str(grid.config_path),
        "git_commit": git_revision(),
        "method": "exact random-site single-mask ESM2 Gibbs generator, batched across edges and cells",
        "model_name": grid.model_name,
        "dtype": grid.dtype,
        "tree": {"path": str(grid.tree_path), "sha256": sha256(grid.tree_path)},
        "root": {"record": grid.root_record, "length": length},
        "cells": cells,
        "total_planned_attempts": sum(cell["planned_attempts"] for cell in cells),
        "note": (
            "approx_sequential_steps is mu*L*(deepest root-to-tip branch length): a lower "
            "bound on batched passes for that cell regardless of GPU batch size."
        ),
    }
    if grid.sample is not None:
        settings = grid.sample
        if settings.nodes is not None:
            node_count = len(settings.nodes)
        else:  # IQ-TREE reports the unrooted tree: a degree-two root is not a node.
            node_count = len(grid.tree.get_nonterminals()) - (len(grid.tree.root.clades) == 2)
        chains = len(grid.cells) * len(settings.structural_lambdas) * node_count * settings.chains_per_node
        payload["sample"] = {
            "structural_lambdas": list(settings.structural_lambdas),
            "nodes_per_cell": node_count,
            "chains_per_node": settings.chains_per_node,
            "total_chains": chains,
            "total_chain_steps": chains * settings.n_iter,
            "sequential_prostt5_steps": settings.n_iter,
            "note": (
                "every chain step needs one ProstT5 encoding; all chains of a process share "
                "each step's batch, but the n_iter steps themselves are sequential."
            ),
        }
    write_json(grid.outdir / "grid_plan.json", payload)
    return payload


# ------------------------------------------------------------------ forward


def forward_complete(grid: Grid, spec: CellSpec) -> bool:
    return (grid.cell_dir(spec) / "simulation" / "manifest.json").exists()


def run_forward(
    grid: Grid,
    predictor,
    *,
    specs: list[CellSpec],
    max_batch: int,
    on_edge=None,
) -> list[str]:
    """Simulate incomplete cells; returns labels written. ``predictor`` is batched."""
    pending = []
    for spec in specs:
        if forward_complete(grid, spec):
            continue
        if (grid.cell_dir(spec) / "simulation").exists():
            raise RuntimeError(
                f"partial output directory blocks restart: {grid.cell_dir(spec) / 'simulation'}; "
                "inspect it and move it aside"
            )
        attempts = grid.planned_attempts(spec)
        if attempts > grid.max_attempts_per_cell:
            raise RuntimeError(f"{spec.label} exceeds max_attempts_per_cell")
        pending.append(spec)
    if not pending:
        return []
    cells = [grid.make_cell(spec) for spec in pending]
    checkpoints = [grid.cell_dir(spec) / "checkpoint.json" for spec in pending]
    progress = [load_checkpoint(path, cell) for path, cell in zip(checkpoints, cells)]
    written: list[str] = []

    def edge_completed(index: int, state) -> None:
        cell, spec = cells[index], pending[index]
        write_checkpoint(checkpoints[index], cell, state)
        if len(state.completed_edges) == len(cell.edges):
            write_cell_outputs(grid, spec, cell, state, max_batch)
            written.append(spec.label)
        if on_edge is not None:
            on_edge(index, state)

    for index, state in enumerate(progress):
        if state is not None and len(state.completed_edges) == len(cells[index].edges):
            write_cell_outputs(grid, pending[index], cells[index], state, max_batch)
            written.append(pending[index].label)
    remaining = [
        index for index, state in enumerate(progress)
        if state is None or len(state.completed_edges) < len(cells[index].edges)
    ]
    if remaining:
        index_map = {position: index for position, index in enumerate(remaining)}
        simulate_cells(
            [cells[index] for index in remaining],
            predictor,
            max_batch=max_batch,
            progress=[progress[index] for index in remaining],
            edge_completed=lambda position, state: edge_completed(index_map[position], state),
            batch_completed=getattr(predictor, "batch_completed", None),
        )
    return written


def write_cell_outputs(grid: Grid, spec: CellSpec, cell: Cell, state, max_batch: int) -> None:
    result = finished_result(cell, state)
    planned = grid.planned_attempts(spec)
    if len(result.events) != planned:
        raise RuntimeError(f"{spec.label} produced {len(result.events)} events; planned {planned}")
    run_root = grid.cell_dir(spec)
    root_path = run_root / "root_sequence.fasta"
    write_fasta(root_path, {f"root_{grid.root_accession}": grid.root_sequence})
    write_run(
        run_root / "simulation",
        cell.tree,
        result,
        manifest={
            "schema_version": 1,
            "config": {
                **cell.configuration,
                "mutation_scale": spec.mutation_scale,
                "proposal_temperature": spec.proposal_temperature,
                "seed": spec.seed,
                "tree": str(grid.tree_path),
                "root_fasta": str(root_path),
                "grid": grid.name,
            },
            "execution": {
                "mode": "batched multi-lane (structure_asr.batched_forward)",
                "max_batch": max_batch,
                "note": "batch composition may change GPU numerics relative to batch-1 runs",
            },
            "node_count": len(result.sequences),
            "event_count": len(result.events),
            "realized_residue_changes": sum(
                event.old_state != event.new_state for event in result.events
            ),
        },
    )


class TimedPredictor:
    """Wrap a batched predictor with progress and a per-batch timing log."""

    def __init__(self, predictor, timings: Path, total: int, initial: int):
        from tqdm.auto import tqdm

        self.predictor = predictor
        self.timings = timings
        if not timings.exists():
            timings.parent.mkdir(parents=True, exist_ok=True)
            timings.write_text("batch_size\tforward_seconds\n", encoding="utf-8")
        self.progress = tqdm(total=total, initial=initial, desc="ESM2 grid", unit="call")

    def __call__(self, sequences, sites):
        profiles = self.predictor(sequences, sites)
        seconds = float(getattr(self.predictor, "last_forward_seconds", float("nan")))
        with self.timings.open("a", encoding="utf-8") as handle:
            handle.write(f"{len(sequences)}\t{seconds:.9f}\n")
        self.progress.update(len(sequences))
        return profiles


# ------------------------------------------------------------------- encode


def encode_complete(grid: Grid, spec: CellSpec) -> bool:
    return (grid.cell_dir(spec) / "three_di" / "manifest.json").exists()


def run_encode(
    grid: Grid,
    encoder,
    *,
    specs: list[CellSpec],
    shard_name: str,
    batch_size: int,
    model_label: str,
    model_info: dict,
    resume: bool,
) -> list[str]:
    """Encode every node of forward-complete cells; tips feed ASR, internals are truth.

    ``encoder(identifiers, sequences, checkpoint, resume)`` returns 3Di strings and
    may record ``model_info["resolved"]`` (the loaded snapshot path).
    """
    pending = [spec for spec in specs if not encode_complete(grid, spec)]
    if not pending:
        return []
    missing = [spec.label for spec in pending if not forward_complete(grid, spec)]
    if missing:
        raise RuntimeError(f"forward simulation incomplete for: {', '.join(missing)}")
    identifiers, sequences = [], []
    for spec in pending:
        nodes = read_fasta(grid.cell_dir(spec) / "simulation" / "ground_truth_nodes.fasta")
        for name, sequence in nodes.items():
            identifiers.append(f"{spec.label}::{name}")
            sequences.append(sequence)
    work = grid.outdir / "three_di_work"
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / f"encoding_{shard_name}.jsonl"
    structures = dict(zip(identifiers, encoder(identifiers, sequences, checkpoint, resume)))
    for spec in pending:
        run_root = grid.cell_dir(spec)
        nodes = read_fasta(run_root / "simulation" / "ground_truth_nodes.fasta")
        tips = read_fasta(run_root / "simulation" / "leaves.fasta")
        encoded = {name: structures[f"{spec.label}::{name}"] for name in nodes}
        for name, sequence in nodes.items():
            if len(encoded[name]) != len(sequence):
                raise RuntimeError(f"{spec.label} {name}: 3Di length differs from AA length")
        (run_root / "truth").mkdir(parents=True, exist_ok=True)
        write_fasta(run_root / "truth" / "ground_truth_nodes_3di.fasta", encoded)
        (run_root / "three_di").mkdir(parents=True, exist_ok=True)
        tip_fasta = run_root / "three_di" / "leaves_3di.fasta"
        write_fasta(tip_fasta, {name: encoded[name] for name in tips})
        write_json(
            run_root / "three_di" / "manifest.json",
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "method": THREE_DI_METHOD,
                "model": model_label,
                "resolved_model_path": model_info.get("resolved", ""),
                "do_sample": False,
                "num_beams": 1,
                "batch_size": batch_size,
                "aa_nodes_sha256": sha256(run_root / "simulation" / "ground_truth_nodes.fasta"),
                "leaves_3di_sha256": sha256(tip_fasta),
                "truth_3di": "../truth/ground_truth_nodes_3di.fasta (evaluation only)",
                "checkpoint": str(checkpoint),
            },
        )
    return [spec.label for spec in pending]


# ---------------------------------------------------------------------- asr


def asr_jobs(
    *,
    aa_tips: Path,
    three_di_tips: Path | None,
    topology: Path,
    outdir: Path,
    seed: int,
    asr: dict,
) -> list[iqtree.AsrJob]:
    threads = int(asr.get("threads_per_job", 1))
    jobs = [
        iqtree.AsrJob(
            modality="aa",
            alignment=aa_tips,
            topology=topology,
            prefix=outdir / "aa",
            seed=seed,
            threads=threads,
            model=str(asr.get("aa_model", "MFP")),
        )
    ]
    if three_di_tips is not None:
        matrix = asr.get("three_di_matrix")
        if matrix is None:
            raise ValueError("[asr].three_di_matrix is required for 3Di reconstruction")
        matrix = Path(matrix)
        if not matrix.exists():
            raise FileNotFoundError(f"{matrix} not found; run scripts/fetch_3di_matrix.sh")
        name = str(asr.get("three_di_model_name", iqtree.DEFAULT_3DI_MODEL_NAME))
        definition = iqtree.write_matrix_definition(matrix, outdir / "three_di_model.nex", name)
        jobs.append(
            iqtree.AsrJob(
                modality="3di",
                alignment=three_di_tips,
                topology=topology,
                prefix=outdir / "3di",
                seed=seed,
                threads=threads,
                model="MFP",
                model_definition=definition,
                model_set=name,
            )
        )
    return jobs


def asr_complete(grid: Grid, spec: CellSpec) -> bool:
    asr_dir = grid.cell_dir(spec) / "asr"
    return all((asr_dir / f"{modality}.asr_manifest.json").exists() for modality in ("aa", "3di"))


def run_jobs(jobs: list[iqtree.AsrJob], executable: str, workers: int) -> list[dict]:
    version = iqtree.iqtree_version(executable)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(iqtree.run_job, job, executable, version) for job in jobs]
        results = []
        for job, future in zip(jobs, futures):
            manifest = future.result()
            print(f"{job.prefix}: {manifest['best_fit_model']}", flush=True)
            results.append(manifest)
    return results


def run_asr(grid: Grid, *, specs: list[CellSpec], executable: str, workers: int) -> list[dict]:
    topology = iqtree.write_topology(grid.tree, grid.outdir / "asr_topology.nwk")
    jobs = []
    for spec in specs:
        run_root = grid.cell_dir(spec)
        if not encode_complete(grid, spec):
            raise RuntimeError(f"3Di encoding incomplete for {spec.label}")
        jobs.extend(
            asr_jobs(
                aa_tips=run_root / "simulation" / "leaves.fasta",
                three_di_tips=run_root / "three_di" / "leaves_3di.fasta",
                topology=topology,
                outdir=run_root / "asr",
                seed=spec.seed,
                asr=grid.asr,
            )
        )
    return run_jobs(jobs, executable, workers)


# ------------------------------------------------------------------- sample


def sample_dir(grid: Grid, spec: CellSpec, structural_lambda: float) -> Path:
    return grid.cell_dir(spec) / "sample" / f"lambda_{rate_label(structural_lambda)}"


def sample_run_complete(path: Path) -> bool:
    manifest = path / "run_manifest.json"
    return manifest.exists() and json.loads(manifest.read_text())["status"] == "complete"


def sample_complete(grid: Grid, spec: CellSpec) -> bool:
    return grid.sample is not None and all(
        sample_run_complete(sample_dir(grid, spec, value))
        for value in grid.sample.structural_lambdas
    )


def three_di_node_map(asr_dir: Path) -> dict[str, str]:
    """AA-tree internal node -> 3Di-tree internal node with the same tip partition."""
    aa = vertex_signatures(Phylo.read(str(asr_dir / "aa.treefile"), "newick"), suppress_root=False)
    three_di = vertex_signatures(
        Phylo.read(str(asr_dir / "3di.treefile"), "newick"), suppress_root=False
    )
    return {aa[signature]: three_di[signature] for signature in aa if signature in three_di}


def prepare_sample(grid: Grid, *, specs: list[CellSpec], model: str) -> list[sampling.SamplerRun]:
    """Validate inputs and build the incomplete sampler runs of ``specs`` (no model load).

    Reads only ASR outputs; ground truth is never opened.
    """
    settings = grid.sample
    if settings is None:
        raise ValueError(f"{grid.config_path} has no [sample] section")
    runs = []
    for spec in specs:
        pending = [
            value for value in settings.structural_lambdas
            if not sample_run_complete(sample_dir(grid, spec, value))
        ]
        if not pending:
            continue
        if not asr_complete(grid, spec):
            raise RuntimeError(f"ASR incomplete for {spec.label}")
        asr_dir = grid.cell_dir(spec) / "asr"
        aa = load_state(asr_dir / "aa.state", settings.probability_floor)
        three_di_states = load_state(asr_dir / "3di.state", settings.probability_floor)
        node_map = three_di_node_map(asr_dir)
        nodes = list(settings.nodes) if settings.nodes else sorted(aa, key=node_sort_key)
        three_di = OrderedDict()
        for node in nodes:
            if node not in aa:
                raise ValueError(f"{spec.label}: {node} is absent from asr/aa.state")
            if node not in node_map or node_map[node] not in three_di_states:
                raise ValueError(f"{spec.label}: no 3Di posterior shares {node}'s topology")
            three_di[node] = three_di_states[node_map[node]]
            if aa[node].probabilities.shape[0] != three_di[node].probabilities.shape[0]:
                raise ValueError(f"{spec.label}: AA and 3Di lengths differ for {node}")
        site_proposals = {
            node: sampling.uncertainty_site_weights(
                aa[node], settings.site_uncertainty_mixture, settings.site_weight_power
            )
            for node in nodes
        }
        provenance = {
            "grid": grid.name,
            "cell": spec.label,
            "aa_state_sha256": sha256(asr_dir / "aa.state"),
            "three_di_state_sha256": sha256(asr_dir / "3di.state"),
            "three_di_nodes": {node: node_map[node] for node in nodes},
        }
        for value in pending:
            arguments = argparse.Namespace(
                condition=f"{spec.label}__lambda_{rate_label(value)}",
                n_iter=settings.n_iter,
                chains_per_node=settings.chains_per_node,
                mcmc_temperature=settings.mcmc_temperature,
                structural_lambda=value,
                site_uncertainty_mixture=settings.site_uncertainty_mixture,
                site_weight_power=settings.site_weight_power,
                seed=spec.seed if settings.seed is None else settings.seed,
                start_mode=settings.start_mode,
                probability_floor=settings.probability_floor,
                model=model,
                max_chunk_aa=settings.max_chunk_aa,
            )
            runs.append(
                sampling.SamplerRun(
                    output=sample_dir(grid, spec, value),
                    config={**sampling.config_payload(arguments, nodes), **provenance},
                    nodes=nodes,
                    aa=aa,
                    three_di=three_di,
                    site_proposals=site_proposals,
                    checkpoint_every=settings.checkpoint_every,
                )
            )
    return runs


class SharedEncoder:
    """Encode each distinct sequence once per step and log per-step timing.

    Runs of one cell share starts and RNG streams, so their proposals coincide
    until their accept/reject decisions first differ.
    """

    def __init__(self, encode, timings: Path):
        self.encode = encode
        self.timings = timings
        if not timings.exists():
            timings.parent.mkdir(parents=True, exist_ok=True)
            timings.write_text("sequences\tunique_sequences\tencode_seconds\n", encoding="utf-8")

    def __call__(self, sequences: list[str]) -> list[str]:
        unique = list(dict.fromkeys(sequences))
        started = time.perf_counter()
        structures = dict(zip(unique, self.encode(unique)))
        seconds = time.perf_counter() - started
        with self.timings.open("a", encoding="utf-8") as handle:
            handle.write(f"{len(sequences)}\t{len(unique)}\t{seconds:.6f}\n")
        return [structures[sequence] for sequence in sequences]


def run_sample(
    grid: Grid,
    encode,
    *,
    specs: list[CellSpec],
    model: str,
    resolved_model: str,
    shard_name: str,
    resume: bool,
) -> list[str]:
    """Advance every incomplete (cell, lambda) run together; returns their directories."""
    runs = prepare_sample(grid, specs=specs, model=model)
    if not runs:
        return []
    for run in runs:
        run.write_site_proposals()
    encoder = SharedEncoder(encode, grid.outdir / f"sample_timings_{shard_name}.tsv")
    sampling.advance_runs(runs, encoder, resolved_model, resume=resume, desc="Joint ASR grid")
    return [str(run.output) for run in runs]


# ----------------------------------------------------------------- evaluate


def evaluate_reconstruction(
    *,
    condition: str,
    tau: float,
    asr_dir: Path,
    true_tree,
    truth_paths: dict[str, Path],
    site_writer,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Topology-mapped truth comparison for one reconstruction directory."""
    dataset_rows, node_rows, mapping_rows = [], [], []
    for modality, truth_path in truth_paths.items():
        inferred = Phylo.read(str(asr_dir / f"{modality}.treefile"), "newick")
        mapping = topology_mapping(true_tree, inferred)
        nodes, dataset = score_dataset(
            condition=condition,
            tau=tau,
            modality=modality,
            state_path=asr_dir / f"{modality}.state",
            truth=read_ordered_fasta(truth_path),
            mapping_rows=mapping,
            site_writer=site_writer,
        )
        manifest_file = asr_dir / f"{modality}.asr_manifest.json"
        if manifest_file.exists():
            dataset["best_fit_model"] = json.loads(manifest_file.read_text())["best_fit_model"]
        dataset_rows.append(dataset)
        node_rows.extend(nodes)
        mapping_rows.extend({"condition": condition, "modality": modality, **row} for row in mapping)
    return dataset_rows, node_rows, mapping_rows


def mismatches(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("sequence lengths differ")
    return sum(a != b for a, b in zip(left, right))


def evaluate_samples(grid: Grid, spec: CellSpec) -> list[dict]:
    """Per (lambda, node): errors of the AA-MAP start and the best-scoring visited state.

    Runs are found on disk; partial runs are reported with their status.
    """
    run_root = grid.cell_dir(spec)
    runs = sorted((run_root / "sample").glob("lambda_*/run_manifest.json"))
    if not runs:
        return []
    mapping = {
        row["inferred_node"]: row["ground_truth_node"]
        for row in topology_mapping(
            load_tree(grid.tree_path), Phylo.read(str(run_root / "asr" / "aa.treefile"), "newick")
        )
    }
    truth_aa = read_ordered_fasta(run_root / "simulation" / "ground_truth_nodes.fasta")
    truth_3di = read_ordered_fasta(run_root / "truth" / "ground_truth_nodes_3di.fasta")
    rows = []
    for manifest_path in runs:
        manifest = json.loads(manifest_path.read_text())
        directory = manifest_path.parent
        config = manifest["configuration"]
        start_aa = read_ordered_fasta(directory / "aa_map_start_aa.fasta")
        start_3di = read_ordered_fasta(directory / "aa_map_start_3di.fasta")
        best_aa = read_ordered_fasta(directory / "joint_asr_best_aa.fasta")
        best_3di = read_ordered_fasta(directory / "joint_asr_best_3di.fasta")
        with (directory / "chain_summary.tsv").open(encoding="utf-8") as handle:
            chains = list(csv.DictReader(handle, delimiter="\t"))
        for node in config["nodes"]:
            truth = mapping.get(node, "")
            node_chains = [row for row in chains if row["node"] == node]
            row = {
                "condition": spec.label,
                "mutation_scale": spec.mutation_scale,
                "tau": spec.proposal_temperature,
                "seed": spec.seed,
                "structural_lambda": config["structural_lambda"],
                "status": manifest["status"],
                "completed_steps": manifest["completed_steps"],
                "inferred_node": node,
                "ground_truth_node": truth,
                "sites": len(best_aa[node]),
                "mean_acceptance_rate": sum(
                    float(chain["acceptance_rate"]) for chain in node_chains
                ) / len(node_chains),
            }
            if truth:
                row.update(
                    start_aa_errors=mismatches(start_aa[node], truth_aa[truth]),
                    best_aa_errors=mismatches(best_aa[node], truth_aa[truth]),
                    start_3di_errors=mismatches(start_3di[node], truth_3di[truth]),
                    best_3di_errors=mismatches(best_3di[node], truth_3di[truth]),
                )
                row["delta_aa_errors"] = row["best_aa_errors"] - row["start_aa_errors"]
            rows.append(row)
    return rows


SAMPLE_COLUMNS = [
    "condition", "mutation_scale", "tau", "seed", "structural_lambda", "status",
    "completed_steps", "inferred_node", "ground_truth_node", "sites", "mean_acceptance_rate",
    "start_aa_errors", "best_aa_errors", "delta_aa_errors", "start_3di_errors", "best_3di_errors",
]


def run_evaluate(grid: Grid, *, specs: list[CellSpec], outdir: Path) -> list[dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_rows, node_rows, mapping_rows, sample_rows = [], [], [], []
    with (outdir / "site_probabilities.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SITE_COLUMNS, delimiter="\t")
        writer.writeheader()
        for spec in specs:
            if not asr_complete(grid, spec):
                print(f"Skipping {spec.label}: ASR incomplete", flush=True)
                continue
            run_root = grid.cell_dir(spec)
            datasets, nodes, mappings = evaluate_reconstruction(
                condition=spec.label,
                tau=spec.proposal_temperature,
                asr_dir=run_root / "asr",
                true_tree=load_tree(grid.tree_path),
                truth_paths={
                    "aa": run_root / "simulation" / "ground_truth_nodes.fasta",
                    "3di": run_root / "truth" / "ground_truth_nodes_3di.fasta",
                },
                site_writer=writer,
            )
            for row in datasets:
                row.update(mutation_scale=spec.mutation_scale, seed=spec.seed)
            dataset_rows.extend(datasets)
            node_rows.extend(nodes)
            mapping_rows.extend(mappings)
            sample_rows.extend(evaluate_samples(grid, spec))
    if dataset_rows:
        write_tsv(outdir / "dataset_summary.tsv", dataset_rows)
        write_tsv(outdir / "node_summary.tsv", node_rows)
        write_tsv(outdir / "node_mapping.tsv", mapping_rows)
    if sample_rows:
        write_tsv(outdir / "sample_summary.tsv", sample_rows, SAMPLE_COLUMNS)
    return dataset_rows


# ----------------------------------------------------------------- commands


def default_iqtree(value: str | None, grid_asr: dict | None = None) -> str:
    return value or os.environ.get("ASR_IQTREE") or str((grid_asr or {}).get("iqtree", "iqtree3"))


def command_forward(args, grid: Grid) -> None:
    from structure_asr.esm2 import make_batched_site_predictor

    specs = grid.shard(args.shard)
    plan(grid)
    total = sum(grid.planned_attempts(spec) for spec in specs if not forward_complete(grid, spec))
    done = 0
    for spec in specs:
        checkpoint = grid.cell_dir(spec) / "checkpoint.json"
        if not forward_complete(grid, spec) and checkpoint.exists():
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            done += sum(len(events) for events in payload["events_by_edge"].values())
    if args.validate_only:
        print(f"{len(specs)} cells in shard; {total - done:,} ESM2 calls remain", flush=True)
        return
    predictor = TimedPredictor(
        make_batched_site_predictor(grid.model_name, args.device, grid.dtype),
        grid.outdir / f"batch_timings_{(args.shard or '0/1').replace('/', '_of_')}.tsv",
        total,
        done,
    )
    written = run_forward(grid, predictor, specs=specs, max_batch=args.max_batch)
    predictor.progress.close()
    print(f"Forward simulation complete; wrote {len(written)} cells", flush=True)


def command_encode(args, grid: Grid) -> None:
    from structure_asr.prostt5 import encode_with_checkpoint, load_prostt5

    specs = grid.shard(args.shard)
    if args.validate_only:
        missing = [spec.label for spec in specs if not forward_complete(grid, spec)]
        print(f"{len(specs)} cells; forward incomplete: {missing or 'none'}", flush=True)
        return
    loaded = {}

    def encoder(identifiers, sequences, checkpoint, resume):
        torch_module, tokenizer, model, device, resolved = load_prostt5(
            args.model, args.device, args.dtype
        )
        loaded["resolved"] = str(resolved)
        return encode_with_checkpoint(
            identifiers, sequences, checkpoint, tokenizer, model, device, torch_module,
            args.batch_size, args.max_chunk_aa, resume,
        )

    written = run_encode(
        grid,
        encoder,
        specs=specs,
        shard_name=(args.shard or "0/1").replace("/", "_of_"),
        batch_size=args.batch_size,
        model_label=args.model,
        model_info=loaded,
        resume=args.resume,
    )
    print(f"3Di encoding complete; wrote {len(written)} cells", flush=True)


def command_sample(args, grid: Grid) -> None:
    from structure_asr.prostt5 import load_prostt5

    specs = grid.shard(args.shard)
    runs = prepare_sample(grid, specs=specs, model=args.model)
    if args.validate_only:
        chains = sum(len(run.nodes) * int(run.config["chains_per_node"]) for run in runs)
        print(f"{len(specs)} cells in shard; {len(runs)} incomplete (cell, lambda) runs; "
              f"{chains} chains; inputs valid", flush=True)
        return
    if not runs:
        print("Sampling already complete for this shard", flush=True)
        return
    torch_module, tokenizer, model, device, resolved = load_prostt5(
        args.model, args.device, args.dtype
    )

    def encode(sequences: list[str]) -> list[str]:
        return sampling.encode_sequences(
            sequences, tokenizer, model, device, torch_module,
            args.model_batch_size, grid.sample.max_chunk_aa,
        )

    written = run_sample(
        grid, encode, specs=specs, model=args.model, resolved_model=str(resolved),
        shard_name=(args.shard or "0/1").replace("/", "_of_"), resume=args.resume,
    )
    print(f"Sampling complete; wrote {len(written)} runs", flush=True)


def command_status(_args, grid: Grid) -> None:
    stages = ["forward", "encode", "asr"] + (["sample"] if grid.sample is not None else [])
    print("cell\t" + "\t".join(stages))
    for spec in grid.cells:
        flags = [forward_complete(grid, spec), encode_complete(grid, spec), asr_complete(grid, spec)]
        if grid.sample is not None:
            flags.append(sample_complete(grid, spec))
        print(spec.label + "\t" + "\t".join("yes" if flag else "-" for flag in flags))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asr-grid", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def grid_command(name: str, help_text: str):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("grid", type=Path, help="grid TOML file")
        command.add_argument("--outdir", type=Path, help="override [grid].outdir")
        command.add_argument("--shard", help="process cells with index %% n == i, as i/n")
        return command

    grid_command("plan", "expand the grid and count model calls (CPU)")
    forward = grid_command("forward", "batched ESM2 forward simulation (GPU)")
    forward.add_argument("--max-batch", type=int, default=64)
    forward.add_argument("--device", default="cuda")
    forward.add_argument("--validate-only", action="store_true")
    encode = grid_command("encode", "ProstT5 AA->3Di for all simulated nodes (GPU)")
    encode.add_argument("--model", default="Rostlab/ProstT5")
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    encode.add_argument("--batch-size", type=int, default=32)
    encode.add_argument("--max-chunk-aa", type=int, default=500)
    encode.add_argument("--resume", action="store_true")
    encode.add_argument("--validate-only", action="store_true")
    asr = grid_command("asr", "IQ-TREE marginal ASR on the fixed topology (CPU)")
    asr.add_argument("--iqtree", help="IQ-TREE executable (default: $ASR_IQTREE or [asr].iqtree)")
    asr.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    sample = grid_command("sample", "joint AA/3Di MH sampling per cell and lambda (GPU)")
    sample.add_argument("--model", default="Rostlab/ProstT5")
    sample.add_argument("--device", default="cuda")
    sample.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    sample.add_argument("--model-batch-size", type=int, default=32)
    sample.add_argument("--resume", action="store_true")
    sample.add_argument("--validate-only", action="store_true")
    evaluate = grid_command("evaluate", "post-inference comparison with truth (CPU)")
    evaluate.add_argument("--evaluation-dir", type=Path)
    grid_command("status", "per-cell stage completion")

    reconstruct = sub.add_parser("reconstruct", help="IQ-TREE ASR for one existing tip dataset")
    reconstruct.add_argument("--aa-tips", type=Path, required=True)
    reconstruct.add_argument("--three-di-tips", type=Path)
    reconstruct.add_argument("--tree", type=Path, required=True, help="known topology source")
    reconstruct.add_argument("--three-di-matrix", type=Path)
    reconstruct.add_argument("--outdir", type=Path, required=True)
    reconstruct.add_argument("--seed", type=int, default=731024)
    reconstruct.add_argument("--threads-per-job", type=int, default=1)
    reconstruct.add_argument("--iqtree")

    evaluate_dir = sub.add_parser("evaluate-dir", help="evaluate one reconstruction directory")
    evaluate_dir.add_argument("--asr-dir", type=Path, required=True)
    evaluate_dir.add_argument("--tree", type=Path, required=True)
    evaluate_dir.add_argument("--truth-aa", type=Path, required=True)
    evaluate_dir.add_argument("--truth-3di", type=Path)
    evaluate_dir.add_argument("--condition", required=True)
    evaluate_dir.add_argument("--tau", type=float, default=float("nan"))
    evaluate_dir.add_argument("--outdir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "reconstruct":
        tree = load_tree(args.tree)
        topology = iqtree.write_topology(tree, args.outdir / "asr_topology.nwk")
        jobs = asr_jobs(
            aa_tips=args.aa_tips.resolve(),
            three_di_tips=args.three_di_tips.resolve() if args.three_di_tips else None,
            topology=topology.resolve(),
            outdir=args.outdir.resolve(),
            seed=args.seed,
            asr={
                "threads_per_job": args.threads_per_job,
                "three_di_matrix": args.three_di_matrix.resolve() if args.three_di_matrix else None,
            },
        )
        run_jobs(jobs, default_iqtree(args.iqtree), len(jobs))
        return
    if args.command == "evaluate-dir":
        args.outdir.mkdir(parents=True, exist_ok=True)
        truth_paths = {"aa": args.truth_aa}
        if args.truth_3di is not None:
            truth_paths["3di"] = args.truth_3di
        with (args.outdir / "site_probabilities.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SITE_COLUMNS, delimiter="\t")
            writer.writeheader()
            datasets, nodes, mappings = evaluate_reconstruction(
                condition=args.condition, tau=args.tau, asr_dir=args.asr_dir,
                true_tree=load_tree(args.tree), truth_paths=truth_paths, site_writer=writer,
            )
        write_tsv(args.outdir / "dataset_summary.tsv", datasets)
        write_tsv(args.outdir / "node_summary.tsv", nodes)
        write_tsv(args.outdir / "node_mapping.tsv", mappings)
        for row in datasets:
            print(
                f"{row['modality']}: matched={row['matched_nodes']} "
                f"MAP accuracy={row['map_accuracy']:.4%} "
                f"mean p(truth)={row['mean_p_ground_truth']:.5f} "
                f"model={row.get('best_fit_model', '?')}",
                flush=True,
            )
        return

    grid = load_grid(args.grid, args.outdir)
    if args.command == "plan":
        payload = plan(grid)
        for cell in payload["cells"]:
            print(f"{cell['label']}\t{cell['planned_attempts']:,} calls\t"
                  f">= {cell['approx_sequential_steps']:,} sequential passes")
        print(f"{len(payload['cells'])} cells, {payload['total_planned_attempts']:,} ESM2 calls; "
              f"wrote {grid.outdir / 'grid_plan.json'}")
        if "sample" in payload:
            sample = payload["sample"]
            print(f"sample: {sample['total_chains']:,} chains, {sample['total_chain_steps']:,} "
                  f"ProstT5 chain-steps in {sample['sequential_prostt5_steps']:,} sequential steps")
    elif args.command == "forward":
        command_forward(args, grid)
    elif args.command == "encode":
        command_encode(args, grid)
    elif args.command == "asr":
        run_asr(grid, specs=grid.shard(args.shard),
                executable=default_iqtree(args.iqtree, grid.asr), workers=args.jobs)
    elif args.command == "sample":
        command_sample(args, grid)
    elif args.command == "evaluate":
        rows = run_evaluate(grid, specs=grid.shard(args.shard),
                            outdir=args.evaluation_dir or grid.outdir / "evaluation")
        print(f"Evaluated {len(rows)} cell/modality reconstructions", flush=True)
    else:
        command_status(args, grid)


if __name__ == "__main__":
    main()
