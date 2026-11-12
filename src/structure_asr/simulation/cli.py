"""Command-line entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .config import build_composite_scorer, load_config, load_object, public_manifest, resolve_path
from .dca import DcaModel
from .evaluate import compare_reconstructions, score_sequences, write_comparison, write_scores
from .evolve import (
    DcaConditionalProposal,
    EvolutionResult,
    ForwardSimulator,
    MaskedSiteConditionalProposal,
    MutationEvent,
    ProfileConditionalProposal,
    SimulationSettings,
    UniformSubstitutionProposal,
)
from .io import read_fasta, write_run
from .scoring import CachedProfilePredictor, PseudoPerplexityScorer, ScoreResult
from .tree import load_tree, tree_to_newick


def _checkpoint_fingerprint(config, tree, root_sequence: str) -> str:
    payload = json.dumps(public_manifest(config), sort_keys=True, separators=(",", ":"))
    material = payload + "\n" + tree_to_newick(tree) + "\n" + root_sequence
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _load_checkpoint(path: Path, fingerprint: str) -> tuple[int, EvolutionResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported checkpoint schema in {path}")
    if payload.get("fingerprint") != fingerprint:
        raise RuntimeError(f"checkpoint does not match this config/tree/root: {path}")
    result = EvolutionResult(
        sequences={str(key): str(value) for key, value in payload["sequences"].items()},
        scores={
            str(key): ScoreResult(
                energy=float(value["energy"]),
                metrics={str(k): float(v) for k, v in value.get("metrics", {}).items()},
            )
            for key, value in payload["scores"].items()
        },
        events=[MutationEvent(**event) for event in payload["events"]],
    )
    return int(payload["completed_edges"]), result


def _write_checkpoint(
    path: Path,
    fingerprint: str,
    completed_edges: int,
    total_edges: int,
    result: EvolutionResult,
) -> None:
    payload = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "completed_edges": completed_edges,
        "total_edges": total_edges,
        "sequences": result.sequences,
        "scores": {
            node: {"energy": score.energy, "metrics": dict(score.metrics)}
            for node, score in result.scores.items()
        },
        "events": [asdict(event) for event in result.events],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def simulate(config_path: str) -> None:
    config, base = load_config(config_path)
    simulation = config["simulation"]
    tree_path = resolve_path(base, simulation["tree"])
    root_path = resolve_path(base, simulation["root_fasta"])
    output_dir = resolve_path(base, simulation["output_dir"])
    tree = load_tree(tree_path)
    root_records = read_fasta(root_path)
    root_id = simulation.get("root_id")
    if root_id is None:
        if len(root_records) != 1:
            raise ValueError("root_fasta has multiple records; set simulation.root_id")
        root_sequence = next(iter(root_records.values()))
    else:
        root_sequence = root_records[str(root_id)]

    dca_model = None
    if "dca" in config:
        dca_model = DcaModel.load(resolve_path(base, config["dca"]["model_npz"]))
    scorer = build_composite_scorer(config.get("scorers", [{"kind": "dca"}]), dca_model, base)
    proposal_name = str(simulation.get("proposal", "dca" if dca_model is not None else "uniform"))
    if proposal_name == "dca":
        if dca_model is None:
            raise ValueError("DCA proposal requires [dca].model_npz")
        proposal = DcaConditionalProposal(dca_model)
    elif proposal_name == "uniform":
        alphabet = tuple(simulation.get("alphabet", "ACDEFGHIKLMNPQRSTVWY"))
        proposal = UniformSubstitutionProposal(alphabet)
    elif proposal_name == "ofs":
        profile_scorers = [
            component
            for _weight, component in scorer.scorers
            if isinstance(component, PseudoPerplexityScorer)
        ]
        if len(profile_scorers) != 1:
            raise ValueError(
                "OFS proposal requires exactly one Python scorer factory to return "
                "PseudoPerplexityScorer"
            )
        profile_scorer = profile_scorers[0]
        cached_predictor = CachedProfilePredictor(profile_scorer.profile_predictor, max_size=2)
        profile_scorer.profile_predictor = cached_predictor
        proposal = ProfileConditionalProposal(
            profile_predictor=cached_predictor,
            alphabet=tuple(profile_scorer.alphabet),
            proposal_temperature=float(simulation.get("proposal_temperature", 1.0)),
        )
    elif proposal_name == "exact_mlm":
        provider_config = config.get("exact_mlm")
        if not provider_config or "factory" not in provider_config:
            raise ValueError("exact_mlm proposal requires [exact_mlm].factory")
        parameters = dict(provider_config.get("parameters", {}))
        for key in list(parameters):
            if key.endswith("_path"):
                parameters[key] = str(resolve_path(base, str(parameters[key])))
        provider = load_object(str(provider_config["factory"]))(parameters)
        if not callable(provider):
            if not hasattr(provider, "predict_site_profile"):
                raise TypeError(
                    f"{provider_config['factory']} must return a callable or an object "
                    "with predict_site_profile"
                )
            provider = provider.predict_site_profile
        proposal = MaskedSiteConditionalProposal(
            site_profile_predictor=provider,
            alphabet=tuple(simulation.get("alphabet", "ACDEFGHIKLMNPQRSTVWY")),
            proposal_temperature=float(simulation.get("proposal_temperature", 1.0)),
        )
    else:
        raise ValueError(
            "simulation.proposal must be 'dca', 'ofs', 'exact_mlm', or 'uniform'"
        )

    settings = SimulationSettings(
        mutation_scale=float(simulation.get("mutation_scale", 1.0)),
        temperature=float(simulation.get("temperature", 1.0)),
        seed=int(simulation.get("seed", 1)),
        kernel=str(simulation.get("kernel", "metropolis")),
    )
    checkpoint_path = None
    initial_result = None
    start_edge = 0
    edge_completed = None
    if simulation.get("checkpoint_path"):
        checkpoint_path = resolve_path(base, str(simulation["checkpoint_path"]))
        checkpoint_every = int(simulation.get("checkpoint_every_branches", 1))
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every_branches must be positive")
        fingerprint = _checkpoint_fingerprint(config, tree, root_sequence)
        if checkpoint_path.exists():
            start_edge, initial_result = _load_checkpoint(checkpoint_path, fingerprint)
            print(
                f"Resuming {checkpoint_path} after {start_edge} completed branches",
                flush=True,
            )

        def save_completed_edge(completed_edges, total_edges, partial_result):
            if completed_edges % checkpoint_every == 0 or completed_edges == total_edges:
                _write_checkpoint(
                    checkpoint_path,
                    fingerprint,
                    completed_edges,
                    total_edges,
                    partial_result,
                )

        edge_completed = save_completed_edge

    result = ForwardSimulator(scorer, proposal, settings).simulate(
        tree,
        root_sequence,
        initial_result=initial_result,
        start_edge=start_edge,
        edge_completed=edge_completed,
    )
    write_run(
        output_dir,
        tree,
        result,
        manifest={
            "schema_version": 1,
            "config": public_manifest(config),
            "source_config": str(Path(config_path).resolve()),
            "node_count": len(result.sequences),
            "event_count": len(result.events),
        },
    )
    print(f"Wrote benchmark replicate to {output_dir}")


def evaluate(truth: str, reconstruction: str, output: str) -> None:
    rows = compare_reconstructions(truth, reconstruction)
    write_comparison(output, rows)
    valid = [row for row in rows if row["status"] == "ok"]
    mean_distance = (
        sum(float(row["normalized_hamming"]) for row in valid) / len(valid)
        if valid
        else float("nan")
    )
    print(f"Compared {len(valid)} nodes; mean normalized Hamming distance={mean_distance:.6g}")


def score(config_path: str, fasta: str, output: str) -> None:
    config, base = load_config(config_path)
    dca_model = None
    if "dca" in config:
        dca_model = DcaModel.load(resolve_path(base, config["dca"]["model_npz"]))
    scorer = build_composite_scorer(config.get("scorers", [{"kind": "dca"}]), dca_model, base)
    rows = score_sequences(fasta, scorer)
    write_scores(output, rows)
    print(f"Scored {len(rows)} sequences; wrote {Path(output).resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forward-asr")
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate_parser = subparsers.add_parser("simulate", help="run a forward-evolution replicate")
    simulate_parser.add_argument("--config", required=True)
    evaluate_parser = subparsers.add_parser("evaluate", help="compare ASR FASTA with known node sequences")
    evaluate_parser.add_argument("--truth", required=True)
    evaluate_parser.add_argument("--reconstruction", required=True)
    evaluate_parser.add_argument("--output", required=True)
    score_parser = subparsers.add_parser("score", help="score any FASTA with DCA/LM/AlphaFold plug-ins")
    score_parser.add_argument("--config", required=True)
    score_parser.add_argument("--fasta", required=True)
    score_parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "simulate":
        simulate(args.config)
    elif args.command == "evaluate":
        evaluate(args.truth, args.reconstruction, args.output)
    else:
        score(args.config, args.fasta, args.output)


if __name__ == "__main__":
    main()
