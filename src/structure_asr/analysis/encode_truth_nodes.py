#!/usr/bin/env python3
"""Encode complete simulated node FASTAs with the established ProstT5 path."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from structure_asr.simulation.io import read_fasta, write_fasta
from structure_asr.prostt5 import (
    MODEL_ID,
    encode_with_checkpoint,
    load_prostt5,
)


def parse_condition_input(value: str) -> tuple[str, Path]:
    try:
        condition, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use CONDITION=/path/to/truth.fasta") from error
    if condition not in {"mu1_tau1", "mu1_tau1p5", "mu1_tau2"}:
        raise argparse.ArgumentTypeError(f"unsupported condition: {condition}")
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"input FASTA does not exist: {path}")
    return condition, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition-input",
        action="append",
        type=parse_condition_input,
        required=True,
        help="repeat as CONDITION=/path/to/ground_truth_nodes.fasta",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-chunk-aa", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    inputs = OrderedDict(args.condition_input)
    if len(inputs) != len(args.condition_input):
        raise ValueError("each condition may be supplied only once")
    if args.batch_size < 1 or args.max_chunk_aa < 1:
        raise ValueError("batch size and chunk length must be positive")

    output = args.outdir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    combined: OrderedDict[str, str] = OrderedDict()
    records_by_condition: dict[str, OrderedDict[str, str]] = {}
    for condition, fasta_path in inputs.items():
        records = read_fasta(fasta_path)
        if len(records) != 77:
            raise ValueError(f"{condition} has {len(records)} nodes instead of 77")
        if any(len(sequence) != 396 for sequence in records.values()):
            raise ValueError(f"{condition} contains a non-396-residue sequence")
        records_by_condition[condition] = records
        for identifier, sequence in records.items():
            combined[f"{condition}__{identifier}"] = sequence

    write_fasta(output / "combined_ground_truth_aa.fasta", combined)
    torch_module, tokenizer, model, device, resolved_model = load_prostt5(
        args.model, args.device, args.dtype
    )
    structures = encode_with_checkpoint(
        list(combined),
        list(combined.values()),
        output / "ground_truth_3di_checkpoint.jsonl",
        tokenizer,
        model,
        device,
        torch_module,
        args.batch_size,
        args.max_chunk_aa,
        args.resume,
    )
    combined_3di = dict(zip(combined, structures))
    write_fasta(output / "combined_ground_truth_3di.fasta", combined_3di)

    written = []
    for condition, records in records_by_condition.items():
        output_fasta = output / f"ground_truth_{condition}_3di.fasta"
        write_fasta(
            output_fasta,
            {
                identifier: combined_3di[f"{condition}__{identifier}"]
                for identifier in records
            },
        )
        written.append(str(output_fasta))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "full ProstT5 seq2seq deterministic greedy <AA2fold> generation",
        "model": args.model,
        "resolved_model_path": str(resolved_model),
        "conditions": list(inputs),
        "nodes_per_condition": 77,
        "sequence_length": 396,
        "do_sample": False,
        "num_beams": 1,
        "outputs": written,
    }
    (output / "ground_truth_3di_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
