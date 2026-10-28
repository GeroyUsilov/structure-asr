#!/usr/bin/env python3
"""Convert evolved AA tips to 3Di using the existing aleks_claude ProstT5 path.

This adapts ``protein_mcmc_utils.yield_3di`` and the resumable batching in
``phylo_3di_workflow/run_phylo_3di.py`` / ``runpod_3di_uncertainty``. It uses
deterministic greedy generation from the full ProstT5 seq2seq model, not the
separate encoder-only CNN predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from forward_asr_benchmark.io import read_fasta, write_fasta


MODEL_ID = "Rostlab/ProstT5"
THREEDI_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_AA = str.maketrans({token: "X" for token in "UZOBJ"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rate_label(rate: float) -> str:
    return format(rate, "g").replace("-", "m").replace(".", "p")


def has_model_weights(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def materialize_model(model_name_or_path: str) -> Path:
    requested = Path(model_name_or_path).expanduser()
    if requested.exists():
        if not (requested / "config.json").exists() or not has_model_weights(requested):
            raise RuntimeError(f"{requested} is not a complete ProstT5 model directory")
        return requested.resolve()
    from huggingface_hub import snapshot_download

    print(
        f"Downloading/caching {model_name_or_path} under "
        f"{os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))}",
        flush=True,
    )
    return Path(snapshot_download(repo_id=model_name_or_path)).resolve()


def load_prostt5(
    model_name_or_path: str,
    device_name: str,
    dtype_name: str,
):
    import torch
    from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    if device.type == "cpu" and dtype_name == "float16":
        raise ValueError("float16 ProstT5 inference is not supported on CPU")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]
    local_model = materialize_model(model_name_or_path)
    print(f"Loading ProstT5 tokenizer from {local_model}", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(
        str(local_model), do_lower_case=False, local_files_only=True
    )
    print(f"Loading full ProstT5 seq2seq model on {device} ({dtype_name})", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(local_model),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device, local_model


def generate_equal_length_batch(
    sequences: Sequence[str], tokenizer, model, device, torch_module
) -> list[str]:
    """Deterministic AA-to-3Di translation matching protein_mcmc_utils.yield_3di."""
    if not sequences:
        return []
    length = len(sequences[0])
    if any(len(sequence) != length for sequence in sequences):
        raise ValueError("ProstT5 batch contains unequal sequence lengths")
    prepared = [
        "<AA2fold> " + " ".join(sequence.translate(AMBIGUOUS_AA))
        for sequence in sequences
    ]
    tokenized = tokenizer(
        prepared,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    ).to(device)
    with torch_module.inference_mode():
        generated = model.generate(
            tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            min_length=length + 1,
            max_length=length + 1,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    structures = ["".join(text.split()).upper() for text in decoded]
    for structure in structures:
        if len(structure) != length:
            raise RuntimeError(
                f"ProstT5 length mismatch: {length} AA produced {len(structure)} 3Di states"
            )
        invalid = set(structure) - THREEDI_ALPHABET
        if invalid:
            raise RuntimeError(f"ProstT5 produced invalid 3Di states: {sorted(invalid)}")
    if device.type == "cuda":
        torch_module.cuda.empty_cache()
    return structures


def balanced_chunk_slices(length: int, max_chunk_aa: int) -> list[tuple[int, int]]:
    chunk_count = max(1, math.ceil(length / max_chunk_aa))
    chunk_size = math.ceil(length / chunk_count)
    return [
        (start, min(start + chunk_size, length))
        for start in range(0, length, chunk_size)
    ]


def encode_batch(
    sequences: Sequence[str],
    tokenizer,
    model,
    device,
    torch_module,
    max_chunk_aa: int,
) -> list[str]:
    slices = balanced_chunk_slices(len(sequences[0]), max_chunk_aa)
    assembled = ["" for _ in sequences]
    for start, end in slices:
        chunks = generate_equal_length_batch(
            [sequence[start:end] for sequence in sequences],
            tokenizer,
            model,
            device,
            torch_module,
        )
        assembled = [prefix + chunk for prefix, chunk in zip(assembled, chunks)]
    return assembled


def load_checkpoint(
    path: Path, identifiers: Sequence[str], sequences: Sequence[str]
) -> dict[int, str]:
    completed: dict[int, str] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Ignoring incomplete checkpoint line {line_number}", flush=True)
                break
            index = int(row["index"])
            if not 0 <= index < len(sequences):
                raise ValueError(f"checkpoint index {index} is out of range")
            if row["identifier"] != identifiers[index] or row["aa"] != sequences[index]:
                raise ValueError("3Di checkpoint does not match current AA inputs")
            structure = str(row["three_di"]).upper()
            if len(structure) != len(sequences[index]):
                raise ValueError(f"checkpoint 3Di length mismatch at index {index}")
            invalid = set(structure) - THREEDI_ALPHABET
            if invalid:
                raise ValueError(
                    f"checkpoint contains invalid 3Di states at index {index}: "
                    f"{sorted(invalid)}"
                )
            completed[index] = structure
    return completed


def encode_with_checkpoint(
    identifiers: Sequence[str],
    sequences: Sequence[str],
    checkpoint: Path,
    tokenizer,
    model,
    device,
    torch_module,
    batch_size: int,
    max_chunk_aa: int,
    resume: bool,
) -> list[str]:
    from tqdm.auto import tqdm

    completed = load_checkpoint(checkpoint, identifiers, sequences) if resume else {}
    if not resume:
        checkpoint.write_text("", encoding="utf-8")
    pending = [index for index in range(len(sequences)) if index not in completed]
    progress = tqdm(
        total=len(sequences),
        initial=len(completed),
        desc="ProstT5 AA→3Di",
        unit="seq",
    )
    for start in range(0, len(pending), batch_size):
        indices = pending[start : start + batch_size]
        structures = encode_batch(
            [sequences[index] for index in indices],
            tokenizer,
            model,
            device,
            torch_module,
            max_chunk_aa,
        )
        with checkpoint.open("a", encoding="utf-8") as handle:
            for index, structure in zip(indices, structures):
                handle.write(
                    json.dumps(
                        {
                            "index": index,
                            "identifier": identifiers[index],
                            "aa": sequences[index],
                            "three_di": structure,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                completed[index] = structure
            handle.flush()
            os.fsync(handle.fileno())
        progress.update(len(indices))
    progress.close()
    return [completed[index] for index in range(len(sequences))]


def outputs_are_complete(results_dir: Path, labels: Sequence[str]) -> bool:
    expected_method = "deterministic greedy <AA2fold> generation"
    for label in labels:
        directory = results_dir / f"mu_{label}"
        fasta = directory / "leaves_3di.fasta"
        manifest_path = directory / "leaves_3di_manifest.json"
        if not fasta.exists() or not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if manifest.get("method") != expected_method:
            print(f"Ignoring incompatible cached 3Di output in {directory}", flush=True)
            return False
    return True


def build_archive(results_dir: Path, labels: Sequence[str], archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".tmp")
    with tarfile.open(temporary, "w:gz") as bundle:
        for name in ("sweep_plan.json", "sweep_summary.json", "three_di_manifest.json"):
            bundle.add(results_dir / name, arcname=f"ef_tu_3di_outputs/{name}")
        for label in labels:
            source = results_dir / f"mu_{label}"
            archive_root = f"ef_tu_3di_outputs/mu_{label}"
            files = {
                source / "leaves_3di.fasta": "leaves_3di.fasta",
                source / "leaves_3di_manifest.json": "leaves_3di_manifest.json",
                source / "simulation" / "leaves.fasta": "leaves_aa.fasta",
                source / "simulation" / "tree.nwk": "tree.nwk",
                source / "simulation" / "manifest.json": "evolution_manifest.json",
            }
            for path, name in files.items():
                bundle.add(path, arcname=f"{archive_root}/{name}")
    temporary.replace(archive)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--rates", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-chunk-aa", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_chunk_aa < 1:
        raise ValueError("batch size and chunk length must be positive")

    results_dir = args.results_dir.resolve()
    labels = [rate_label(rate) for rate in args.rates]
    summary_path = results_dir / "sweep_summary.json"
    if not summary_path.exists() or json.loads(summary_path.read_text()).get("status") != "complete":
        raise RuntimeError("the requested evolution sweep is not complete")

    aa_by_run: dict[str, dict[str, str]] = {}
    for label in labels:
        records = read_fasta(results_dir / f"mu_{label}" / "simulation" / "leaves.fasta")
        if len(records) != 39:
            raise RuntimeError(f"mu_{label} has {len(records)} AA tips instead of 39")
        if any(len(sequence) != 396 for sequence in records.values()):
            raise RuntimeError(f"mu_{label} contains a non-396-aa tip")
        aa_by_run[label] = records

    resolved_model = None
    if not outputs_are_complete(results_dir, labels):
        work = results_dir / "three_di_work"
        work.mkdir(parents=True, exist_ok=True)
        combined_records = {
            f"mu_{label}__{identifier}": sequence
            for label in labels
            for identifier, sequence in aa_by_run[label].items()
        }
        identifiers = list(combined_records)
        sequences = list(combined_records.values())
        write_fasta(work / "combined_leaves_aa.fasta", combined_records)
        torch_module, tokenizer, model, device, resolved_model = load_prostt5(
            args.model, args.device, args.dtype
        )
        structures = encode_with_checkpoint(
            identifiers,
            sequences,
            work / "encoding_checkpoint.jsonl",
            tokenizer,
            model,
            device,
            torch_module,
            args.batch_size,
            args.max_chunk_aa,
            args.resume,
        )
        combined_3di = dict(zip(identifiers, structures))
        write_fasta(work / "combined_leaves_3di.fasta", combined_3di)

        for rate, label in zip(args.rates, labels):
            destination = results_dir / f"mu_{label}"
            output_records = {
                identifier: combined_3di[f"mu_{label}__{identifier}"]
                for identifier in aa_by_run[label]
            }
            output_fasta = destination / "leaves_3di.fasta"
            write_fasta(output_fasta, output_records)
            manifest = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "mutation_scale": rate,
                "tip_count": len(output_records),
                "sequence_length": 396,
                "aa_fasta_sha256": sha256(destination / "simulation" / "leaves.fasta"),
                "three_di_fasta_sha256": sha256(output_fasta),
                "method": "deterministic greedy <AA2fold> generation",
                "model": args.model,
                "resolved_model_path": str(resolved_model),
                "do_sample": False,
                "num_beams": 1,
                "implementation": "adapted from aleks_claude/protein_mcmc_utils.py and phylo_3di_workflow",
            }
            (destination / "leaves_3di_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    per_run = []
    for rate, label in zip(args.rates, labels):
        fasta = results_dir / f"mu_{label}" / "leaves_3di.fasta"
        records = read_fasta(fasta)
        if len(records) != 39 or any(len(sequence) != 396 for sequence in records.values()):
            raise RuntimeError(f"cached mu_{label} 3Di output failed validation")
        per_run.append(
            {
                "mutation_scale": rate,
                "tip_count": 39,
                "sequence_length": 396,
                "three_di_fasta_sha256": sha256(fasta),
            }
        )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "method": "full ProstT5 seq2seq deterministic greedy AA-to-3Di",
        "model": args.model,
        "batch_size": args.batch_size,
        "max_chunk_aa": args.max_chunk_aa,
        "checkpoint": "three_di_work/encoding_checkpoint.jsonl",
        "runs": per_run,
    }
    (results_dir / "three_di_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    build_archive(results_dir, labels, args.output_archive.resolve())
    print(f"3Di download archive: {args.output_archive.resolve()}", flush=True)
    print(f"{len(labels)}-RATE EF-TU 3DI CONVERSION PASSED", flush=True)


if __name__ == "__main__":
    main()
