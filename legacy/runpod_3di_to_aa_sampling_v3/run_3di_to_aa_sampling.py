#!/usr/bin/env python3
"""Sample ProstT5 amino-acid sequences from fixed ML 3Di reconstructions.

For each input 3Di posterior table, this program:
1. takes the row-wise maximum-posterior 3Di state to form one fixed sequence;
2. samples repeated ProstT5 <fold2AA> translations of that sequence;
3. tabulates empirical amino-acid probabilities at every site.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


AA_STATES = list("ARNDCQEGHILKMFPSTWYV")
OUTPUT_AA_STATES = AA_STATES + ["X"]
PROBABILITY_COLUMNS = [f"p_{state}" for state in AA_STATES]
AMBIGUOUS_AA_RE = re.compile(r"[UZOBJ]")
NODE_FILES = OrderedDict(
    [
        ("GluGln_3Di16", "3Di_Node16.csv"),
        ("Node11", "3Di_Node11.csv"),
        ("Node245", "3Di_Node245.csv"),
    ]
)
CHECKPOINT_SCHEMA_VERSION = 3


def load_3di_posterior(path: Path, expected_node: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Site", *PROBABILITY_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if frame.empty:
        raise ValueError(f"{path}: no sites")

    sites = frame["Site"].to_numpy(dtype=int)
    if not np.array_equal(sites, np.arange(1, len(frame) + 1)):
        raise ValueError(f"{path}: Site must be consecutive and 1-based")

    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError(f"{path}: probabilities must be finite and nonnegative")
    totals = probabilities.sum(axis=1)
    if (totals <= 0).any():
        bad_site = int(frame.loc[np.flatnonzero(totals <= 0)[0], "Site"])
        raise ValueError(f"{path}: site {bad_site} has zero probability mass")

    normalized = probabilities / totals[:, None]
    argmax = normalized.argmax(axis=1)
    frame = frame.copy()
    frame.loc[:, PROBABILITY_COLUMNS] = normalized
    frame["ML_3Di_State"] = [AA_STATES[index] for index in argmax]
    frame["ML_3Di_Posterior"] = normalized[np.arange(len(frame)), argmax]
    frame["Analysis_Node"] = expected_node
    return frame


def maximum_likelihood_3di(frame: pd.DataFrame) -> str:
    return "".join(frame["ML_3Di_State"].astype(str))


def balanced_chunk_slices(length: int, max_chunk_states: int) -> List[Tuple[int, int]]:
    if length <= 0 or max_chunk_states <= 0:
        raise ValueError("length and max_chunk_states must be positive")
    chunk_count = max(1, math.ceil(length / max_chunk_states))
    chunk_size = math.ceil(length / chunk_count)
    return [(start, min(start + chunk_size, length)) for start in range(0, length, chunk_size)]


def write_fasta(path: Path, records: Iterable[Tuple[str, str]], line_width: int = 80) -> None:
    with path.open("w") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n")
            for start in range(0, len(sequence), line_width):
                handle.write(sequence[start : start + line_width] + "\n")


def normalize_generated_aa(raw: str, expected_length: int) -> str:
    compact = "".join(raw.split())
    if len(compact) != expected_length:
        raise RuntimeError(
            f"ProstT5 length mismatch: expected {expected_length} AA tokens, "
            f"received {len(compact)} ({compact[:40]!r})"
        )
    lowercase_positions = [index + 1 for index, character in enumerate(compact) if character.islower()]
    if lowercase_positions:
        raise RuntimeError(
            "ProstT5 emitted lowercase 3Di tokens during <fold2AA> at output "
            f"positions {lowercase_positions[:10]}"
        )
    compact = AMBIGUOUS_AA_RE.sub("X", compact)
    invalid = sorted(set(compact).difference(OUTPUT_AA_STATES))
    if invalid:
        raise RuntimeError(f"ProstT5 emitted unsupported AA tokens: {invalid}")
    return compact


def resolve_device(torch_module, requested: str):
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch_module.device(requested)


def resolve_dtype(torch_module, requested: str, device):
    if requested == "auto":
        requested = "float16" if device.type == "cuda" else "float32"
    if device.type == "cpu" and requested == "float16":
        raise ValueError("float16 ProstT5 inference is not supported on CPU")
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[requested]


def materialize_model(model_name_or_path: str) -> str:
    candidates = [
        Path(model_name_or_path).expanduser(),
        Path("/tmp/ProstT5"),
        Path("/workspace/ProstT5"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    from huggingface_hub import snapshot_download

    cache = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    print(f"Downloading/caching {model_name_or_path} under {cache}...", flush=True)
    return snapshot_download(repo_id=model_name_or_path)


def load_prostt5(model_name_or_path: str, device_name: str, dtype_name: str):
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-runpod.txt before running inference") from exc

    model_path = materialize_model(model_name_or_path)
    device = resolve_device(torch, device_name)
    dtype = resolve_dtype(torch, dtype_name, device)
    print(f"Loading ProstT5 tokenizer from {model_path}...", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(
        model_path, do_lower_case=False, local_files_only=True
    )
    print(f"Loading ProstT5 on {device} ({dtype})...", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device, dtype, model_path


def sample_equal_length_chunk(
    structure: str,
    count: int,
    tokenizer,
    model,
    device,
    torch_module,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> List[Optional[str]]:
    if count <= 0:
        return []
    length = len(structure)
    prepared_one = "<fold2AA> " + " ".join(structure.lower())
    tokenized = tokenizer(
        [prepared_one] * count,
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
            do_sample=True,
            num_beams=1,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            renormalize_logits=True,
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    samples: List[Optional[str]] = []
    for text in decoded:
        try:
            samples.append(normalize_generated_aa(text, length))
        except RuntimeError:
            samples.append(None)
    if device.type == "cuda":
        torch_module.cuda.empty_cache()
    return samples


def read_checkpoint(
    path: Path, sequence_length: int, allow_legacy_unversioned: bool = False
) -> List[str]:
    if not path.exists():
        return []
    samples: List[str] = []
    with path.open() as handle:
        for expected_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            version = record.get("schema_version")
            if version is None and allow_legacy_unversioned:
                pass
            elif version != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError(
                    f"{path}: incompatible checkpoint schema {version!r}; "
                    f"expected {CHECKPOINT_SCHEMA_VERSION}"
                )
            if int(record["index"]) != expected_index:
                raise ValueError(f"{path}: checkpoint indices are not consecutive")
            sequence = str(record["aa"])
            if len(sequence) != sequence_length:
                raise ValueError(f"{path}: checkpoint sample has the wrong length")
            invalid = sorted(set(sequence).difference(OUTPUT_AA_STATES))
            if invalid:
                raise ValueError(f"{path}: checkpoint sample contains invalid AA tokens {invalid}")
            samples.append(sequence)
    return samples


def append_checkpoint(path: Path, start: int, samples: Sequence[str]) -> None:
    with path.open("a") as handle:
        for offset, sequence in enumerate(samples):
            handle.write(
                json.dumps(
                    {
                        "schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "sampling_filter": "reject_non_AA_outputs",
                        "index": start + offset,
                        "aa": sequence,
                    }
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def sample_node(
    node: str,
    structure: str,
    node_outdir: Path,
    tokenizer,
    model,
    device,
    torch_module,
    n_samples: int,
    batch_size: int,
    seed: int,
    max_chunk_states: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    resume: bool,
    resume_from_node_dir: Optional[Path],
) -> List[str]:
    checkpoint = node_outdir / "sampling_checkpoint.jsonl"
    if resume and not checkpoint.exists() and resume_from_node_dir is not None:
        legacy_checkpoint = resume_from_node_dir / "sampling_checkpoint.jsonl"
        if legacy_checkpoint.exists():
            imported = read_checkpoint(
                legacy_checkpoint,
                len(structure),
                allow_legacy_unversioned=True,
            )
            if len(imported) > n_samples:
                raise ValueError(
                    f"{legacy_checkpoint} has {len(imported)} samples but --n-samples is {n_samples}"
                )
            append_checkpoint(checkpoint, 0, imported)
            print(
                f"{node}: imported {len(imported)} completed samples from {legacy_checkpoint}",
                flush=True,
            )
    samples = read_checkpoint(checkpoint, len(structure)) if resume else []
    if len(samples) > n_samples:
        raise ValueError(
            f"{checkpoint} has {len(samples)} samples but --n-samples is {n_samples}"
        )
    if samples:
        print(f"{node}: resuming from {len(samples)}/{n_samples} samples", flush=True)
    elif checkpoint.exists():
        checkpoint.unlink()

    chunks = balanced_chunk_slices(len(structure), max_chunk_states)
    attempt_round = 0
    while len(samples) < n_samples:
        start = len(samples)
        count = min(batch_size, n_samples - start)
        chunk_outputs: List[List[Optional[str]]] = []
        for chunk_index, (left, right) in enumerate(chunks):
            batch_seed = (
                seed
                + start * 1009
                + attempt_round * 10_000_019
                + chunk_index * 1_000_003
            )
            torch_module.manual_seed(batch_seed)
            if device.type == "cuda":
                torch_module.cuda.manual_seed_all(batch_seed)
            chunk_outputs.append(
                sample_equal_length_chunk(
                    structure[left:right],
                    count,
                    tokenizer,
                    model,
                    device,
                    torch_module,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty,
                )
            )
        batch: List[str] = []
        rejected = 0
        for parts in zip(*chunk_outputs):
            if any(part is None for part in parts):
                rejected += 1
                continue
            batch.append("".join(part for part in parts if part is not None))
        if any(len(sequence) != len(structure) for sequence in batch):
            raise RuntimeError("Internal error while assembling sampled AA chunks")
        batch = batch[: n_samples - start]
        if batch:
            append_checkpoint(checkpoint, start, batch)
            samples.extend(batch)
        attempt_round += 1
        print(
            f"{node}: accepted {len(samples)}/{n_samples} AA sequences "
            f"({rejected} rejected in latest batch)",
            flush=True,
        )
    return samples


def summarize_samples(
    node: str, frame: pd.DataFrame, samples: Sequence[str]
) -> Tuple[pd.DataFrame, dict]:
    if not samples:
        raise ValueError("No AA samples supplied")
    length = len(frame)
    if any(len(sequence) != length for sequence in samples):
        raise ValueError("Sampled AA sequences do not all match the 3Di site count")

    state_to_index = {state: index for index, state in enumerate(OUTPUT_AA_STATES)}
    counts = np.zeros((length, len(OUTPUT_AA_STATES)), dtype=np.int64)
    for sequence in samples:
        for site_index, state in enumerate(sequence):
            counts[site_index, state_to_index[state]] += 1
    probabilities = counts / len(samples)
    nonzero = probabilities > 0
    entropy = -(np.where(nonzero, probabilities * np.log2(np.where(nonzero, probabilities, 1)), 0)).sum(axis=1)
    top_index = probabilities.argmax(axis=1)

    output = pd.DataFrame(
        {
            "Node": node,
            "Site": frame["Site"].to_numpy(dtype=int),
            "ML_3Di_State": frame["ML_3Di_State"].to_numpy(),
            "ML_3Di_Posterior": frame["ML_3Di_Posterior"].to_numpy(dtype=float),
            "AA_State": [OUTPUT_AA_STATES[index] for index in top_index],
            "AA_top_probability": probabilities[np.arange(length), top_index],
            "AA_entropy_bits": entropy,
            "AA_effective_states": np.power(2.0, entropy),
        }
    )
    for index, state in enumerate(OUTPUT_AA_STATES):
        output[f"p_{state}"] = probabilities[:, index]
        output[f"count_{state}"] = counts[:, index]

    summary = {
        "Node": node,
        "sites": length,
        "samples": len(samples),
        "chunks": len(balanced_chunk_slices(length, 500)),
        "mean_AA_entropy_bits": float(entropy.mean()),
        "median_AA_entropy_bits": float(np.median(entropy)),
        "mean_AA_top_probability": float(output["AA_top_probability"].mean()),
        "fraction_noncanonical_X": float(probabilities[:, -1].mean()),
    }
    return output, summary


def validate_inputs(input_dir: Path) -> Dict[str, dict]:
    report: Dict[str, dict] = {}
    for node, filename in NODE_FILES.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frame = load_3di_posterior(path, node)
        ml = maximum_likelihood_3di(frame)
        state_matches = None
        if "State" in frame.columns:
            state_matches = int(
                (frame["State"].astype(str).str.upper() == frame["ML_3Di_State"]).sum()
            )
        report[node] = {
            "file": str(path),
            "sites": len(frame),
            "ML_3Di_length": len(ml),
            "State_column_matches_argmax": state_matches,
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "inputs")
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional prior results directory whose unversioned checkpoints are imported once",
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chunk-states", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--model", default="Rostlab/ProstT5")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0 or args.batch_size <= 0 or args.max_chunk_states <= 0:
        raise ValueError("sample count, batch size, and chunk size must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("invalid stochastic decoding settings")
    if args.repetition_penalty <= 0:
        raise ValueError("--repetition-penalty must be positive")

    validation_report = validate_inputs(args.input_dir)
    print("Input validation:")
    print(json.dumps(validation_report, indent=2))
    if args.validate_only:
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    inputs = {
        node: load_3di_posterior(args.input_dir / filename, node)
        for node, filename in NODE_FILES.items()
    }
    torch_module, tokenizer, model, device, dtype, model_path = load_prostt5(
        args.model, args.device, args.dtype
    )

    all_tables: List[pd.DataFrame] = []
    summaries: List[dict] = []
    for node_index, (node, frame) in enumerate(inputs.items()):
        node_outdir = args.outdir / node
        node_outdir.mkdir(parents=True, exist_ok=True)
        ml_structure = maximum_likelihood_3di(frame)
        write_fasta(
            node_outdir / "maximum_likelihood_3di.fasta",
            [(f"{node}_maximum_likelihood_3Di", ml_structure)],
        )
        samples = sample_node(
            node,
            ml_structure,
            node_outdir,
            tokenizer,
            model,
            device,
            torch_module,
            args.n_samples,
            args.batch_size,
            args.seed + node_index * 10_000_019,
            args.max_chunk_states,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repetition_penalty,
            args.resume,
            (args.resume_from / node) if args.resume_from is not None else None,
        )
        write_fasta(
            node_outdir / "sampled_aa.fasta",
            ((f"{node}_sample_{index + 1:06d}", sequence) for index, sequence in enumerate(samples)),
        )
        table, summary = summarize_samples(node, frame, samples)
        table.to_csv(node_outdir / "empirical_aa_probabilities.tsv", sep="\t", index=False)
        all_tables.append(table)
        summary["chunks"] = len(balanced_chunk_slices(len(frame), args.max_chunk_states))
        summaries.append(summary)

    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(args.outdir / "all_nodes_empirical_aa_probabilities.tsv", sep="\t", index=False)
    pd.DataFrame(summaries).to_csv(args.outdir / "node_summary.tsv", sep="\t", index=False)
    manifest = {
        "method": "fixed maximum-posterior 3Di sequence translated with stochastic ProstT5 <fold2AA> decoding",
        "model_requested": args.model,
        "model_resolved": model_path,
        "device": str(device),
        "dtype": str(dtype),
        "n_samples_per_node": args.n_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "max_chunk_states": args.max_chunk_states,
        "generation": {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "num_beams": 1,
            "invalid_output_handling": "reject and retry any sample containing non-AA output tokens",
        },
        "nodes": validation_report,
        "probability_definition": "empirical per-site frequencies across ProstT5 stochastic translations",
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Finished. Results written to {args.outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
