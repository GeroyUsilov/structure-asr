#!/usr/bin/env python3
"""Propagate ancestral AA uncertainty through ProstT5 and compare 3Di posteriors.

For each configured node, this program:
1. samples gap-free protein sequences independently from sitewise AA posteriors;
2. deterministically converts each sequence to ProstT5 3Di states;
3. estimates the empirical 3Di distribution at every alignment site;
4. compares it with the node's stored sitewise 3Di probability distribution;
5. writes site tables, summaries, checkpoints, and diagnostic figures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


NODE_FILES = {
    "Node16": ("AAseq_Node16.csv", "3Di_Node16.csv"),
    "Node11": ("Seq_Node11.csv", "3Di_Node11.csv"),
    "Node245": ("Seq_Node245.csv", "3Di_Node245.csv"),
}

AMBIGUOUS_AA_RE = re.compile(r"[UZOBJ]")
REQUIRED_COLUMNS = {"Node", "Site", "State"}


def probability_columns(frame: pd.DataFrame) -> List[str]:
    columns = [column for column in frame.columns if column.startswith("p_")]
    if len(columns) != 20:
        raise ValueError(f"Expected 20 probability columns, found {len(columns)}")
    return columns


def normalized_probabilities(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Probability matrix contains non-finite values")
    if (values < 0).any():
        raise ValueError("Probability matrix contains negative values")
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Probability matrix contains an all-zero row")
    return values / totals


def load_node_inputs(input_dir: Path, node: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aa_name, di_name = NODE_FILES[node]
    aa = pd.read_csv(input_dir / aa_name)
    direct = pd.read_csv(input_dir / di_name)
    for label, frame in [(aa_name, aa), (di_name, direct)]:
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")
        probability_columns(frame)
        if frame.empty:
            raise ValueError(f"{label} is empty")
        labels = set(frame["Node"].astype(str))
        if labels != {node}:
            raise ValueError(f"{label} contains node labels {sorted(labels)}, expected {node}")
        normalized_probabilities(frame, probability_columns(frame))
    if len(aa) != len(direct):
        raise ValueError(f"{node}: AA and 3Di files have different site counts")
    if not aa["Site"].reset_index(drop=True).equals(direct["Site"].reset_index(drop=True)):
        raise ValueError(f"{node}: AA and 3Di Site columns do not match")
    return aa.reset_index(drop=True), direct.reset_index(drop=True)


def validate_inputs(input_dir: Path) -> Dict[str, dict]:
    report: Dict[str, dict] = {}
    for node in NODE_FILES:
        aa, direct = load_node_inputs(input_dir, node)
        aa_cols = probability_columns(aa)
        di_cols = probability_columns(direct)
        aa_sums = aa[aa_cols].sum(axis=1)
        di_sums = direct[di_cols].sum(axis=1)
        aa_argmax = pd.Series(aa_cols, index=aa_cols).iloc[
            np.argmax(aa[aa_cols].to_numpy(dtype=float), axis=1)
        ].str[2:].to_numpy()
        di_argmax = pd.Series(di_cols, index=di_cols).iloc[
            np.argmax(direct[di_cols].to_numpy(dtype=float), axis=1)
        ].str[2:].to_numpy()
        report[node] = {
            "sites": len(aa),
            "aa_state_gaps": int((aa["State"].astype(str) == "-").sum()),
            "direct_3di_state_gaps": int((direct["State"].astype(str) == "-").sum()),
            "aa_state_differs_from_probability_argmax": int(
                (aa["State"].astype(str).to_numpy() != aa_argmax).sum()
            ),
            "direct_state_differs_from_probability_argmax": int(
                (direct["State"].astype(str).to_numpy() != di_argmax).sum()
            ),
            "aa_probability_sum_min": float(aa_sums.min()),
            "aa_probability_sum_max": float(aa_sums.max()),
            "direct_probability_sum_min": float(di_sums.min()),
            "direct_probability_sum_max": float(di_sums.max()),
        }
    return report


def entropy_bits(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    terms[positive] = probabilities[positive] * np.log2(probabilities[positive])
    return -terms.sum(axis=-1)


def js_divergence_bits(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    midpoint = (first + second) / 2.0

    def kl_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        terms = np.zeros_like(left)
        positive = left > 0
        terms[positive] = left[positive] * np.log2(left[positive] / right[positive])
        return terms.sum(axis=-1)

    return 0.5 * kl_divergence(first, midpoint) + 0.5 * kl_divergence(second, midpoint)


def sample_sequences(
    aa_frame: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
) -> List[str]:
    columns = probability_columns(aa_frame)
    alphabet = np.array([column[2:] for column in columns], dtype="U1")
    probabilities = normalized_probabilities(aa_frame, columns)
    sampled = np.empty((n_samples, len(aa_frame)), dtype="U1")
    for site_index, site_probabilities in enumerate(probabilities):
        choices = rng.choice(len(alphabet), size=n_samples, p=site_probabilities)
        sampled[:, site_index] = alphabet[choices]
    return ["".join(row) for row in sampled]


def write_fasta(path: Path, records: Iterable[Tuple[str, str]]) -> None:
    with path.open("w") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    identifier = None
    parts: List[str] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    records.append((identifier, "".join(parts)))
                identifier = line[1:].split()[0]
                parts = []
            else:
                if identifier is None:
                    raise ValueError(f"Sequence appears before FASTA header in {path}")
                parts.append(line)
    if identifier is not None:
        records.append((identifier, "".join(parts)))
    return records


def prepare_sampled_sequences(
    node: str,
    aa_frame: pd.DataFrame,
    node_outdir: Path,
    n_samples: int,
    seed: int,
    resume: bool,
) -> List[str]:
    fasta_path = node_outdir / "sampled_aa.fasta"
    if resume and fasta_path.exists():
        records = read_fasta(fasta_path)
        sequences = [sequence for _, sequence in records]
        if len(sequences) != n_samples:
            raise ValueError(
                f"{fasta_path} contains {len(sequences)} samples, expected {n_samples}. "
                "Use the original --n-samples or start with a new output directory."
            )
        if any(len(sequence) != len(aa_frame) for sequence in sequences):
            raise ValueError(f"{fasta_path} contains a sequence with the wrong length")
        print(f"{node}: reusing {len(sequences)} sampled AA sequences", flush=True)
        return sequences

    sequences = sample_sequences(aa_frame, n_samples, np.random.default_rng(seed))
    write_fasta(
        fasta_path,
        ((f"{node}_sample_{index + 1:06d}", sequence) for index, sequence in enumerate(sequences)),
    )
    print(f"{node}: sampled {len(sequences)} AA sequences", flush=True)
    return sequences


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

    print(
        f"Downloading/caching {model_name_or_path} under "
        f"{os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))}...",
        flush=True,
    )
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


def balanced_chunk_slices(length: int, max_chunk_aa: int) -> List[Tuple[int, int]]:
    chunk_count = max(1, math.ceil(length / max_chunk_aa))
    chunk_size = math.ceil(length / chunk_count)
    return [
        (start, min(start + chunk_size, length))
        for start in range(0, length, chunk_size)
    ]


def generate_equal_length_batch(
    sequences: Sequence[str], tokenizer, model, device, torch_module
) -> List[str]:
    if not sequences:
        return []
    length = len(sequences[0])
    if any(len(sequence) != length for sequence in sequences):
        raise ValueError("ProstT5 batch contains unequal sequence lengths")
    prepared = [
        "<AA2fold> " + " ".join(AMBIGUOUS_AA_RE.sub("X", sequence))
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
    if device.type == "cuda":
        torch_module.cuda.empty_cache()
    return structures


def encode_batch(
    sequences: Sequence[str],
    tokenizer,
    model,
    device,
    torch_module,
    max_chunk_aa: int,
) -> List[str]:
    if not sequences:
        return []
    length = len(sequences[0])
    slices = balanced_chunk_slices(length, max_chunk_aa)
    assembled = ["" for _ in sequences]
    for start, end in slices:
        chunk_results = generate_equal_length_batch(
            [sequence[start:end] for sequence in sequences],
            tokenizer,
            model,
            device,
            torch_module,
        )
        assembled = [prefix + chunk for prefix, chunk in zip(assembled, chunk_results)]
    if any(len(structure) != length for structure in assembled):
        raise RuntimeError("Internal error while assembling chunked 3Di encodings")
    return assembled


def load_encoding_checkpoint(
    path: Path, expected_sequences: Sequence[str]
) -> Dict[int, str]:
    completed: Dict[int, str] = {}
    if not path.exists():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Ignoring incomplete checkpoint line {line_number} in {path}", flush=True)
                break
            index = int(record["index"])
            if index < 0 or index >= len(expected_sequences):
                raise ValueError(f"Checkpoint index {index} is out of range")
            if record["aa"] != expected_sequences[index]:
                raise ValueError(
                    f"Checkpoint AA sequence {index} does not match sampled_aa.fasta"
                )
            structure = str(record["three_di"]).upper()
            if len(structure) != len(expected_sequences[index]):
                raise ValueError(f"Checkpoint 3Di sequence {index} has the wrong length")
            completed[index] = structure
    return completed


def encode_with_checkpoint(
    node: str,
    sequences: Sequence[str],
    node_outdir: Path,
    tokenizer,
    model,
    device,
    torch_module,
    batch_size: int,
    max_chunk_aa: int,
    resume: bool,
) -> List[str]:
    checkpoint_path = node_outdir / "encoding_checkpoint.jsonl"
    completed = load_encoding_checkpoint(checkpoint_path, sequences) if resume else {}
    if completed:
        print(f"{node}: resuming with {len(completed)}/{len(sequences)} encodings", flush=True)
    if not resume:
        checkpoint_path.write_text("")

    pending = [index for index in range(len(sequences)) if index not in completed]
    for batch_start in range(0, len(pending), batch_size):
        indices = pending[batch_start : batch_start + batch_size]
        structures = encode_batch(
            [sequences[index] for index in indices],
            tokenizer,
            model,
            device,
            torch_module,
            max_chunk_aa,
        )
        with checkpoint_path.open("a") as handle:
            for index, structure in zip(indices, structures):
                record = {
                    "index": index,
                    "aa": sequences[index],
                    "three_di": structure,
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                completed[index] = structure
            handle.flush()
            os.fsync(handle.fileno())
        print(
            f"{node}: encoded {len(completed)}/{len(sequences)} samples",
            flush=True,
        )

    structures = [completed[index] for index in range(len(sequences))]
    write_fasta(
        node_outdir / "sampled_3di.fasta",
        ((f"{node}_sample_{index + 1:06d}", sequence) for index, sequence in enumerate(structures)),
    )
    return structures


def empirical_probabilities(
    structures: Sequence[str], tokens: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    if not structures:
        raise ValueError("No 3Di structures were supplied")
    length = len(structures[0])
    if any(len(structure) != length for structure in structures):
        raise ValueError("3Di structures have unequal lengths")
    token_to_index = {token: index for index, token in enumerate(tokens)}
    counts = np.zeros((length, len(tokens)), dtype=np.int64)
    for sample_index, structure in enumerate(structures):
        for site_index, token in enumerate(structure.upper()):
            if token not in token_to_index:
                raise ValueError(
                    f"Sample {sample_index} site {site_index + 1}: unknown 3Di token {token!r}"
                )
            counts[site_index, token_to_index[token]] += 1
    return counts / len(structures), counts


def bootstrap_entropy_intervals(
    empirical: np.ndarray,
    n_samples: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_bootstrap <= 0:
        nan_values = np.full(empirical.shape[0], np.nan)
        return nan_values, nan_values.copy()
    lower = np.empty(empirical.shape[0], dtype=float)
    upper = np.empty(empirical.shape[0], dtype=float)
    for site_index, probabilities in enumerate(empirical):
        draws = rng.multinomial(n_samples, probabilities, size=n_bootstrap) / n_samples
        entropies = entropy_bits(draws)
        lower[site_index], upper[site_index] = np.quantile(entropies, [0.025, 0.975])
    return lower, upper


def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    mask = np.isfinite(first) & np.isfinite(second)
    if mask.sum() < 2 or np.std(first[mask]) == 0 or np.std(second[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(first[mask], second[mask])[0, 1])


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    mask = np.isfinite(first) & np.isfinite(second)
    if mask.sum() < 2:
        return float("nan")
    first_ranks = pd.Series(first[mask]).rank(method="average").to_numpy()
    second_ranks = pd.Series(second[mask]).rank(method="average").to_numpy()
    return safe_pearson(first_ranks, second_ranks)


def analyze_node(
    node: str,
    aa_frame: pd.DataFrame,
    direct_frame: pd.DataFrame,
    structures: Sequence[str],
    node_outdir: Path,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    aa_columns = probability_columns(aa_frame)
    direct_columns = probability_columns(direct_frame)
    aa_probabilities = normalized_probabilities(aa_frame, aa_columns)
    direct_probabilities = normalized_probabilities(direct_frame, direct_columns)
    tokens = [column[2:] for column in direct_columns]
    empirical, counts = empirical_probabilities(structures, tokens)

    aa_entropy = entropy_bits(aa_probabilities)
    empirical_entropy = entropy_bits(empirical)
    direct_entropy = entropy_bits(direct_probabilities)
    max_entropy = math.log2(len(tokens))
    empirical_ci_low, empirical_ci_high = bootstrap_entropy_intervals(
        empirical,
        len(structures),
        n_bootstrap,
        np.random.default_rng(bootstrap_seed),
    )
    divergence = js_divergence_bits(empirical, direct_probabilities)

    empirical_top_indices = np.argmax(empirical, axis=1)
    direct_top_indices = np.argmax(direct_probabilities, axis=1)
    empirical_top = np.array(tokens, dtype=object)[empirical_top_indices]
    direct_top = np.array(tokens, dtype=object)[direct_top_indices]
    provided_state = direct_frame["State"].astype(str).str.upper().to_numpy()
    provided_is_gap = provided_state == "-"
    token_to_index = {token: index for index, token in enumerate(tokens)}
    empirical_probability_of_provided_state = np.full(len(direct_frame), np.nan)
    for site_index, state in enumerate(provided_state):
        if state in token_to_index:
            empirical_probability_of_provided_state[site_index] = empirical[
                site_index, token_to_index[state]
            ]

    site_table = pd.DataFrame(
        {
            "Node": node,
            "Site": direct_frame["Site"].to_numpy(),
            "aa_provided_state": aa_frame["State"].astype(str).to_numpy(),
            "direct_3di_provided_state": provided_state,
            "direct_3di_probability_top_state": direct_top,
            "empirical_3di_top_state": empirical_top,
            "aa_entropy_bits": aa_entropy,
            "aa_normalized_entropy": aa_entropy / max_entropy,
            "empirical_3di_entropy_bits": empirical_entropy,
            "empirical_3di_entropy_ci95_low": empirical_ci_low,
            "empirical_3di_entropy_ci95_high": empirical_ci_high,
            "empirical_3di_normalized_entropy": empirical_entropy / max_entropy,
            "direct_3di_entropy_bits": direct_entropy,
            "direct_3di_normalized_entropy": direct_entropy / max_entropy,
            "entropy_delta_empirical_minus_direct_bits": empirical_entropy - direct_entropy,
            "empirical_3di_effective_states": np.power(2.0, empirical_entropy),
            "direct_3di_effective_states": np.power(2.0, direct_entropy),
            "empirical_3di_top_probability": empirical.max(axis=1),
            "direct_3di_top_probability": direct_probabilities.max(axis=1),
            "js_divergence_bits": divergence,
            "top_state_agreement_probability_argmax": empirical_top == direct_top,
            "provided_state_is_gap": provided_is_gap,
            "empirical_top_matches_provided_state": np.where(
                provided_is_gap, np.nan, empirical_top == provided_state
            ),
            "empirical_probability_of_direct_top_state": empirical[
                np.arange(len(empirical)), direct_top_indices
            ],
            "empirical_probability_of_provided_state": empirical_probability_of_provided_state,
        }
    )
    site_table.to_csv(node_outdir / "site_uncertainty.tsv", sep="\t", index=False)

    empirical_table = pd.DataFrame(
        {
            "Node": node,
            "Site": direct_frame["Site"].to_numpy(),
            "empirical_top_state": empirical_top,
            "empirical_top_probability": empirical.max(axis=1),
        }
    )
    for token_index, token in enumerate(tokens):
        empirical_table[f"p_{token}"] = empirical[:, token_index]
        empirical_table[f"count_{token}"] = counts[:, token_index]
    empirical_table.to_csv(
        node_outdir / "empirical_3di_probabilities.tsv", sep="\t", index=False
    )

    high_disagreement = site_table.sort_values(
        ["js_divergence_bits", "Site"], ascending=[False, True]
    ).head(25).copy()
    high_disagreement.insert(0, "disagreement_rank", np.arange(1, len(high_disagreement) + 1))
    high_disagreement.to_csv(
        node_outdir / "high_disagreement_sites.tsv", sep="\t", index=False
    )

    non_gap_mask = ~provided_is_gap
    summary = {
        "Node": node,
        "sites": len(site_table),
        "samples": len(structures),
        "aa_provided_state_gaps": int((aa_frame["State"].astype(str) == "-").sum()),
        "direct_3di_provided_state_gaps": int(provided_is_gap.sum()),
        "mean_aa_entropy_bits": float(np.mean(aa_entropy)),
        "mean_empirical_3di_entropy_bits": float(np.mean(empirical_entropy)),
        "median_empirical_3di_entropy_bits": float(np.median(empirical_entropy)),
        "mean_direct_3di_entropy_bits": float(np.mean(direct_entropy)),
        "median_direct_3di_entropy_bits": float(np.median(direct_entropy)),
        "mean_entropy_delta_empirical_minus_direct_bits": float(
            np.mean(empirical_entropy - direct_entropy)
        ),
        "median_entropy_delta_empirical_minus_direct_bits": float(
            np.median(empirical_entropy - direct_entropy)
        ),
        "pearson_empirical_vs_direct_entropy": safe_pearson(
            empirical_entropy, direct_entropy
        ),
        "spearman_empirical_vs_direct_entropy": safe_spearman(
            empirical_entropy, direct_entropy
        ),
        "spearman_aa_vs_empirical_3di_entropy": safe_spearman(
            aa_entropy, empirical_entropy
        ),
        "mean_js_divergence_bits": float(np.mean(divergence)),
        "median_js_divergence_bits": float(np.median(divergence)),
        "probability_argmax_top_state_agreement_fraction": float(
            np.mean(empirical_top == direct_top)
        ),
        "provided_state_agreement_fraction_excluding_gaps": float(
            np.mean(empirical_top[non_gap_mask] == provided_state[non_gap_mask])
        ),
    }
    return site_table, empirical_table, summary


def create_plots(site_tables: Mapping[str, pd.DataFrame], outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes = list(NODE_FILES)
    colors = {
        "aa": "#6B7280",
        "empirical": "#2563EB",
        "direct": "#D97706",
        "divergence": "#7C3AED",
    }
    entropy_ceiling = math.log2(20)

    figure, axes = plt.subplots(1, len(nodes), figsize=(15, 4), sharex=True, sharey=True)
    bins = np.linspace(0, entropy_ceiling, 32)
    for axis, node in zip(axes, nodes):
        table = site_tables[node]
        axis.hist(
            table["empirical_3di_entropy_bits"],
            bins=bins,
            alpha=0.55,
            color=colors["empirical"],
            label="AA-sampled → ProstT5",
        )
        axis.hist(
            table["direct_3di_entropy_bits"],
            bins=bins,
            alpha=0.55,
            color=colors["direct"],
            label="Stored 3Di posterior",
        )
        axis.set_title(node)
        axis.set_xlabel("3Di entropy (bits)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Number of sites")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Distribution of sitewise 3Di uncertainty")
    figure.tight_layout()
    figure.savefig(outdir / "entropy_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, len(nodes), figsize=(15, 4), sharex=True, sharey=True)
    for axis, node in zip(axes, nodes):
        table = site_tables[node]
        x = table["direct_3di_entropy_bits"].to_numpy()
        y = table["empirical_3di_entropy_bits"].to_numpy()
        rho = safe_spearman(x, y)
        axis.scatter(x, y, s=13, alpha=0.55, color=colors["empirical"], linewidths=0)
        axis.plot([0, entropy_ceiling], [0, entropy_ceiling], "--", color="#111827", lw=1)
        axis.text(0.04, 0.94, f"Spearman ρ = {rho:.3f}", transform=axis.transAxes, va="top")
        axis.set_title(node)
        axis.set_xlabel("Stored 3Di entropy (bits)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("AA-sampled → ProstT5 entropy (bits)")
    figure.suptitle("Sitewise 3Di uncertainty comparison")
    figure.tight_layout()
    figure.savefig(outdir / "entropy_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(len(nodes), 1, figsize=(15, 9), sharex=False)
    for axis, node in zip(axes, nodes):
        table = site_tables[node]
        sites = table["Site"].to_numpy()
        axis.plot(sites, table["aa_entropy_bits"], color=colors["aa"], lw=0.8, alpha=0.75, label="AA posterior")
        axis.plot(sites, table["empirical_3di_entropy_bits"], color=colors["empirical"], lw=1.0, label="AA-sampled → ProstT5")
        axis.plot(sites, table["direct_3di_entropy_bits"], color=colors["direct"], lw=1.0, label="Stored 3Di posterior")
        axis.set_ylim(-0.05, entropy_ceiling + 0.05)
        axis.set_ylabel(f"{node}\nEntropy (bits)")
        axis.grid(alpha=0.18)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("Alignment site")
    figure.suptitle("AA and 3Di uncertainty along each sequence")
    figure.tight_layout()
    figure.savefig(outdir / "per_site_uncertainty.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(len(nodes), 1, figsize=(15, 7.5), sharex=False)
    for axis, node in zip(axes, nodes):
        table = site_tables[node]
        axis.plot(
            table["Site"],
            table["js_divergence_bits"],
            color=colors["divergence"],
            lw=1.0,
        )
        axis.set_ylim(-0.01, 1.01)
        axis.set_ylabel(f"{node}\nJSD (bits)")
        axis.grid(alpha=0.18)
    axes[-1].set_xlabel("Alignment site")
    figure.suptitle("Distributional disagreement: sampled versus stored 3Di posterior")
    figure.tight_layout()
    figure.savefig(outdir / "js_divergence_by_site.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_manifest(
    args,
    validation_report: Mapping[str, dict],
    input_dir: Path,
    model_path: str,
) -> dict:
    input_hashes = {}
    for aa_name, di_name in NODE_FILES.values():
        for name in [aa_name, di_name]:
            input_hashes[name] = sha256_file(input_dir / name)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": {
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "bootstrap_replicates": args.bootstrap,
            "max_chunk_aa": args.max_chunk_aa,
            "device": args.device,
            "dtype": args.dtype,
            "model_argument": args.model,
            "resolved_model_path": model_path,
            "resume": args.resume,
        },
        "method_notes": {
            "aa_sampling": "Independent categorical draw at every alignment site from the 20 p_X columns; State='-' is not sampled because no p_- column exists.",
            "prostt5": "Deterministic greedy AA2fold generation; long sequences are divided into balanced chunks no longer than max_chunk_aa.",
            "empirical_uncertainty": "Plug-in Shannon entropy of sampled 3Di frequencies, in bits.",
            "direct_uncertainty": "Shannon entropy of row-normalized probabilities from the stored 3Di CSV, in bits.",
            "distribution_distance": "Jensen-Shannon divergence in bits.",
        },
        "validation": validation_report,
        "input_sha256": input_hashes,
        "versions": {
            package: installed_version(package)
            for package in ["numpy", "pandas", "matplotlib", "torch", "transformers", "huggingface-hub"]
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "inputs")
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--max-chunk-aa", type=int, default=500)
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
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_chunk_aa <= 0:
        raise ValueError("--max-chunk-aa must be positive")
    if args.bootstrap < 0:
        raise ValueError("--bootstrap cannot be negative")

    validation_report = validate_inputs(args.input_dir)
    print("Input validation:")
    print(json.dumps(validation_report, indent=2))
    if args.validate_only:
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    inputs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    sampled_sequences: Dict[str, List[str]] = {}
    for node_index, node in enumerate(NODE_FILES):
        aa, direct = load_node_inputs(args.input_dir, node)
        inputs[node] = (aa, direct)
        node_outdir = args.outdir / node
        node_outdir.mkdir(parents=True, exist_ok=True)
        sampled_sequences[node] = prepare_sampled_sequences(
            node,
            aa,
            node_outdir,
            args.n_samples,
            args.seed + node_index,
            args.resume,
        )

    torch_module, tokenizer, model, device, dtype, model_path = load_prostt5(
        args.model, args.device, args.dtype
    )
    site_tables: Dict[str, pd.DataFrame] = {}
    summaries: List[dict] = []
    high_disagreement_tables: List[pd.DataFrame] = []
    for node_index, node in enumerate(NODE_FILES):
        aa, direct = inputs[node]
        node_outdir = args.outdir / node
        structures = encode_with_checkpoint(
            node,
            sampled_sequences[node],
            node_outdir,
            tokenizer,
            model,
            device,
            torch_module,
            args.batch_size,
            args.max_chunk_aa,
            args.resume,
        )
        site_table, _, summary = analyze_node(
            node,
            aa,
            direct,
            structures,
            node_outdir,
            args.bootstrap,
            args.seed + 100_000 + node_index,
        )
        site_tables[node] = site_table
        summaries.append(summary)
        high_disagreement_tables.append(
            pd.read_csv(node_outdir / "high_disagreement_sites.tsv", sep="\t")
        )

    combined_sites = pd.concat(site_tables.values(), ignore_index=True)
    combined_sites.to_csv(args.outdir / "all_sites_uncertainty.tsv", sep="\t", index=False)
    pd.DataFrame(summaries).to_csv(args.outdir / "node_summary.tsv", sep="\t", index=False)
    pd.concat(high_disagreement_tables, ignore_index=True).to_csv(
        args.outdir / "high_disagreement_sites.tsv", sep="\t", index=False
    )
    create_plots(site_tables, args.outdir)
    manifest = build_manifest(args, validation_report, args.input_dir, model_path)
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nNode summary:")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nResults written to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
