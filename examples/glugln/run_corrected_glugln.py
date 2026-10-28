#!/usr/bin/env python3
"""Run the corrected Glu/Gln cross-reconstruction uncertainty analysis.

The AA and 3Di reconstructions come from different trees, so their internal-node
numbers intentionally differ: AAseq Node168 is compared with 3Di Node16.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import uncertainty_core as core


PAIR_LABEL = "GluGln_AA168_3Di16"
AA_NODE = "Node168"
DI_NODE = "Node16"
AA_FILENAME = "AAseq_Node168.csv"
DI_FILENAME = "3Di_Node16.csv"
REQUIRED_COLUMNS = {"Node", "Site", "State"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "inputs")
    parser.add_argument("--outdir", type=Path, default=Path("results/glugln_uncertainty"))
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--max-chunk-aa", type=int, default=500)
    parser.add_argument("--model", default="Rostlab/ProstT5")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.n_samples <= 0 or args.batch_size <= 0 or args.max_chunk_aa <= 0:
        parser.error("sample count, batch size, and max chunk length must be positive")
    if args.bootstrap < 0:
        parser.error("bootstrap replicate count cannot be negative")
    return args


def load_and_validate(input_dir: Path):
    aa_path = input_dir / AA_FILENAME
    di_path = input_dir / DI_FILENAME
    aa = pd.read_csv(aa_path)
    direct = pd.read_csv(di_path)

    for filename, frame, expected_node in [
        (AA_FILENAME, aa, AA_NODE),
        (DI_FILENAME, direct, DI_NODE),
    ]:
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
        columns = core.probability_columns(frame)
        core.normalized_probabilities(frame, columns)
        labels = set(frame["Node"].astype(str))
        if labels != {expected_node}:
            raise ValueError(
                f"{filename} contains node labels {sorted(labels)}, expected {expected_node}"
            )
        if frame.empty:
            raise ValueError(f"{filename} is empty")

    aa = aa.reset_index(drop=True)
    direct = direct.reset_index(drop=True)
    if len(aa) != len(direct):
        raise ValueError("AA Node168 and 3Di Node16 have different site counts")
    if not aa["Site"].equals(direct["Site"]):
        raise ValueError("AA Node168 and 3Di Node16 Site columns do not match")

    report = {
        "pair_label": PAIR_LABEL,
        "aa_source_node": AA_NODE,
        "direct_3di_source_node": DI_NODE,
        "sites": len(aa),
        "site_min": int(aa["Site"].min()),
        "site_max": int(aa["Site"].max()),
        "aa_probability_columns": len(core.probability_columns(aa)),
        "direct_3di_probability_columns": len(core.probability_columns(direct)),
        "aa_provided_state_gaps": int((aa["State"].astype(str) == "-").sum()),
        "direct_3di_provided_state_gaps": int(
            (direct["State"].astype(str) == "-").sum()
        ),
    }
    return aa, direct, report, aa_path, di_path


def create_single_pair_plots(site_table: pd.DataFrame, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    empirical_color = "#2563EB"
    direct_color = "#D97706"
    aa_color = "#6B7280"
    divergence_color = "#7C3AED"
    ceiling = math.log2(20)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bins = np.linspace(0, ceiling, 32)
    ax.hist(
        site_table["empirical_3di_entropy_bits"], bins=bins, alpha=0.58,
        color=empirical_color, label="AA Node168 samples → ProstT5",
    )
    ax.hist(
        site_table["direct_3di_entropy_bits"], bins=bins, alpha=0.58,
        color=direct_color, label="Stored 3Di Node16 posterior",
    )
    ax.set(xlabel="3Di entropy (bits)", ylabel="Number of sites",
           title="Glu/Gln sitewise 3Di uncertainty distributions")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "entropy_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    x = site_table["direct_3di_entropy_bits"].to_numpy()
    y = site_table["empirical_3di_entropy_bits"].to_numpy()
    rho = core.safe_spearman(x, y)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(x, y, s=17, alpha=0.58, color=empirical_color, linewidths=0)
    ax.plot([0, ceiling], [0, ceiling], "--", color="#111827", lw=1)
    ax.text(0.04, 0.95, f"Spearman ρ = {rho:.3f}", transform=ax.transAxes, va="top")
    ax.set(
        xlabel="Stored 3Di Node16 entropy (bits)",
        ylabel="AA Node168 samples → ProstT5 entropy (bits)",
        title="Sitewise uncertainty comparison",
        xlim=(-0.05, ceiling + 0.05), ylim=(-0.05, ceiling + 0.05),
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "entropy_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    sites = site_table["Site"]
    ax.plot(sites, site_table["aa_entropy_bits"], color=aa_color, lw=1,
            alpha=0.72, label="AA Node168")
    ax.plot(sites, site_table["empirical_3di_entropy_bits"], color=empirical_color,
            lw=1.2, label="AA samples → ProstT5")
    ax.plot(sites, site_table["direct_3di_entropy_bits"], color=direct_color,
            lw=1.2, label="Stored 3Di Node16")
    ax.set(xlabel="Alignment site", ylabel="Entropy (bits)",
           title="Uncertainty across Glu/Gln alignment sites")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(outdir / "per_site_uncertainty.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(sites, site_table["js_divergence_bits"], color=divergence_color, lw=1.2)
    ax.set(xlabel="Alignment site", ylabel="Jensen–Shannon divergence (bits)",
           title="Distributional disagreement: sampled versus stored 3Di posterior",
           ylim=(-0.01, 1.01))
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(outdir / "js_divergence_by_site.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    aa, direct, validation, aa_path, di_path = load_and_validate(args.input_dir)
    print(json.dumps(validation, indent=2), flush=True)
    if args.validate_only:
        print("Validation passed; no model was loaded.", flush=True)
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    pair_outdir = args.outdir / PAIR_LABEL
    pair_outdir.mkdir(parents=True, exist_ok=True)

    sequences = core.prepare_sampled_sequences(
        PAIR_LABEL, aa, pair_outdir, args.n_samples, args.seed, args.resume
    )
    torch_module, tokenizer, model, device, dtype, model_path = core.load_prostt5(
        args.model, args.device, args.dtype
    )
    structures = core.encode_with_checkpoint(
        PAIR_LABEL,
        sequences,
        pair_outdir,
        tokenizer,
        model,
        device,
        torch_module,
        args.batch_size,
        args.max_chunk_aa,
        args.resume,
    )
    site_table, empirical_table, summary = core.analyze_node(
        PAIR_LABEL,
        aa,
        direct,
        structures,
        pair_outdir,
        args.bootstrap,
        args.seed + 1,
    )
    summary.update({"aa_source_node": AA_NODE, "direct_3di_source_node": DI_NODE})
    pd.DataFrame([summary]).to_csv(args.outdir / "node_summary.tsv", sep="\t", index=False)
    site_table.to_csv(args.outdir / "all_sites_uncertainty.tsv", sep="\t", index=False)
    shutil.copy2(
        pair_outdir / "high_disagreement_sites.tsv",
        args.outdir / "high_disagreement_sites.tsv",
    )
    create_single_pair_plots(site_table, args.outdir)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "intentional_cross_tree_mapping": {
            "aa_posterior_file": AA_FILENAME,
            "aa_source_node": AA_NODE,
            "direct_3di_posterior_file": DI_FILENAME,
            "direct_3di_source_node": DI_NODE,
        },
        "parameters": {
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "bootstrap_replicates": args.bootstrap,
            "max_chunk_aa": args.max_chunk_aa,
            "device": str(device),
            "dtype": str(dtype),
            "model_argument": args.model,
            "resolved_model_path": model_path,
            "resume": args.resume,
        },
        "method_notes": {
            "aa_sampling": "Independent categorical draws per site from AA Node168 p_X columns.",
            "prostt5": "Deterministic greedy AA2fold generation.",
            "empirical_uncertainty": "Shannon entropy of sampled ProstT5 3Di frequencies, in bits.",
            "direct_uncertainty": "Shannon entropy of row-normalized stored 3Di Node16 probabilities, in bits.",
            "distribution_distance": "Jensen-Shannon divergence in bits.",
        },
        "validation": validation,
        "input_sha256": {
            AA_FILENAME: core.sha256_file(aa_path),
            DI_FILENAME: core.sha256_file(di_path),
        },
        "package_versions": {
            name: core.installed_version(name)
            for name in ["numpy", "pandas", "matplotlib", "torch", "transformers", "sentencepiece"]
        },
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nCorrected analysis complete.", flush=True)
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)
    print(f"Results: {args.outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
