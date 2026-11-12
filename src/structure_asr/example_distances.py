"""Reproduce nine-condition AA/3Di ECDFs from the bundled tip FASTAs."""
import argparse
from pathlib import Path

from structure_asr.analysis.temperature_sweep import (
    MUS, TAUS, condition_slug, read_fasta, validate_records, pairwise_hamming, plot_ecdfs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-root", type=Path, default=Path("examples/ef_tu"))
    parser.add_argument("--outdir", type=Path, default=Path("results/tip_distances"))
    args = parser.parse_args()
    distances = {}
    for modality, empirical in (("aa", "QE1.9d_AA_SEQ.fasta"), ("3di", "QE1.9d_3Di.fasta")):
        distances[modality] = {}
        for mu in MUS:
            for tau in TAUS:
                slug = condition_slug(mu, tau)
                path = args.example_root / "tips" / modality / f"ef_tu_tips_{slug}_{modality}.fasta"
                records = read_fasta(path)
                validate_records(records, source=str(path), expected_count=39, expected_length=396)
                values, _ = pairwise_hamming(records, args.outdir / modality / f"{slug}.tsv")
                distances[modality][slug] = values
        path = args.example_root / "empirical" / empirical
        records = read_fasta(path)
        validate_records(records, source=str(path))
        values, _ = pairwise_hamming(records, args.outdir / modality / "empirical.tsv")
        distances[modality]["empirical"] = values
    plot_ecdfs(distances, args.outdir)
    print(args.outdir)


if __name__ == "__main__":
    main()
