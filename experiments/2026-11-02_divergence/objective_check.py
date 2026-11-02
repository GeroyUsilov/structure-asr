"""Evaluation-only diagnostic: joint objective of truth vs the AA-MAP start."""
import csv, sys
from pathlib import Path
from structure_asr.posteriors import load_state, sequence_log_probability

def fa(p):
    d, k = {}, None
    for l in open(p):
        l = l.strip()
        if l.startswith(">"): k = l[1:].split()[0]; d[k] = ""
        elif k: d[k] += l
    return d

grid = Path("results/runpod/divergence/grid")
mapping = {}
for r in csv.DictReader(open(grid / "evaluation/node_mapping.tsv"), delimiter="\t"):
    if r["matched"] == "True":
        mapping[(r["condition"], r["modality"], r["ground_truth_node"])] = r["inferred_node"]
print("cell\tnode\tAA errs\tdLogP_AA(truth-MAP)\tdLogP_3Di(f(truth)-f(MAP))\tlambda where truth wins (T irrelevant)")
for cell in ["mu_1__tau_1p5__seed_731024", "mu_1__tau_2__seed_731024", "mu_2__tau_2__seed_731024", "mu_4__tau_1p5__seed_731024"]:
    c = grid / cell
    aa = load_state(c / "asr/aa.state", 5e-6); di = load_state(c / "asr/3di.state", 5e-6)
    truth_aa = fa(c / "simulation/ground_truth_nodes.fasta"); truth_3di = fa(c / "truth/ground_truth_nodes_3di.fasta")
    start_aa = fa(c / "sample/lambda_0p02/aa_map_start_aa.fasta"); start_3di = fa(c / "sample/lambda_0p02/aa_map_start_3di.fasta")
    for gt in ["Internal_0001", "Internal_0029", "Internal_0034"]:
        an = mapping[(cell, "aa", gt)]; dn = mapping[(cell, "3di", gt)]
        key = next(k for k in start_aa if k.split("|")[0].endswith(an) or k == an or an in k.split("_"))
        sa, s3 = start_aa[key], start_3di[next(k for k in start_3di if an in k)]
        d_aa = sequence_log_probability(aa[an], truth_aa[gt]) - sequence_log_probability(aa[an], sa)
        d_3 = sequence_log_probability(di[dn], truth_3di[gt]) - sequence_log_probability(di[dn], s3)
        errs = sum(a != b for a, b in zip(sa, truth_aa[gt]))
        need = (f"lambda > {-d_aa / d_3:.2f}" if d_3 > 0 else "never") if errs else "-"
        print(f"{cell[:13]}\t{an}\t{errs}\t{d_aa:.1f}\t{d_3:.1f}\t{need}")
