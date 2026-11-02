"""Evaluation-only: does lambda rank truth above AA-MAP in the tau=0.5 cells?"""
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

grid = Path(sys.argv[1])
cells = sorted(p.name for p in grid.glob("mu_*"))
mapping = {}
for r in csv.DictReader(open(grid / "evaluation/node_mapping.tsv"), delimiter="\t"):
    if r["matched"] == "True":
        mapping[(r["condition"], r["modality"], r["ground_truth_node"])] = r["inferred_node"]

def map_sequence(posterior):
    return "".join(posterior.alphabet[int(row.argmax())] for row in posterior.probabilities)

print("cell\tnode\tAA errs(MAP)\t3Di errs(MAP enc)\tdLogP_AA\tdLogP_3Di\tlambda for truth to win")
for cell in cells:
    c = grid / cell
    aa = load_state(c / "asr/aa.state", 5e-6); di = load_state(c / "asr/3di.state", 5e-6)
    truth_aa = fa(c / "simulation/ground_truth_nodes.fasta")
    truth_3di = fa(c / "truth/ground_truth_nodes_3di.fasta")
    for gt in ["Internal_0001", "Internal_0029", "Internal_0034"]:
        an = mapping[(cell, "aa", gt)]; dn = mapping[(cell, "3di", gt)]
        start_aa = map_sequence(aa[an]); start_3di = map_sequence(di[dn])
        d_aa = sequence_log_probability(aa[an], truth_aa[gt]) - sequence_log_probability(aa[an], start_aa)
        d_3 = sequence_log_probability(di[dn], truth_3di[gt]) - sequence_log_probability(di[dn], start_3di)
        errs = sum(x != y for x, y in zip(start_aa, truth_aa[gt]))
        e3 = sum(x != y for x, y in zip(start_3di, truth_3di[gt]))
        need = f"> {-d_aa / d_3:.2f}" if d_3 > 0 else "never"
        print(f"{cell[:14]}\t{an}\t{errs}\t{e3}\t{d_aa:.1f}\t{d_3:.1f}\t{need}")
