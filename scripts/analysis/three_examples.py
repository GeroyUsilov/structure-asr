"""Three single-node examples of what the 3Di term does to a reconstruction.

Evaluation-only, post-inference (scientific rule 1). Each example is one
ancestral node in one run, never a sum over nodes. The baseline is the chain's
start, the AA maximum-likelihood (marginal MAP) reconstruction. "With 3Di" is
the best sample under S_λ = ℓAA + λ·ℓ3Di. Bars show the change in correctly
reconstructed residues out of 396, so positive means closer to the true ancestor.

Examples, picked from results/figures/cross_run_agreement.tsv (current λ sampler
only; the legacy α kernel is kept out, rule 3):
  - largest 3Di gain:   cal-1        μ=1 τ=1.5 λ=0.1  Node31
  - largest AA gain:    tlsample     μ=8 τ=0.5 λ=0.5  Node29
  - largest AA loss:    divergence   μ=2 τ=2   λ=0.05 Node29

Usage: .venv/bin/python scripts/analysis/three_examples.py
Writes results/figures/three_examples.png and three_examples_dark.png.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/figures")
SITES = 396
EXAMPLES = (
    ("Most improved 3Di", "cal-1", "mu_1__tau_1p5__seed_731024", "0.1", "Node31"),
    ("Most improved AA", "tlsample", "mu_8__tau_0p5__seed_731024", "0.5", "Node29"),
    ("3Di degraded it", "divergence", "mu_2__tau_2__seed_731024", "0.05", "Node29"),
)
THEME = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984",
              "better": "#2a78d6", "worse": "#e34948", "zero": "#6f6e69"},
    "dark": {"surface": "#1a1a19", "primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#8a8984",
             "better": "#3987e5", "worse": "#e66767", "zero": "#9b9a93"},
}


def load(run: str, condition: str, weight: str, node: str) -> dict[str, int]:
    path = Path("results/runpod") / run / "grid" / "evaluation" / "sample_summary.tsv"
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if (row["condition"], row["structural_lambda"], row["inferred_node"]) == (condition, weight, node):
                return {key: int(row[key]) for key in
                        ("start_aa_errors", "best_aa_errors", "start_3di_errors", "best_3di_errors")}
    raise KeyError(f"{run} {condition} λ={weight} {node} not in {path}")


def label(condition: str, weight: str, node: str, run: str) -> str:
    mu, tau = condition.split("__")[:2]
    mu, tau = mu.split("_")[1], tau.split("_")[1].replace("p", ".")
    return f"μ={mu}  τ={tau}  λ={weight}\n{node}  ·  {run} run"


def draw(mode: str) -> None:
    theme = THEME[mode]
    rows = [(title, run, cond, w, node, load(run, cond, w, node)) for title, run, cond, w, node in EXAMPLES]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, panels = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, layout="constrained",
                                  gridspec_kw={"width_ratios": [1, 1.6]})
    figure.patch.set_facecolor(theme["surface"])

    for panel, channel, name in ((panels[0], "aa", "Amino acid"), (panels[1], "3di", "3Di structure")):
        panel.set_facecolor(theme["surface"])
        reach = max(abs(r[5][f"start_{channel}_errors"] - r[5][f"best_{channel}_errors"]) for r in rows)
        for y, (_, _, _, _, _, errors) in enumerate(rows):
            start, best = errors[f"start_{channel}_errors"], errors[f"best_{channel}_errors"]
            delta = start - best  # residues gained: fewer errors is better
            colour = theme["better"] if delta > 0 else theme["worse"] if delta < 0 else theme["zero"]
            panel.barh(y, delta, height=0.42, color=colour, zorder=2)
            text = f"{delta:+d}".replace("-", "\u2212") + f"   ({start} \u2192 {best} errors)"
            # Labels always sit right of the bar's positive end (or of zero, for a
            # loss), so a negative label never runs into the row labels.
            panel.text(max(delta, 0) + 0.035 * reach, y, text, ha="left", va="center",
                       color=theme["primary"], fontsize=9.5, zorder=3)
        panel.set_xlim(-reach * 1.15, reach * 2.6)
        panel.axvline(0, color=theme["zero"], linewidth=1, zorder=1)
        panel.set_title(f"{name}: change in correct residues (of {SITES})", loc="left",
                        color=theme["primary"], fontsize=10.5)
        panel.tick_params(colors=theme["secondary"], length=0)
        panel.grid(axis="x", color=theme["muted"], alpha=0.25, linewidth=0.6, zorder=0)
        for spine in panel.spines.values():
            spine.set_visible(False)

    panels[0].set_yticks(range(len(rows)))
    panels[0].set_yticklabels([f"{title}\n{label(cond, w, node, run)}" for title, run, cond, w, node, _ in rows],
                              color=theme["primary"], fontsize=9.5)
    panels[0].invert_yaxis()
    figure.suptitle("What the 3Di term did to single ancestral nodes, vs the AA maximum-likelihood reconstruction",
                    x=0.01, ha="left", color=theme["primary"], fontsize=12)
    figure.text(0.01, -0.02,
                "Each row is one node in one run, not a sum. Blue = closer to the true ancestor, red = further. "
                "Baseline = chain start (AA marginal MAP); with 3Di = best sample under ℓAA + λ·ℓ3Di.",
                ha="left", color=theme["secondary"], fontsize=8.5)
    suffix = "" if mode == "light" else "_dark"
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / f"three_examples{suffix}.png", dpi=200, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    for mode in THEME:
        draw(mode)
    print(f"wrote {OUT}/three_examples.png and three_examples_dark.png")
