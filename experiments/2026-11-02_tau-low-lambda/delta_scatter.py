"""Change in agreement with the true ancestor, AA against 3Di.

Evaluation-only, run after inference (scientific rule 1): both axes are measured
against ground truth, so nothing here may feed back into a sampler or a choice
of lambda.

Baseline is the AA-model maximum-likelihood reconstruction (the IQ-TREE marginal
MAP, which is also the sampler's start); the comparison point is the
best-scoring sample under S_lambda = lAA + lambda * l3Di. Both axes are
percentage points of the 396 sites per node:

    delta% = (start_errors - best_errors) / sites * 100

so positive is *more* agreement with the truth, on both axes. The quadrants are
the whole point: upper right is a real win on both channels, lower right is
AA bought with structure, upper left is structure bought with AA, lower left is
a loss on both.

Small multiples by lambda rather than one panel, because every chain at
lambda <= 0.02 sits exactly on the origin and would overplot the moved points.
mu carries both hue and marker shape (composite encoding), so the series stay
distinguishable in CVD and in print.

Usage: .venv/bin/python experiments/.../delta_scatter.py [grid outdir]
Writes delta_scatter.png, delta_scatter_dark.png and delta_agreement.tsv beside
this script.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated default palette; the first three
# validate all-pairs in both modes, which is what a scatter needs.
SERIES = {
    2.0: {"light": "#2a78d6", "dark": "#3987e5", "marker": "o", "label": "μ=2"},
    4.0: {"light": "#eb6834", "dark": "#d95926", "marker": "s", "label": "μ=4"},
    8.0: {"light": "#1baf7a", "dark": "#199e70", "marker": "^", "label": "μ=8"},
}
THEME = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984"},
    "dark": {"surface": "#1a1a19", "primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#8a8984"},
}
PANELS = (0.1, 0.5, 1.0)


def load(outdir: Path) -> list[dict]:
    """Per node/lambda deltas in percentage points, positive = improvement."""
    path = outdir / "evaluation" / "sample_summary.tsv"
    records = []
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sites = int(row["sites"])
            records.append(
                {
                    "mu": float(row["mutation_scale"]),
                    "lam": float(row["structural_lambda"]),
                    "node": row["inferred_node"],
                    "sites": sites,
                    "aa": (int(row["start_aa_errors"]) - int(row["best_aa_errors"])) / sites * 100,
                    "three_di": (int(row["start_3di_errors"]) - int(row["best_3di_errors"])) / sites * 100,
                }
            )
    return records


def write_table(records: list[dict], path: Path) -> None:
    """The table view the contrast relief rule obliges for the light palette."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["mutation_scale", "structural_lambda", "node", "sites", "delta_aa_pct", "delta_3di_pct"])
        for r in sorted(records, key=lambda r: (r["mu"], r["lam"], r["node"])):
            writer.writerow(
                [f"{r['mu']:g}", f"{r['lam']:g}", r["node"], r["sites"], f"{r['aa']:+.3f}", f"{r['three_di']:+.3f}"]
            )


CANDIDATE_OFFSETS = ((9, -3), (9, 5), (9, -12), (-26, -3), (-26, 5), (9, 12))


def _offset(record: dict, placed: list[tuple[float, float]], limit: float) -> tuple[int, int]:
    """First label offset that does not land on an already-placed label.

    Points coincide often here (several λ give the same best sample), so a fixed
    offset would stack labels illegibly in the dense panels.
    """
    scale = limit / 3.8
    for dx, dy in CANDIDATE_OFFSETS:
        spot = (record["aa"] + dx * 0.045 * scale, record["three_di"] + dy * 0.045 * scale)
        if all(abs(spot[0] - x) > 0.62 * scale or abs(spot[1] - y) > 0.28 * scale for x, y in placed):
            placed.append(spot)
            return dx, dy
    placed.append((record["aa"], record["three_di"]))
    return CANDIDATE_OFFSETS[0]


def draw(records: list[dict], mode: str, path: Path) -> None:
    ink = THEME[mode]
    limit = max(abs(r[key]) for r in records for key in ("aa", "three_di")) * 1.25
    stuck = sum(1 for r in records if r["lam"] <= 0.02)

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 5.1), sharex=True, sharey=True)
    figure.patch.set_facecolor(ink["surface"])

    for panel, lam in zip(axes, PANELS):
        placed: list[tuple[float, float]] = []
        panel.set_facecolor(ink["surface"])
        panel.axhline(0, color=ink["muted"], linewidth=1, zorder=1)
        panel.axvline(0, color=ink["muted"], linewidth=1, zorder=1)
        panel.grid(True, color=ink["muted"], alpha=0.18, linewidth=0.6, zorder=0)
        panel.set_axisbelow(True)
        for spine in panel.spines.values():
            spine.set_visible(False)
        panel.tick_params(colors=ink["secondary"], labelsize=9, length=0)
        panel.set_xlim(-limit, limit)
        panel.set_ylim(-limit, limit)
        panel.set_aspect("equal")
        panel.set_title(f"λ = {lam:g}", color=ink["primary"], fontsize=12, pad=10, loc="left")

        for mu, style in SERIES.items():
            points = [r for r in records if r["mu"] == mu and r["lam"] == lam]
            panel.scatter(
                [r["aa"] for r in points],
                [r["three_di"] for r in points],
                s=110,
                marker=style["marker"],
                color=style[mode],
                edgecolors=ink["surface"],
                linewidths=2,
                zorder=4,
                label=style["label"] if lam == PANELS[0] else None,
            )
            # Selective direct labels: only the chains that actually moved.
            for r in points:
                if r["aa"] == 0 and r["three_di"] == 0:
                    continue
                panel.annotate(
                    r["node"].replace("Node", "N"),
                    (r["aa"], r["three_di"]),
                    textcoords="offset points",
                    xytext=_offset(r, placed, limit),
                    fontsize=8.5,
                    color=ink["secondary"],
                    zorder=5,
                )

    axes[0].set_ylabel("Δ 3Di agreement with truth (pp)", color=ink["secondary"], fontsize=10)

    for panel, text, xy in (
        (axes[0], "both improve", (0.97, 0.97)),
        (axes[0], "structure gained,\nAA lost", (0.03, 0.97)),
        (axes[0], "both lost", (0.03, 0.03)),
    ):
        panel.annotate(
            text,
            xy=xy,
            xycoords="axes fraction",
            ha="right" if xy[0] > 0.5 else "left",
            va="top" if xy[1] > 0.5 else "bottom",
            fontsize=8.5,
            color=ink["muted"],
            style="italic",
        )

    legend = axes[0].legend(
        loc="lower right", frameon=False, fontsize=9.5, labelcolor=ink["secondary"], handletextpad=0.4
    )
    legend.set_zorder(6)

    both = sum(1 for r in records if r["aa"] > 0 and r["three_di"] > 0)
    figure.suptitle(
        f"Structural gains come out of amino-acid accuracy: {both} of 45 chains improved on both",
        color=ink["primary"],
        fontsize=14,
        x=0.012,
        ha="left",
        y=0.985,
    )
    figure.text(
        0.012,
        0.915,
        f"τ=0.5 cells, 3 ancestral nodes × 3 μ. Positive = closer to the true ancestor. "
        f"λ=0 and λ=0.02 have no panel: all {stuck} of those chains sit exactly on the origin.",
        color=ink["secondary"],
        fontsize=9.5,
        ha="left",
    )
    figure.text(
        0.5,
        0.03,
        "Δ AA agreement with truth (pp)    — baseline: AA-model ML reconstruction —",
        color=ink["secondary"],
        fontsize=10,
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.88))
    figure.savefig(path, dpi=200, facecolor=ink["surface"])
    plt.close(figure)


def main(outdir: Path) -> None:
    records = load(outdir)
    here = Path(__file__).parent
    write_table(records, here / "delta_agreement.tsv")
    draw(records, "light", here / "delta_scatter.png")
    draw(records, "dark", here / "delta_scatter_dark.png")
    moved = [r for r in records if r["aa"] or r["three_di"]]
    both = [r for r in moved if r["aa"] > 0 and r["three_di"] > 0]
    print(f"{len(records)} chains, {len(moved)} moved, {len(both)} improved on both channels:")
    for r in both:
        print(f"  μ={r['mu']:g} λ={r['lam']:g} {r['node']}: AA {r['aa']:+.2f} pp, 3Di {r['three_di']:+.2f} pp")
    print(f"wrote {here / 'delta_scatter.png'}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/runpod/tlsample/grid"))
