"""Every sampled chain we have run, as change-in-agreement against the truth.

Evaluation-only, run after inference (scientific rule 1): every axis and the
colour are measured against ground truth, so nothing here may feed back into a
sampler, a proposal, or a choice of λ.

Axes, both in percentage points of the node's sites, positive = closer to the
true ancestor:

    Δ = (start_errors - best_errors) / sites * 100

The baseline (`start`) is the AA-model maximum-likelihood reconstruction, which
is also the sampler's start; the comparison point is the best-scoring sample
under S_λ = ℓAA + λ·ℓ3Di.

**Colour is realized divergence, not μ.** μ is an attempt-rate knob whose
meaning changes with τ: at τ=0.5 most attempts redraw the residue already
present, so μ=8 τ=0.5 and μ=1 τ=1.5 reach similar leaf–root *identity* at very
different substitution counts. What is comparable across cells is the realized
number of accepted AA substitutions per site on the path from the ancestral node
to its descendant tips, averaged over those tips, counted from the simulation's
own event log. Multiple hits at one site count once each, so this is a
substitutions/site branch length and can exceed 1.

Shape is τ; one panel per λ.

Provenance (scientific rule 3) is kept distinct rather than pooled:

- The four grid runs share the batched, uncertainty-weighted kernel and the λ
  score scale, so they share the λ panels. The two calibration runs are the same
  cell and seed on different GPUs — they are plotted as the replicates they are,
  not merged.
- The legacy run (`results/reproduced`) used the legacy batch-1 kernel, the
  legacy acceptance rule and the α scale (0.5·ℓAA + 0.5·ℓ3Di, α=0.5), on a
  different reconstruction. It is **not** placed on a λ panel; it gets its own,
  in neutral ink, and its realized divergence is unmeasured because that run's
  event log was not retained.

Usage: .venv/bin/python scripts/analysis/cross_run_scatter.py [--outdir DIR]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

# Grid runs that share the λ scale: label -> directory holding the grid output.
GRID_RUNS = (
    ("divergence", Path("results/runpod/divergence/grid")),
    ("tau-low λ scan", Path("results/runpod/tlsample/grid")),
    ("calibration RTX4090", Path("results/runpod/cal-1/grid")),
    ("calibration A100", Path("results/runpod/cal-a100/grid")),
)
LEGACY = Path("results/reproduced/legacy_joint_comparison.tsv")
LEGACY_TAU = {"mu1_tau1p5": 1.5, "mu1_tau2": 2.0}
LEGACY_SITES = 396

# Sequential blue, light->dark. Discrete marks, so the light mode starts at the
# ordinal floor (step 250) rather than the near-surface step 100.
RAMP_LIGHT = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
RAMP_DARK = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
TAU_MARKERS = {0.5: "o", 1.5: "s", 2.0: "^"}
THEME = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984",
              "neutral": "#6f6e69", "ramp": RAMP_LIGHT},
    "dark": {"surface": "#1a1a19", "primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#8a8984",
             "neutral": "#9b9a93", "ramp": RAMP_DARK},
}


def realized_divergence(events: Path, sites: int) -> dict[str, float]:
    """Mean substitutions/site from each node to the tips below it.

    Topology comes from the event log's parent/child pairs, which cover every
    edge; an edge with no accepted change still appears, with a count of zero.
    """
    subs: dict[tuple[str, str], int] = {}
    children: dict[str, list[str]] = defaultdict(list)
    seen_child: set[str] = set()
    with open(events) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            edge = (row["parent"], row["child"])
            subs.setdefault(edge, 0)
            if row["child"] not in seen_child:
                seen_child.add(row["child"])
                children[row["parent"]].append(row["child"])
            if row["accepted"] == "True" and row["old_state"] != row["new_state"]:
                subs[edge] += 1

    distances: dict[str, float] = {}
    for node in list(children) + sorted(seen_child):
        totals: list[int] = []
        stack = [(node, 0)]
        while stack:
            current, accumulated = stack.pop()
            if current not in children:
                totals.append(accumulated)
                continue
            for child in children[current]:
                stack.append((child, accumulated + subs[(current, child)]))
        if totals:
            distances[node] = sum(totals) / len(totals) / sites
    return distances


def load_grids() -> list[dict]:
    records: list[dict] = []
    for label, outdir in GRID_RUNS:
        summary = outdir / "evaluation" / "sample_summary.tsv"
        if not summary.exists():
            print(f"  skipping {label}: no sample_summary.tsv")
            continue
        mapping: dict[tuple[str, str], str] = {}
        with open(outdir / "evaluation" / "node_mapping.tsv") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["modality"] == "aa" and row["matched"] == "True":
                    mapping[(row["condition"], row["inferred_node"])] = row["ground_truth_node"]

        divergences: dict[str, dict[str, float]] = {}
        with open(summary) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                condition, sites = row["condition"], int(row["sites"])
                if condition not in divergences:
                    events = outdir / condition / "simulation" / "events.tsv"
                    divergences[condition] = realized_divergence(events, sites) if events.exists() else {}
                true_node = mapping.get((condition, row["inferred_node"]))
                records.append(
                    {
                        "run": label,
                        "scale": "lambda",
                        "condition": condition,
                        "mu": float(row["mutation_scale"]),
                        "tau": float(row["tau"]),
                        "weight": float(row["structural_lambda"]),
                        "node": row["inferred_node"],
                        "sites": sites,
                        "divergence": divergences[condition].get(true_node),
                        "aa": (int(row["start_aa_errors"]) - int(row["best_aa_errors"])) / sites * 100,
                        "three_di": (int(row["start_3di_errors"]) - int(row["best_3di_errors"])) / sites * 100,
                    }
                )
    return records


def load_legacy() -> list[dict]:
    if not LEGACY.exists():
        return []
    records = []
    with open(LEGACY) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            records.append(
                {
                    "run": "legacy (batch-1, α=0.5)",
                    "scale": "alpha",
                    "condition": row["condition"],
                    "mu": 1.0,
                    "tau": LEGACY_TAU.get(row["condition"], float("nan")),
                    "weight": 0.5,
                    "node": row["node"],
                    "sites": LEGACY_SITES,
                    "divergence": None,
                    "aa": (int(row["aa_start_errors"]) - int(row["aa_best_errors"])) / LEGACY_SITES * 100,
                    "three_di": (int(row["3di_start_errors"]) - int(row["3di_best_errors"])) / LEGACY_SITES * 100,
                }
            )
    return records


def write_table(records: list[dict], path: Path) -> None:
    """Table view: the light ramp's pale steps sit under 3:1 on the surface."""
    fields = ["run", "scale", "condition", "mu", "tau", "weight", "node", "sites", "divergence", "aa", "three_di"]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["run", "score_scale", "condition", "mutation_scale", "tau", "lambda_or_alpha", "node",
             "sites", "subs_per_site_to_tips", "delta_aa_pct", "delta_3di_pct"]
        )
        for r in sorted(records, key=lambda r: (r["run"], r["weight"], r["condition"], r["node"])):
            writer.writerow(
                [r[f] if not isinstance(r[f], float) else f"{r[f]:g}" for f in fields[:8]]
                + [f"{r['divergence']:.3f}" if r["divergence"] is not None else "",
                   f"{r['aa']:+.3f}", f"{r['three_di']:+.3f}"]
            )


def draw(records: list[dict], mode: str, path: Path) -> None:
    ink = THEME[mode]
    cmap = LinearSegmentedColormap.from_list("divergence", ink["ramp"])
    grids = [r for r in records if r["scale"] == "lambda"]
    legacy = [r for r in records if r["scale"] == "alpha"]
    measured = [r["divergence"] for r in grids if r["divergence"] is not None]
    norm = LogNorm(vmin=min(measured), vmax=max(measured))

    lambdas = sorted({r["weight"] for r in grids})
    panels = [("lambda", lam) for lam in lambdas] + ([("alpha", 0.5)] if legacy else [])
    # Cropped to +/-5 pp on both axes. A handful of chains (all of them large
    # 3Di losses in the un-folded tau>=1.5 cells) fall outside and are dropped
    # rather than clipped, so no mark sits at a position it does not hold; each
    # panel states how many it is not showing. Equal scales on both axes, so a
    # 45-degree slope really is a one-for-one trade.
    crop = 5.0

    columns = 4
    rows = -(-(len(panels) + 1) // columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(16, 4.6 * rows), sharex=True, sharey=True, layout="constrained"
    )
    figure.patch.set_facecolor(ink["surface"])
    flat = axes.ravel()

    for panel in flat[len(panels):]:
        panel.set_facecolor(ink["surface"])
        panel.set_axis_off()

    for panel, (scale, weight) in zip(flat, panels):
        points = [r for r in (grids if scale == "lambda" else legacy) if r["weight"] == weight]
        panel.set_facecolor(ink["surface"])
        panel.axhline(0, color=ink["muted"], linewidth=1, zorder=1)
        panel.axvline(0, color=ink["muted"], linewidth=1, zorder=1)
        panel.grid(True, color=ink["muted"], alpha=0.16, linewidth=0.6, zorder=0)
        panel.set_axisbelow(True)
        for spine in panel.spines.values():
            spine.set_visible(False)
        panel.tick_params(colors=ink["secondary"], labelsize=9, length=0)
        panel.set_xlim(-crop, crop)
        panel.set_ylim(-crop, crop)
        panel.set_aspect("equal")
        title = f"λ = {weight:g}" if scale == "lambda" else "legacy kernel, α = 0.5"
        panel.set_title(title, color=ink["primary"], fontsize=11.5, pad=8, loc="left")

        inside = [r for r in points if abs(r["aa"]) <= crop and abs(r["three_di"]) <= crop]
        for tau, marker in TAU_MARKERS.items():
            chosen = [r for r in inside if r["tau"] == tau]
            if not chosen:
                continue
            if scale == "alpha":
                panel.scatter(
                    [r["aa"] for r in chosen], [r["three_di"] for r in chosen],
                    s=70, marker=marker, facecolors="none", edgecolors=ink["neutral"],
                    linewidths=1.6, zorder=4,
                )
            else:
                panel.scatter(
                    [r["aa"] for r in chosen], [r["three_di"] for r in chosen],
                    s=70, marker=marker, c=[r["divergence"] for r in chosen], cmap=cmap, norm=norm,
                    edgecolors=ink["surface"], linewidths=1.2, zorder=4,
                )
        stuck = sum(1 for r in points if r["aa"] == 0 and r["three_di"] == 0)
        note = f"{len(points) - stuck} of {len(points)} chains moved"
        dropped = len(points) - len(inside)
        if dropped:
            note += f"  ·  {dropped} outside ±{crop:g} pp, not shown"
        panel.annotate(
            note, xy=(0.97, 0.045), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=8.5, color=ink["muted"],
        )

    flat[0].annotate("both improve →", xy=(0.97, 0.965), xycoords="axes fraction", ha="right", va="top",
                     fontsize=8.5, color=ink["muted"], style="italic")
    flat[0].annotate("← structure gained, AA lost", xy=(0.03, 0.965), xycoords="axes fraction",
                     ha="left", va="top", fontsize=8.5, color=ink["muted"], style="italic")

    for index in range(0, len(panels), columns):
        flat[index].set_ylabel("Δ 3Di agreement (pp)", color=ink["secondary"], fontsize=10)
    for index in range(max(0, len(flat) - columns), len(flat)):
        flat[index].set_xlabel("Δ AA agreement (pp)", color=ink["secondary"], fontsize=10)
    # The row above the empty slot still needs its own x labels.
    for index in range(len(panels) - 1, len(panels)):
        flat[index].set_xlabel("Δ AA agreement (pp)", color=ink["secondary"], fontsize=10)
        flat[index].tick_params(labelbottom=True)

    handles = [
        Line2D([], [], marker=marker, linestyle="none", markersize=9,
               markerfacecolor=ink["neutral"], markeredgecolor=ink["surface"], label=f"τ = {tau:g}")
        for tau, marker in TAU_MARKERS.items()
    ]
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=9, markerfacecolor="none",
                          markeredgecolor=ink["neutral"], label="legacy α run (divergence\nnot measured)"))
    legend = flat[len(panels)].legend(
        handles=handles, loc="center", frameon=False, fontsize=10.5,
        labelcolor=ink["secondary"], handletextpad=0.6, labelspacing=1.0,
    )
    legend.set_zorder(6)

    bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.02, pad=0.012, aspect=38
    )
    bar.set_label("realized divergence: AA substitutions per site, node → its tips",
                  color=ink["secondary"], fontsize=9.5)
    bar.ax.tick_params(colors=ink["secondary"], labelsize=8.5, length=0)
    bar.outline.set_visible(False)

    figure.suptitle(
        "Change in agreement with the true ancestor, every chain we have sampled",
        color=ink["primary"], fontsize=15, x=0.006, ha="left",
    )
    figure.text(
        0.006, 0.962,
        "Baseline is the AA-model ML reconstruction; positive is closer to the truth on that channel. Colour is measured divergence, not μ —\n"
        "at τ=0.5 most attempts redraw the residue already there. Cropped to ±5 pp on both axes; chains outside are dropped, not clipped, and\n"
        "each panel counts them. The legacy panel used a different kernel and score scale, so it is not comparable point-for-point.",
        color=ink["secondary"], fontsize=9.5, ha="left", va="top", linespacing=1.5,
    )
    figure.get_layout_engine().set(rect=(0, 0.025, 1, 0.875))
    figure.savefig(path, dpi=200, facecolor=ink["surface"])
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/figures"))
    arguments = parser.parse_args()
    arguments.outdir.mkdir(parents=True, exist_ok=True)

    records = load_grids() + load_legacy()
    write_table(records, arguments.outdir / "cross_run_agreement.tsv")
    draw(records, "light", arguments.outdir / "cross_run_scatter.png")
    draw(records, "dark", arguments.outdir / "cross_run_scatter_dark.png")

    moved = [r for r in records if r["aa"] or r["three_di"]]
    both = [r for r in moved if r["aa"] > 0 and r["three_di"] > 0]
    aa_up = [r for r in moved if r["aa"] > 0]
    print(f"{len(records)} chains from {len({r['run'] for r in records})} runs; {len(moved)} moved")
    print(f"  AA improved:        {len(aa_up)}")
    print(f"  both improved:      {len(both)}")
    for r in sorted(both, key=lambda r: -r["aa"]):
        print(f"    {r['run']:>22} λ/α={r['weight']:g} {r['condition']} {r['node']}: "
              f"AA {r['aa']:+.2f} pp, 3Di {r['three_di']:+.2f} pp")
    print(f"wrote {arguments.outdir / 'cross_run_scatter.png'}")


if __name__ == "__main__":
    main()
