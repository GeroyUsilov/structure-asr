"""Plot direct AA and 3Di ASR entropies for the corrected Glu/Gln node pair."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SOURCE = Path('/Users/christopher/Desktop/aleks_claude/runpod_glugln_aa168_3di16_uncertainty/inputs')
REFERENCE = Path('/Users/christopher/Desktop/aleks_claude/results_glugln/GluGln_AA168_3Di16/site_uncertainty.tsv')
OUTPUT = Path(__file__).resolve().parent / 'glugln_model_entropy_figures'


def read_entropy(name, node):
    with (SOURCE / name).open() as handle:
        reader = csv.DictReader(handle)
        columns = [c for c in reader.fieldnames if c.startswith('p_')]
        assert len(columns) == 20
        rows = sorted(reader, key=lambda r: int(r['Site']))
    assert all(r['Node'] == node for r in rows)
    sites = np.array([int(r['Site']) for r in rows])
    p = np.array([[float(r[c]) for c in columns] for r in rows])
    assert np.isfinite(p).all() and (p >= 0).all() and (p.sum(axis=1) > 0).all()
    p /= p.sum(axis=1, keepdims=True)
    logs = np.zeros_like(p)
    np.log2(p, out=logs, where=p > 0)
    return sites, -(p * logs).sum(axis=1)


def save(fig, stem):
    fig.savefig(OUTPUT / f'{stem}.png', dpi=220, facecolor='white')
    fig.savefig(OUTPUT / f'{stem}.pdf', facecolor='white')
    plt.close(fig)


def main():
    sites, aa = read_entropy('AAseq_Node168.csv', 'Node168')
    di_sites, di = read_entropy('3Di_Node16.csv', 'Node16')
    assert np.array_equal(sites, di_sites)
    assert np.array_equal(sites, np.arange(1, 308))
    with REFERENCE.open() as handle:
        reference = list(csv.DictReader(handle, delimiter='\t'))
    np.testing.assert_allclose(aa, [float(r['aa_entropy_bits']) for r in reference], atol=1e-12)
    np.testing.assert_allclose(di, [float(r['direct_3di_entropy_bits']) for r in reference], atol=1e-12)
    OUTPUT.mkdir(exist_ok=True)
    font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/GillSans.ttc')
    plt.rcParams.update({'font.family': 'Gill Sans', 'font.size': 12,
                         'axes.labelsize': 12, 'xtick.labelsize': 12,
                         'ytick.labelsize': 12, 'legend.fontsize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    blue, orange = '#2463A6', '#D97916'
    fig, ax = plt.subplots(figsize=(4, 4), layout='constrained')
    ax.plot(sites, aa, color=blue, lw=0.35, label='AA model')
    ax.plot(sites, di, color=orange, lw=0.35, label='3Di model')
    ax.set(xlabel='Alignment site', ylabel='Shannon entropy (bits)',
           xlim=(1, 307),
           ylim=(-0.05, 4.05))
    ax.grid(False)
    legend = ax.legend(frameon=False, loc='upper right')
    for line in legend.get_lines():
        line.set_linewidth(2.0)
    save(fig, 'glugln_aa_3di_entropy_by_site')

    fig, ax = plt.subplots(figsize=(4, 4), layout='constrained')
    ax.plot([0, 4.05], [0, 4.05], ls='--', lw=0.35, color='#999999', zorder=0)
    ax.scatter(aa, di, s=25, color=blue, alpha=0.6, edgecolors='none')
    ax.set(xlabel='AA model entropy (bits)', ylabel='3Di model entropy (bits)',
           xlim=(-0.06, 4.05), ylim=(-0.06, 4.05))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    save(fig, 'glugln_aa_vs_3di_entropy_scatter')
    print(f'Validated 307 sites against existing entropy results. Figures: {OUTPUT}')


if __name__ == '__main__':
    main()
