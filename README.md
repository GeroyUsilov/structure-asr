# structure-asr

Structure-informed ancestral sequence reconstruction (ASR).

Standard ASR reconstructs each ancestral site independently from amino-acid
substitution models. This project tests whether a reconstructed structural
signal, the 3Di alphabet predicted by ProstT5, improves recovery of the true
ancestor in sequence (AA identity) and in structure (3Di identity).

## Method

1. **Simulation with known ancestors.** Sequences evolve down the EF-Tu
   phylogeny (39 tips, 396 residues) by repeated masked-site resampling from
   ESM2, at temperature τ and attempt rate μ. Every ancestor is saved as
   ground truth.
2. **Reconstruction.** From the tips only, IQ-TREE on the known topology
   reconstructs AA posteriors (ModelFinder) and 3Di posteriors (ProstT5-encoded
   tips, Q.3Di.AF matrix). Nodes are paired across trees and with the truth
   by topology.
3. **Structure-guided sampling.** Metropolis–Hastings from the AA marginal-MAP
   reconstruction, targeting
   S_λ(x) = ℓ_AA(x) + λ·ℓ_3Di(ProstT5(x)). λ = 0 is the sequence-only control.
4. **Evaluation.** AA and 3Di identity of the best-scoring sample to the true
   ancestor. Truth is read only at this stage.

Equations and conventions: [docs/METHODS.md](docs/METHODS.md).

## Findings

- **Baseline.** The historical μ=1, τ ∈ {1.5, 2} reconstructions reach ~97.9%
  AA marginal-MAP accuracy.
- **Fold loss at τ ≥ 1.5**
  ([divergence](experiments/2026-11-02_divergence/NOTES.md)). Beyond μ ≈ 2,
  many simulated tips are predicted disordered (17/39 at μ=4, τ=2). λ ≤ 0.1
  changed AA errors by at most one site.
- **Fold conserved at τ = 0.5**
  ([tau-low](experiments/2026-11-02_tau-low/NOTES.md)). All tips stay folded
  up to μ=8. 3Di reconstruction is more accurate than AA reconstruction at
  μ ≥ 4 (91.2% vs 85.3% at μ=8). Low-τ lineages converge, which makes these
  cells easier than their divergence suggests.
- **λ scan at τ = 0.5**
  ([tau-low-lambda](experiments/2026-11-02_tau-low-lambda/NOTES.md); λ ∈
  {0, .02, .1, .5, 1}, μ ∈ {2, 4, 8}, three ancestors each).
  - **AA recovery:** never improved. λ ≥ 0.5 made it worse (146 → 158 errors
    at μ=4, λ=1). The chains do raise S_λ; about one in three moves fixes an
    error.
  - **3Di fidelity:** improved in 4 of 9 ancestors (μ=2 and μ=8). One case
    was free (μ=2 Node1: 3Di 31 → 28, AA unchanged), one improved both (μ=8
    Node29: AA 73 → 70, 3Di 24 → 23), and the rest traded AA for 3Di. μ=4
    got worse on both.
- **Open question.** Can a per-site, confidence-weighted structural term
  replace the single global λ?

## Installation

Python 3.11+.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[analysis]'     # or: conda env create -f environment.yml
bash scripts/check.sh            # CPU tests (deterministic model stand-ins) + example replay
```

## Running on GPUs (RunPod)

ESM2 and ProstT5 run on RunPod. A **grid** (a TOML file in `grids/`) runs the
pipeline over (μ, τ, seed, λ). Every stage checkpoints and can resume:

| Stage | Hardware | Command |
|---|---|---|
| Simulate | GPU (ESM2) | `asr-grid forward` |
| Encode to 3Di | GPU (ProstT5) | `asr-grid encode` |
| Reconstruct AA + 3Di | CPU (IQ-TREE) | `asr-grid asr` |
| Structure-guided sampling | GPU (ProstT5) | `asr-grid sample` |
| Evaluate against truth | CPU | `asr-grid evaluate` |

```bash
bash scripts/grid.sh plan grids/example.toml      # preview, no GPU needed
python scripts/runpod/fleet.py --help             # launch / sync / stop pods
```

`fleet.py` sends the committed code to each pod and records the commit in
`grid_plan.json`. Each finished run gets a committed record under
`experiments/` (`scripts/record_run.sh`). Full command reference:
[docs/WORKFLOWS.md](docs/WORKFLOWS.md).

## Repository layout

```
src/structure_asr/       the Python package
  simulation/            forward-evolution engine (tree walk, kernels, scorers, checkpoints)
  forward.py, esm2.py    ESM2-driven simulation;  batched_forward.py, fold_conservation.py
  prostt5.py             amino acids → 3Di
  iqtree.py              fixed-topology AA and 3Di ASR
  posteriors.py          posterior parsing and sequence scores
  sampling.py            structure-guided MH sampler (legacy_sampling.py: historical)
  evaluation.py          topology matching and fidelity vs. truth
  grid.py                multi-setting pipeline driver (asr-grid)
  analysis/, ofs/, smoke/   historical analyses, experimental OFS sweep, GPU smoke test
tests/                   CPU tests with deterministic fakes
grids/                   grid definitions (TOML)
examples/                bundled input data, read-only (EF-Tu, Glu/Gln)
experiments/             committed records of each real run
scripts/                 stage wrappers, check.sh, record_run.sh, analysis/, runpod/
docs/                    METHODS, WORKFLOWS, VALIDATION, EXPERIMENTS; archive/
legacy/                  earlier implementations, kept for reference
```

The older import paths `forward_asr_benchmark` and `runpod_*` still work and
point into `structure_asr`.

## Documentation

- [docs/METHODS.md](docs/METHODS.md): algorithms, equations, conventions
- [docs/WORKFLOWS.md](docs/WORKFLOWS.md): commands, RunPod, output layout
- [docs/VALIDATION.md](docs/VALIDATION.md): test coverage and known limitations
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): historical experiments
- [AGENTS.md](AGENTS.md): working rules
- [CHANGELOG.md](CHANGELOG.md)
