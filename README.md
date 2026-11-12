# structure-asr

**Structure-informed ancestral sequence reconstruction.** This project asks
whether knowing a protein's 3D structure helps us infer what its ancestors
looked like.

This README walks through the idea, how the pieces fit together, and how to
run them. You don't need a GPU to explore it: the tests and a full replay of
the bundled examples run on a laptop.

---

## The idea, in brief

**Ancestral sequence reconstruction (ASR)** starts from present-day protein
sequences and a phylogenetic tree that relates them. It works backwards to
estimate the sequences at the tree's internal nodes, the ancestors. Standard
tools such as IQ-TREE give, for every ancestral site, a probability for each
of the 20 amino acids (a *posterior*).

Those posteriors treat each site independently. Real proteins don't evolve
that way: residues have to keep working together to hold a fold. So we ask
whether adding a structural signal improves the reconstruction.

The structural signal comes from **3Di**, a 20-letter "structural alphabet"
(from Foldseek) that describes each residue's local 3D environment. The
**ProstT5** language model translates an amino-acid sequence into a predicted
3Di sequence, which lets us score how well a candidate ancestor matches the
reconstructed ancestral *structure*.

Real ancestors are unknown, so we can't score real reconstructions. Instead:

1. **Simulate evolution** down a real tree (EF-Tu, 39 tips). The **ESM2**
   protein language model proposes mutations, so the sequences stay
   protein-like. Every ancestral sequence is saved, so the true answer is
   known.
2. **Reconstruct** the ancestors from the simulated tips alone, as one would
   with real data. This is done for both amino acids and 3Di.
3. **Sample** candidate ancestors that balance amino-acid evidence against
   structural evidence. A weight **λ** controls how much structure counts.
4. **Evaluate**: compare against the saved truth. *The only measure of success
   is amino-acid recovery.* A higher model score or better 3Di agreement does
   not count as an improvement on its own.

Truth files are read only in step 4, never during inference or tuning.

## How the pipeline fits together

```
 tree + root sequence
        │
        ▼
 ┌──────────────┐   ESM2 (GPU)        asr-evolve / asr-grid forward
 │ 1. simulate  │──────────────────▶  tips + every ancestor (ground truth)
 └──────────────┘
        │ tips only
        ▼
 ┌──────────────┐   ProstT5 (GPU)     asr-encode / asr-grid encode
 │ 2. encode    │──────────────────▶  3Di versions of the sequences
 └──────────────┘
        ▼
 ┌──────────────┐   IQ-TREE (CPU)     asr-grid asr
 │ 3. reconstruct│─────────────────▶  AA and 3Di posteriors per ancestor
 └──────────────┘
        ▼
 ┌──────────────┐   ProstT5 (GPU)     asr-sample / asr-grid sample
 │ 4. sample    │──────────────────▶  ancestors scored by AA + λ·3Di
 └──────────────┘
        ▼
 ┌──────────────┐   CPU               asr-grid evaluate / asr-examples
 │ 5. evaluate  │──────────────────▶  accuracy vs. truth, tables, figures
 └──────────────┘
```

A **grid** (`asr-grid`, configured by a TOML file in `grids/`) runs all five
stages across many settings at once. The settings are the mutation rate **μ**,
the simulation temperature **τ**, the random seed and the structural weight
**λ**. Each stage checkpoints and can resume.

## Getting started (CPU, no model downloads)

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[analysis]'
bash scripts/check.sh            # tests + replay of the bundled examples
bash scripts/check.sh --quick    # tests only
```

You can use conda instead: `conda env create -f environment.yml`.

The tests use small deterministic stand-ins for ESM2 and ProstT5, so nothing
large is downloaded. The replay rebuilds the published tables and figures
under `results/` from saved inputs.

If NumPy, Biopython, tqdm, pandas and matplotlib are already installed, this
avoids any download:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

## Running the GPU stages

ESM2 and ProstT5 run on [RunPod](https://www.runpod.io/), not locally. The
fleet launcher starts one self-contained pod per shard of a grid. It ships the
committed code and records the commit in `grid_plan.json`:

```bash
bash scripts/grid.sh plan grids/example.toml      # what would run, no GPU needed
bash scripts/grid.sh status grids/example.toml    # progress of an existing run
python scripts/runpod/fleet.py --help             # launch / sync / stop pods
```

For a manual pod with a CUDA PyTorch image:

```bash
git clone https://github.com/GeroyUsilov/structure-asr.git && cd structure-asr
git checkout <commit-or-branch>
python -m pip install -r requirements-gpu.txt
python -m pip install -e '.[analysis]'
```

Most long commands accept `--validate-only` or `--plan-only` to check inputs
without loading a model, for example:

```bash
bash scripts/forward_ef_tu.sh --outdir results/mu123_tau2 \
  --rates 1 2 3 --proposal-temperature 2 --plan-only
bash scripts/sample_node34.sh 0.02 --validate-only
```

The full command reference is in [docs/WORKFLOWS.md](docs/WORKFLOWS.md).

## Repository layout

```
structure-asr/
├── src/structure_asr/        the Python package (everything importable lives here)
│   ├── simulation/           forward-evolution engine: tree walk, mutation kernels,
│   │                         scorers, checkpoints, FASTA I/O  (CLI: forward-asr)
│   ├── forward.py, esm2.py   EF-Tu forward runs with exact masked-site ESM2
│   ├── batched_forward.py    many simulations at once on one GPU
│   ├── fold_conservation.py  optional structural selection during simulation
│   ├── prostt5.py            amino acids → 3Di
│   ├── iqtree.py             fixed-topology IQ-TREE ASR
│   ├── posteriors.py         reading posteriors, sequence scores
│   ├── sampling.py           structure-guided sampler (current)
│   ├── legacy_sampling.py    historical sampler, kept for reproducibility
│   ├── evaluation.py         match nodes by topology, score against truth
│   ├── score_placement.py    where the truth falls in score distributions
│   ├── grid.py               the multi-setting grid driver (asr-grid)
│   ├── benchmark_examples.py, example_distances.py   CPU replay of examples
│   ├── analysis/             historical EF-Tu sweep analyses (Hamming ECDFs)
│   ├── ofs/                  experimental paper-tree OFS sweep
│   └── smoke/                GPU smoke test for the ESM2 generator
├── tests/                    CPU test suite with deterministic fakes
├── grids/                    grid definitions (TOML)
├── examples/                 bundled input data, read-only (EF-Tu, Glu/Gln)
├── experiments/              small committed records of real runs
├── scripts/                  stage wrappers, check.sh, record_run.sh, runpod/
├── docs/                     methods, workflows, validation; archive/ = history
├── legacy/                   earlier implementations, kept as references
├── environment.yml           conda environment (CPU)
└── requirements-gpu.txt      extra pins for GPU pods
```

`results/` and `.cache/` are created as you work and are not tracked by git.

Some older import paths still work: `forward_asr_benchmark` and the
`runpod_*` packages now just point into `structure_asr`. Saved configurations
and older scripts keep running. Please use the `structure_asr` names in new
code.

## What we have found so far

- **The historical cases reproduce.** The stored μ=1, τ∈{1.5, 2} EF-Tu
  reconstructions reach about 97.9% amino-acid marginal-MAP accuracy.
- **A better structural score is not a better ancestor.** Across six historical
  representative reconstructions, joint scores improved while AA errors rose
  from 68 to 75.
- **The structural weight λ has not helped yet.** In the divergence grid no λ
  beat the λ=0 control. Past μ≈2 at τ≥1.5 the simulated proteins lose
  their fold (ProstT5 reads them as disordered). See
  `experiments/2026-11-02_divergence/`.
- **Lower simulation temperature keeps the fold.** At τ=0.5 every tip stays
  folded up to μ=8. See `experiments/2026-11-02_tau-low/`.

[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) and each `experiments/*/NOTES.md`
have the numbers and caveats.

## Further reading

| Document | What's in it |
|---|---|
| [docs/METHODS.md](docs/METHODS.md) | Exact algorithms, equations and conventions |
| [docs/WORKFLOWS.md](docs/WORKFLOWS.md) | Every command, RunPod setup, output layout |
| [docs/VALIDATION.md](docs/VALIDATION.md) | What is tested and the known limitations |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Historical experiments and observations |
| [experiments/README.md](experiments/README.md) | How run records are made |
| [AGENTS.md](AGENTS.md) | Working rules for contributors (scientific and git) |
| [CHANGELOG.md](CHANGELOG.md) | What changed between versions |

Original-file provenance and handoff notes are in
[docs/archive/](docs/archive/).
