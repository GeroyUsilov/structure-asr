# Structure-informed ancestral sequence reconstruction

Forward protein evolution with saved ground truth, AA/3Di ancestral-sequence
sampling, and reproducible examples from the EF-Tu and Glu/Gln project.

Working conventions (scientific rules, git, experiment records): [AGENTS.md](AGENTS.md).

## Start locally — no GPU or model download

Python 3.11+ is required (`tomllib` is used). From this repository:

```bash
python -m venv .venv && .venv/bin/python -m pip install -e '.[analysis]'
bash scripts/check.sh          # CPU tests + example replay (--quick: tests only)
```

If NumPy, Biopython, tqdm, pandas and matplotlib are already installed (e.g. in
conda), `python -m venv --system-site-packages .venv` followed by
`.venv/bin/python -m pip install --no-deps --no-build-isolation -e .` avoids
any download. `scripts/check.sh` uses `.venv/bin/python` when it exists.

The replay writes tables/figures under `results/` using bundled sequences,
posteriors, and completed-chain checkpoints. It does not generate new proteins.
To run without installing this checkout, set `PYTHONPATH=src` and use an
environment with NumPy, Biopython, tqdm, pandas and matplotlib installed.

## Contents

| Location | Purpose |
|---|---|
| `src/forward_asr_benchmark/` | Tree traversal, mutation kernels, checkpoints, FASTA output |
| `src/structure_asr/forward.py`, `esm2.py` | EF-Tu sweeps; exact masked-site ESM2 generator |
| `src/structure_asr/prostt5.py` | Shared working deterministic AA→3Di conversion |
| `src/structure_asr/posteriors.py` | Shared posterior parsing and sequence scores |
| `src/structure_asr/sampling.py` | Uncertainty-weighted Metropolis–Hastings sampler |
| `src/structure_asr/legacy_sampling.py` | Historical equal-weight sampler; separately named |
| `src/structure_asr/evaluation.py`, `score_placement.py` | Topology matching, truth scores and score distributions |
| `src/structure_asr/grid.py`, `batched_forward.py`, `iqtree.py` | Multi-regime grids: batched ESM2 evolution, 3Di encoding, fixed-topology IQ-TREE ASR, batched joint sampling over λ, evaluation |
| `grids/` | Grid definitions (TOML); `calibration.toml` is the GPU timing run |
| `experiments/` | Small committed records of real runs (configs, manifests, summaries, notes) |
| `scripts/` | Stage wrappers, `check.sh` (pre-commit check), `record_run.sh` |
| `examples/ef_tu/` | Tree, root-source alignment, nine tip datasets, two completed ASR cases |
| `examples/glugln/` | Corrected AA Node168 / 3Di Node16 inputs, entropy and logo examples |
| `docs/figures/` | Selected historical result figures |
| `legacy/`, `docs/archive/` | Earlier implementations/instructions, retained as references |

The `src/runpod_*` packages retain old import paths, analysis tools and
experimental OFS code. Active implementations have the clearer names above.

## RunPod workflows

Use a CUDA PyTorch environment. Clone onto the network volume and check out the
commit you intend to run (it is recorded in `grid_plan.json`):

```bash
git clone https://github.com/GeroyUsilov/structure-asr.git && cd structure-asr
git checkout <commit-or-branch>
python -m pip install -r requirements-gpu.txt
python -m pip install -e '.[analysis]'
```

Preview a forward run without loading ESM2:

```bash
bash scripts/forward_ef_tu.sh --outdir results/mu123_tau2 \
  --rates 1 2 3 --proposal-temperature 2 --plan-only
```

Remove `--plan-only` to simulate on the GPU. Use a different output directory
for each temperature/seed. Existing generated configurations and compatibility
factory paths remain supported.

Run one 1,000-step Node34 chain with weak structural guidance:

```bash
bash scripts/sample_node34.sh 0.02 --validate-only
bash scripts/sample_node34.sh 0.02
```

Explore new forward regimes with a grid (μ × τ × seed), optionally sampled at
several structural weights λ. The stages are forward (GPU), 3Di encoding (GPU),
IQ-TREE ASR (CPU), joint sampling (GPU) and evaluation (CPU):

```bash
bash scripts/grid.sh plan grids/example.toml
bash scripts/grid.sh status grids/example.toml
```

The same Node34 command accepts `0.01`, `0.05`, or `0.1`; each weight has its own output
directory. This sampler reads posterior inputs, not ground-truth FASTAs.
The full handoff repository contains truth for evaluation and is not a physically
blinded job archive. See [workflows](docs/WORKFLOWS.md) for conversion, historical
sampling and output conventions.

## Results and current boundary

The stored μ=1, τ∈{1.5,2} cases reproduce ~97.90% AA marginal-MAP accuracy.
Across six representative historical reconstructions, AA errors changed from
68 to 75 despite improved joint scores. Structural-score improvement is not
equivalent to AA reconstruction improvement.

The workflows are separate stages. `asr-grid` adds a documented fixed-topology
IQ-TREE stage for new simulations. The historical posteriors in `examples/`
predate it and have no recorded IQ-TREE command. New runs are recorded under
[experiments/](experiments/README.md).

- [Scientific methods and caveats](docs/METHODS.md)
- [Experiment inventory and findings](docs/EXPERIMENTS.md)
- [Original-file provenance](docs/archive/PROVENANCE.md) and the original
  [handoff notes](docs/archive/HANDOFF.md)
- [Validation and known limitations](docs/VALIDATION.md)

Git: `main` is the trunk (private GitHub remote); feature work happens on short
branches. The first commit is the curated import of the Desktop originals, which
were left unchanged.
