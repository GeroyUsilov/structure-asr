# structure-asr

**Structure-informed ancestral sequence reconstruction.** This project asks
whether a protein family's structure can improve inference of its ancestors.
The aim is ancestors that are right in sequence and, just as importantly,
right in fold.

This README explains the scientific question, the approach, and what has been
found so far. How to run the code comes at the end, and the exact equations
are in [docs/METHODS.md](docs/METHODS.md).

---

## The question

**Ancestral sequence reconstruction (ASR)** infers the sequences of extinct
proteins from present-day relatives and the phylogeny that connects them.
Standard maximum-likelihood ASR fits a substitution model along the tree. For
every ancestral site it reports a posterior probability for each of the 20
amino acids.

Those models treat sites as evolving independently. Real proteins are held
together by a fold, and selection to keep that fold couples residues across
the sequence. A reconstruction can be accurate site by site and still
assemble into a sequence whose structure doesn't match the real ancestor.

Structure is usually far more conserved than sequence, so it may carry
information that sequence-only ASR misses. We encode structure with **3Di**, a
20-letter structural alphabet (introduced by Foldseek) that describes each
residue's local tertiary contacts. 3Di sequences can be aligned, modeled and
reconstructed on a tree like amino acids. The **ProstT5** protein language
model translates amino-acid sequences into predicted 3Di.

So we ask: **when 3Di is reconstructed alongside amino acids and used to guide
the choice of ancestor, do we recover ancestors that are closer to the truth,
in sequence (AA identity) and in structure (3Di identity to the true ancestor)?**

## Approach

### 1. A benchmark where the true ancestors are known

Real ancestors are never observed, so we simulate evolution and keep every
ancestor.

- **Tree and root.** The real EF-Tu phylogeny (39 tips, 77 nodes including
  the root) with its supplied rooting and branch lengths. The root is one
  natural EF-Tu sequence, 396 residues long.
- **Mutation process.** Along each branch, sites are chosen uniformly at
  random. Each chosen site is masked, and a new residue is drawn from the
  **ESM2** masked-language-model conditional, sharpened or flattened by a
  temperature **τ**. The number of attempts per branch is μ × L × branch
  length, so **μ** controls divergence.
- **Output.** Tips, all 38 internal ancestors, and every mutation event.

ESM2 keeps sequences protein-like, but it applies no explicit selection for a
fold. That makes τ a key setting. At τ ≥ 1.5, lineages past μ ≈ 2 drift out
of foldable space. At τ = 0.5 the fold is conserved, but lineages converge on
the same preferred residues. That convergence makes those cells easier to
reconstruct than their divergence suggests (see Findings).

### 2. Two independent reconstructions: sequence and structure

Only the simulated tips are passed to inference; the truth is set aside.

- **Amino-acid ASR.** IQ-TREE on the known topology. ModelFinder selects the
  substitution model, and IQ-TREE re-estimates the branch lengths and writes
  marginal posteriors per ancestral site.
- **3Di ASR.** The tips are translated to 3Di with ProstT5. IQ-TREE then
  reconstructs them with the published **Q.3Di.AF** exchangeability matrix
  (Garg & Hochberg 2025).

The AA and 3Di trees are reconstructed separately, so their node labels need
not agree. Ancestors are paired across trees, and with the truth, **by
topology** (the tips below each node), never by node name.

Using the true topology is a known-tree assumption that favors reconstruction
relative to real-world practice.

### 3. Combining the two: structure-guided sampling

For a candidate ancestor *x* we score

> S_λ(x) = ℓ_AA(x) + λ · ℓ_3Di(ProstT5(x))

ℓ_AA is the log-probability of *x* under the AA posteriors. ℓ_3Di is the
log-probability of *x*'s predicted structure under the reconstructed 3Di
posteriors. **λ** sets how much the structural evidence counts.

A Metropolis–Hastings sampler explores this target. It starts from the AA
marginal-MAP reconstruction (the standard ASR answer). It proposes changes
preferentially at uncertain sites, drawing residues from the AA posterior. It
reports the highest-scoring sequence visited. λ = 0 is the sequence-only
control.

### 4. Evaluation against the hidden truth

Evaluation is a separate stage, run after inference. It is the only code that
reads the true ancestors. No λ, threshold or node choice is tuned on it.
Two kinds of fidelity are measured:

- **Sequence fidelity:** AA identity to the true ancestor.
- **Structural fidelity:** 3Di identity between the reconstruction's
  predicted structure and the true ancestor's.

A gain in either is meaningful. A higher model score S_λ is not, by itself:
the sampler can raise its own objective while moving away from the truth.

## Findings so far

Full numbers and caveats are in each `experiments/*/NOTES.md` and in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

**Historical cases reproduce.** The stored μ=1, τ ∈ {1.5, 2} reconstructions
reach about 97.9% AA marginal-MAP accuracy.

**At τ ≥ 1.5, simulated proteins lose their fold beyond μ ≈ 2**
([divergence](experiments/2026-11-02_divergence/NOTES.md)). By μ=4, τ=2,
ProstT5 reads 17 of 39 tips as almost entirely disordered. In that regime a
λ of 0.1 or less gave no meaningful AA improvement: the best was one site, in
a cell whose tips had largely lost their fold. The objective ranked the AA-MAP
reconstruction above the true ancestor, so the sampler staying put was the
correct behavior for that objective.

**At τ = 0.5 the fold is conserved, and 3Di reconstruction beats AA
reconstruction** ([tau-low](experiments/2026-11-02_tau-low/NOTES.md)). Every
tip stays folded up to μ=8. At μ ≥ 4 the direct 3Di reconstruction is the
more accurate of the two: 91.2% vs 85.3% at μ=8, and one node has 19 3Di
errors against 73 AA errors. This is the first regime where the structural
channel clearly holds information the sequence channel lacks. One caveat:
low-τ lineages converge, so these cells are easier than their divergence
implies.

**New: a full λ scan in the fold-conserving regime**
([tau-low-lambda](experiments/2026-11-02_tau-low-lambda/NOTES.md)). The scan
covered λ ∈ {0, .02, .1, .5, 1} at μ ∈ {2, 4, 8}, τ = 0.5, on three ancestors
each.

- *Sequence fidelity did not improve at any λ.* Small λ leaves the AA-MAP
  start unchanged. λ ≥ 0.5 makes it worse (e.g. 146 → 158 summed errors at
  μ=4, λ=1). This is not a mixing failure: at λ ≥ 0.5 the chains raise S_λ by
  tens of nats and move up to 21 of 396 residues. About a third of those moves
  correct an error; two thirds break a correct residue.
- *Structural fidelity did improve in 4 of 9 ancestors,* at μ=2 and μ=8. The
  structural term pulls reconstructions toward the true ancestor's fold. In
  one case this came free (μ=2 Node1: 3Di errors 31 → 28, AA unchanged). In
  another, both improved (μ=8 Node29 at λ=0.5: AA 73 → 70, 3Di 24 → 23).
  Elsewhere it traded AA accuracy for structural accuracy. μ=4 got worse on
  both.
- *Interpretation.* The structural signal is real but uneven: it helps at
  some sites and hurts at others. ProstT5 maps many sequences to the same
  structure, so "closer to the true fold" can be satisfied by residues that
  are structurally right but chemically wrong. One global λ cannot tell these
  apart, and scanning λ further on this additive objective is unlikely to
  help.

**Next question.** Instead of one global λ, weight the structural term at each
site by how confident the 3Di reconstruction is there. At μ=8 the 3Di MAP is
right at 87% of the sites where the AA MAP is wrong, with mean confidence 0.98.
Whether that confidence is calibrated at the sites that matter is untested.

---

## Using the code

### On a laptop (CPU, no model downloads)

Python 3.11+.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[analysis]'     # or: conda env create -f environment.yml
bash scripts/check.sh            # tests + replay of the bundled examples (~20 s)
bash scripts/check.sh --quick    # tests only
```

The tests use small deterministic stand-ins for ESM2 and ProstT5. The replay
rebuilds the published tables and figures under `results/` from bundled
inputs.

### On GPUs (RunPod)

ESM2 and ProstT5 run on RunPod. A **grid** (a TOML file in `grids/`) runs the
whole pipeline over many (μ, τ, seed, λ) settings. Every stage checkpoints and
can resume:

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

`fleet.py` gives each pod the committed code and records the commit in
`grid_plan.json`. Finished runs get a small committed record under
`experiments/` (`scripts/record_run.sh`). The full command reference, including
manual pods, is [docs/WORKFLOWS.md](docs/WORKFLOWS.md).

### Repository layout

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
scripts/                 stage wrappers, check.sh, record_run.sh, runpod/
docs/                    METHODS, WORKFLOWS, VALIDATION, EXPERIMENTS; archive/
legacy/                  earlier implementations, kept for reference
```

Older import paths (`forward_asr_benchmark`, `runpod_*`) still work and point
into `structure_asr`.

### Further reading

- [docs/METHODS.md](docs/METHODS.md): exact algorithms, equations and conventions
- [docs/WORKFLOWS.md](docs/WORKFLOWS.md): commands, RunPod, output layout
- [docs/VALIDATION.md](docs/VALIDATION.md): what is tested; known limitations
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): historical experiments
- [AGENTS.md](AGENTS.md): working rules (scientific and git)
- [CHANGELOG.md](CHANGELOG.md)
