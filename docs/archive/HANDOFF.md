> Archived 2026-11-01. This was written for the original handoff; current conventions are in `AGENTS.md`.

# Handoff for a fresh agent

## Project question

Can inferred ancestral structural information help resolve ambiguous ancestral
amino acids? We reconstruct AA and 3Di sitewise marginals separately, then score
candidate AA sequences by AA support and the support of their ProstT5-predicted
3Di encoding. The scientific endpoint is AA recovery, not simply higher scores
or better 3Di identity.

The longer-term application is ancestral tRNA synthetases, including ancient
Glu/Gln relationships. Synthetic EF-Tu is the controlled benchmark with known
ancestral sequences.

## Where to start

1. Read `METHODS.md` for exact kernels and conventions.
2. Consult `WORKFLOWS.md` for RunPod setup, launch commands, checkpoints and outputs.
3. For optional local validation, the README describes the CPU test suite (40 tests);
   `bash scripts/reproduce_examples.sh` regenerates historical
   benchmark tables, score histograms, tip-distance ECDFs, Glu/Gln entropy plots
   and logos. All inputs are in this repository; no Desktop paths are needed.
4. Inspect `EXPERIMENTS.md` and the stored `run_manifest.json` files before
   making claims about a run or changing its parameters.

## Current scientific state

- EF-Tu generation used exact masked-site ESM2, **not OFS**, despite the early
  conversation and experimental OFS modules. It did not use ESMFold.
- Same fixed 39-tip topology, 396-AA root; rates scale update counts, not
  realized substitutions or an explicit continuous-time evolutionary rate.
- μ=1, τ=1.5/2 AA-MAP reconstruction was already about 97.90% accurate.
- The initial six representative joint reconstructions used the **legacy**
  acceptance rule and equal score weights. Their total AA errors rose 68→75.
- The later sampler uses AA-uncertainty site proposals, non-self residue
  proposals and the correct Hastings term. Do not attribute its statistical
  properties to the historical equal-weight runs.
- Ground-truth 3Di log scores are far below typical posterior-sample scores.
  This exposes a discrepancy, not a diagnosis of its cause.
- The user-supplied later Node34 candidate had 10 AA and 37 3Di errors versus
  7 and 81 at AA-MAP. Its complete inference provenance was not imported;
  treat it separately from the six reproducible legacy outputs.

## Starting a new session

This handoff provides project background, not an active task queue. Follow the
user's new request; do not resume old simulations, implement previously discussed
extensions, or launch validation/inference merely because they appear in the
historical documentation. Workflow commands are available examples, not assignments.

RunPod source code, dependency files, launch scripts and usage instructions are
included in this repository. `WORKFLOWS.md` is the entry point. Model weights
are not bundled. The available workflows are separate stages. `asr-grid`
(WORKFLOWS "Multi-regime grids") chains batched forward simulation, 3Di
encoding, fixed-topology IQ-TREE ASR, batched joint sampling over structural
weights λ and evaluation for new (μ, τ, seed) grids.
The historical posterior files in `examples/` remain fixed inputs. `VALIDATION.md` records the tested scope and known limitations.

## User preferences and boundaries

- Reuse the existing working ProstT5 code; avoid another bespoke loading path.
- No local ESM2/ProstT5 downloads, no encoder retraining.
- Frequent checkpoints and progress bars; RunPod data/caches on persistent storage.
- Preserve seeds, internal nodes, tip identifiers, exact settings and topology.
- Keep ground-truth evaluation separate from inference and model-based candidate
  selection. Do not use truth to guide proposals or tune on the evaluation cases.
- Keep the engineering practical: one clear entry point per task, no redundant
  bundles or framework layers. Document unresolved limitations plainly.

## Repository history and scope

The Desktop originals were unversioned. A curated baseline was committed to
`main`; active work is on `refactor/documented-handoff`. `PROVENANCE.md` maps the
imported artifacts to originals. No remote was configured or published.
Old modules remain as compatibility shims or labeled experimental references.
All simulation math, RNG ordering and model-generation settings were retained.
