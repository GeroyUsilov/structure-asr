# Worked examples and observed outcomes

## Included artifacts

| Experiment | Location | Status |
|---|---|---|
| EF-Tu μ=.5,1,2,3,4,5,6; τ=1 | `examples/ef_tu/rate_sweep_tau1` | Downloaded tip AA/3Di, trees, manifests |
| EF-Tu μ=1,2,3 × τ=1,1.5,2 | `examples/ef_tu/tips` | Nine correctly labeled AA/3Di tip datasets |
| QE1.9d empirical AA/3Di | `examples/ef_tu/empirical` | Comparison FASTAs |
| μ=1, τ=1.5 and 2 | `examples/ef_tu/mu1_tau*` | All-node truth, inferred marginals/trees |
| Legacy joint Node1/29/34, both temperatures | `mu1_tau*/legacy_results` | Completed 3000-step runs, stored start/best/final and RNG checkpoints |
| Later Node34 uncertainty-weighted sampling | `scripts/sample_node34.sh` | Runnable preset, not a newly completed run |
| Glu/Gln AA Node168 / 3Di Node16 | `examples/glugln` | 307-site inputs, prior uncertainty table, reproducible plotting/analysis |

`selection.tsv` records shallow/intermediate/deep representative-node selection
for each EF-Tu condition. Node numbers from Glu/Gln are not EF-Tu node numbers.

τ=1 tip data are included, but complete τ=1 ancestral truth is not included.
Do not infer complete truth availability from the existence of tip FASTAs.

## Reproduced from saved inputs/checkpoints

Across all 37 inferred nodes, AA-MAP accuracy is 97.9047% for both τ=1.5 and 2.
Mean posterior assigned to true AA states is .9726086 and .9709496, respectively.

| τ | Node | AA errors start→best | Predicted 3Di errors start→best |
|---|---|---|---|
| 1.5 | Node1 | 18→24 | 18→27 |
| 1.5 | Node29 | 6→4 | 15→17 |
| 1.5 | Node34 | 10→13 | 23→44 |
| 2 | Node1 | 23→17 | 26→27 |
| 2 | Node29 | 4→5 | 0→6 |
| 2 | Node34 | 7→12 | 81→65 |

Totals: 68→75 AA errors across six reconstructed sequences. These are the
legacy runs (α=.5, T=2), not runs of the uncertainty-weighted sampler.

Truth mean 3Di log probability/site was about −.673 (τ=1.5) and −1.377 (τ=2),
versus posterior-sample means about −.064 and −.130. See `METHODS.md` for
normalization/flooring conventions and interpretation limits.

## Additional observations from the conversation

A later supplied Node34 τ=2 candidate had 10 AA errors and 37 3Di errors, versus
7 and 81 for AA-MAP; all three AA changes relative to MAP replaced correct
residues. Reported λ=.01 objective values were −19.014→−18.591. This is a
separately evaluated candidate, not proof of a completed four-weight sweep.
No full later-chain result directory was imported; the six-case replay above
does not silently include this candidate as another replicated run.

## Historical presets and interrupted runs

- Weak-structure Node34 sweep: λ=0,.005,.01,.02,.05; 3 chains ×3000 steps.
- Four independent Node34 jobs: λ=.01,.02,.05,.1; 1 chain ×1000 steps.
  The current `sample_node34.sh` reproduces these parameter settings.
- μ=20,40,60 simulation: stopped by the user; do not treat as completed.

These entries record provenance and available presets, not jobs to launch or
resume in a new session.

## Figures

Historical PNGs in `docs/figures/`: nine-condition AA/3Di tip-distance ECDFs,
dataset-wide score histograms, and representative-node histograms at τ=1.5/2.
`scripts/reproduce_examples.sh` rebuilds them under `results/`, together with
the Glu/Gln entropy line/scatter and 10-site probability logos. The selected
logo window is the maximum-mean-3Di-entropy stretch in the middle third,
not an unbiased randomly chosen stretch. Labels use Gill Sans when installed,
otherwise DejaVu Sans; probability glyphs use a monospace font.
