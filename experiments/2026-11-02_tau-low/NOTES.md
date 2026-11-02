# 2026-11-02_tau-low

- Question: does conservative AA resampling (τ=0.5) at high μ keep the fold
  intact, giving the saturated-sequence/conserved-fold regime that
  `2026-11-02_divergence` showed does not exist at τ≥1.5?
- Code: commit 50120df (`grid_plan.json`); fleet run `taulow`,
  `grids/tau_low.toml`. Reconstruction stages only, no `[sample]`.
- Hardware: 1 × RTX 4090, RunPod Secure, $0.89/h. forward ~16 min for 66,918
  ESM2 calls; encode+asr rerun after an OOM (below).
- Raw data: `results/runpod/taulow/grid` (local, ignored); pod deleted.
- Cost: ~$0.90.

## Findings

### 1. τ=0.5 conserves the fold at every μ tested

Per-tip 3Di "D" (coil) fraction and 3Di identity to the root's encoding:

| Cell | tips ≥90% D | <30% D | tip 3Di identity to root (min/median/max) |
|---|---|---|---|
| μ=2 τ=0.5 | 0/39 | 39/39 | 0.79 / 0.82 / 0.86 |
| μ=4 τ=0.5 | 0/39 | 39/39 | 0.75 / 0.79 / 0.83 |
| μ=8 τ=0.5 | 0/39 | 39/39 | 0.72 / 0.77 / 0.81 |
| μ=1 τ=1.5 (reference) | 0/39 | 39/39 | 0.63 / 0.72 / 0.78 |
| μ=4 τ=2 (reference) | 17/39 | 6/39 | ~0.06 / 0.12 / 0.17 |

Every tip stays folded, and tips are structurally *closer* to the root than in
the μ=1 τ=1.5 cell. Lowering τ, not lowering μ, is what preserves structure.

### 2. The cost is that each attempt rarely changes anything

| Cell | attempts changing a residue | leaf–root AA identity | pairwise tip identity |
|---|---|---|---|
| μ=2 τ=0.5 | 33% | 0.51 | 0.67 |
| μ=4 τ=0.5 | 27% | 0.46 | 0.65 |
| μ=8 τ=0.5 | 22% | 0.41 | 0.61 |
| μ=1 τ=1.5 | 78% | 0.45 | 0.44 |

A sharpened conditional redraws the residue already present, so μ=8 at τ=0.5
reaches the leaf–root divergence μ=1 at τ=1.5 already had, for 8× the ESM2 cost.

**Caveat: the divergence is convergent.** Pairwise tip identity is 0.61 at μ=8
τ=0.5 against 0.44 at μ=1 τ=1.5 — much more similar to each other than their
root distance implies, because every lineage keeps choosing the same
ESM2-preferred residue at a site. Low-τ cells are therefore easier to
reconstruct than their leaf–root identity suggests, and are not a realistic
model of independent lineage evolution. Report this whenever a low-τ cell's
accuracy is compared with a τ≥1.5 cell's.

### 3. This is the first regime where 3Di beats AA

MAP accuracy over 37 mapped nodes:

| Cell | AA | 3Di | AA best-fit model |
|---|---|---|---|
| μ=2 τ=0.5 | 95.37% | 95.05% | WAG+F+I+G4 |
| μ=4 τ=0.5 | 89.36% | 91.60% | BLOSUM62+F+I+G4 |
| μ=8 τ=0.5 | 85.29% | 91.22% | VT+F+I+G4 |

At μ≥4 the direct 3Di reconstruction is *more* accurate than the AA one, by 2.3
and 5.9 points. Per node the gap can be large: at μ=8, Node29 has 73 AA errors
against 19 3Di errors. In every earlier cell the 3Di posterior was the weaker of
the two estimates, so this is the first dataset where the structural channel
carries information the AA channel is missing. AA accuracy is also unsaturated
(85–95%), leaving room for a structural term to help.

ModelFinder picks a different AA matrix per cell (WAG/BLOSUM62/VT, all +F+I+G4)
rather than the PMB family chosen for every τ≥1.5 cell, consistent with the
low-entropy, convergent substitution pattern of finding 2.

### 4. The λ diagnostic still needs a GPU step

The `2026-11-02_divergence` objective check compared truth with the sampler's
start under Sλ. It cannot be repeated here yet: the sampler's start is
ProstT5(AA-MAP), which these cells do not contain (no `[sample]` stage ran), and
substituting the *direct* 3Di MAP makes the comparison vacuous — the marginal-MAP
sequence maximizes its own posterior, so Δlog P ≤ 0 for both terms by
construction. Encoding the 37 AA-MAP reconstructions per cell is ~1 min of
ProstT5 and should precede any λ scan here.

## Operational

- **ProstT5 batch 32 is not safe on 24 GB cards.** This 4090 reported 22.03 GiB
  usable and batch 32 peaks near 22.4 GB, so `encode` OOMed where the
  `divergence` pods had squeaked through. `run_shard.sh` now encodes at batch 16
  (`--encode-batch-size`) and samples at 32. The stage was rerun on the same pod
  with `--batch-size 12 --resume`; forward output was unaffected.
- The fleet retry loop leaked a billing pod per attempt because `relaunch`
  rsynced a results directory that does not exist on a fresh run (fixed in
  a17ed37, with deploy-failure cleanup in a28c9e1). One leaked pod was deleted
  manually, ~$0.04.
