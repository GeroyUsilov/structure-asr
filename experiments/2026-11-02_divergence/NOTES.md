# 2026-11-02_divergence

- Question: at higher forward divergence, does the 3Di term (λ) improve AA
  recovery? The μ=1 cells reconstruct at ~97.9% AA MAP, leaving almost nothing
  for structure to fix.
- Code: commit ecc3278 (`grid_plan.json`); fleet run `divergence` (4 shards).
  Evaluated with cbd0e03, which fixed the MAP-state reader (below).
- Grid: μ ∈ {1,2,4,8} × τ ∈ {1.5,2}, seed 731024; λ ∈ {.02,.05,.1} at Node1,
  Node29, Node34; 1000 steps, T=2, mixture .9. No λ=0 control (deferred).
- Hardware: 2 × RTX 3090 + 2 × RTX 4090, RunPod Secure, template
  `runpod-torch-v240`. Sampling 10.2 s/step (4090) and 15.2 s/step (3090) for
  18 chains in one batch of ≤32.
- Raw data: `results/runpod/divergence/grid` (local, ignored); pods deleted.
- Deviations: **shard 2 (μ=2 τ=1.5, μ=8 τ=1.5) is missing.** Its pod finished,
  but the sync loop deleted it while its output rsync was failing, and the laptop
  had slept for ~6 h so the loop was paused (~$7.70 of idle billing). A rerun was
  launched and then cancelled on the findings below. Fixed in ef1e130 (delete only
  after a clean copy; wall-clock waiting).
- Cost: ~$18.70 in total, including the idle time and $0.25 of the cancelled rerun.

## Findings

### 1. Reconstruction accuracy against divergence (AA MAP, 37 nodes)

| μ | τ=1.5 | τ=2 |
|---|---|---|
| 1 | 97.86% | 97.86% |
| 2 | (lost) | 91.74% |
| 4 | 70.46% | 68.76% |
| 8 | (lost) | 22.80% |

3Di MAP accuracy is non-monotonic (89.5%, 84.5%, 53.5%, 46.3%, 70.2%, 60.6%),
which finding 2 explains.

### 2. ProstT5 encodes diverged sequences as disordered

Per-tip 3Di "D" (coil) fraction. A folded EF-Tu tip sits at 0.11–0.19.

| Cell | tips ≥90% D | 30–90% D | <30% D |
|---|---|---|---|
| μ=1 τ=1.5 | 0/39 | 0 | 39 |
| μ=1 τ=2 | 0 | 1 | 38 |
| μ=2 τ=2 | 7 | 9 | 23 |
| μ=4 τ=1.5 | 5 | 23 | 11 |
| μ=4 τ=2 | 17 | 16 | 6 |
| μ=8 τ=2 | 16 | 22 | 1 |

Median tip 3Di identity to the root falls 0.72 (μ=1 τ=1.5) → 0.12 (μ=4 τ=2).
The all-D tips' AA sequences have near-flat residue usage (commonest residue
~25–35 of 396) and read as near-random, so ProstT5 is not misbehaving: the ESM2
generator has no structural selection and drifts out of foldable space. At
μ=4 τ=2 Node29 has 120 AA errors but **0** 3Di errors, because truth and
reconstruction are both all-D — 3Di identity becomes uninformative.

### 3. λ did not improve AA recovery anywhere

Summed AA errors over the three nodes, MAP start → best-scoring sample:

| Cell | λ=.02 | λ=.05 | λ=.1 |
|---|---|---|---|
| μ=1 τ=1.5 | 36 → 36 | 36 → 36 | 36 → 36 |
| μ=1 τ=2 | 37 → 38 | 37 → 38 | 37 → 39 |
| μ=2 τ=2 | 120 → 121 | 120 → 137 | 120 → 136 |
| μ=4 τ=1.5 | 370 → 371 | 370 → 371 | 370 → 382 |
| μ=4 τ=2 | 400 → 400 | 400 → 399 | 400 → 399 |
| μ=8 τ=2 | 1036 → 1037 | 1036 → 1037 | 1036 → 1038 |

At μ=1 the chains barely move from the start. At μ≥2 stronger λ is actively
worse: at μ=2 τ=2, Node29/Node34 3Di errors roughly doubled (195 → 341 summed at
λ=.05) along with AA errors, i.e. the chains chase a 3Di posterior that is itself
wrong at those nodes.

### 4. The objective ranks the AA-MAP reconstruction above the truth

Evaluation-only diagnostic: score each true ancestor and the sampler's start
under Sλ = ℓAA + λℓ3Di.

| Cell | Node | AA errs | ΔℓAA (truth−start) | Δℓ3Di | λ needed for truth to win |
|---|---|---|---|---|---|
| μ1 τ1.5 | Node1 | 21 | −40.9 | +58.2 | >0.70 |
| μ1 τ1.5 | Node29 | 4 | −11.5 | −45.0 | never |
| μ1 τ1.5 | Node34 | 11 | −31.0 | +16.6 | >1.87 |
| μ1 τ2 | Node1 | 24 | −32.6 | +4.8 | >6.80 |
| μ1 τ2 | Node29 | 5 | −7.8 | +39.0 | >0.20 |
| μ1 τ2 | Node34 | 8 | −19.0 | +4.9 | >3.91 |
| μ2 τ2 | Node1 | 51 | −154.7 | −310.6 | never |
| μ2 τ2 | Node29 | 29 | −66.7 | +41.4 | >1.61 |
| μ2 τ2 | Node34 | 40 | −110.0 | −219.7 | never |
| μ4 τ1.5 | Node1 | 90 | −230.5 | +144.8 | >1.59 |
| μ4 τ1.5 | Node29 | 105 | −252.5 | −67.8 | never |
| μ4 τ1.5 | Node34 | 175 | −317.9 | +587.4 | >0.54 |

The 3Di term points toward the truth in 8 of 12 cases but needs λ of 0.2–6.8 to
overcome the AA term, far above the λ ≤ 0.1 range scanned here (and above the
λ=1-equivalent of the legacy α=.5 runs). **The sampler sitting at the MAP start
is the correct answer for this objective**, not a mixing failure. Whole-sequence
scores; per-site moves can still differ.

### 5. ProstT5 amplifies small AA differences

μ=1 τ=2 Node29: 5 AA errors → 28 3Di errors; Node34: 8 → 22. Encoding the AA
reconstruction still beats reconstructing on the 3Di tips directly (40 vs 105
summed 3Di errors at μ=1 τ=1.5; 195 vs 576 at μ=2 τ=2), so the 3Di posterior is
the weaker estimate of the two being combined.

## Interpretation

There is no regime here where the sequence is saturated but the fold is
conserved, which is the regime where structure-aware ASR should pay off. At μ=1
the AA signal is already near-perfect; at μ≥2 the fold is gone. Scanning λ on
this generator cannot answer the question. Next steps taken:
`grids/tau_low.toml` (does conservative resampling preserve the fold at high μ?)
and `[fold]` structural selection (METHODS 7b, commit a28c9e1).

## Code fixes this run produced

- cbd0e03 `evaluate`: IQ-TREE 3.1.4 prints `-` in the `State` column for some
  3Di sites under +F models (μ≥2 here), which crashed evaluation with
  `KeyError: 'p_-'`. Now falls back to the argmax; wherever a state is printed it
  equals the argmax, so earlier results are unchanged.
- ef1e130 `fleet sync`: terminate only after the outputs copy cleanly; wall-clock
  waiting so a sleeping laptop does not delay the next sync.
- a17ed37 / a28c9e1 `fleet`: `relaunch` no longer rsyncs a results directory that
  does not exist yet, and a pod whose deploy fails is deleted instead of left
  billing. Both leaked pods during this session.
