# 2026-11-02_tau-low-lambda

- Question: in the τ=0.5 regime — the only one found so far where the fold is
  conserved and the 3Di posterior is the *more* accurate estimate
  (`2026-11-02_tau-low`, findings 1 and 3) — does the structural term improve AA
  recovery, and does it need a λ far above the ≤0.1 range that
  `2026-11-02_divergence` scanned?
- Code: commit 73942b6 (`grid_plan.json`); fleet run `tlsample`,
  `grids/tau_low.toml` with the `[sample]` section added.
- Grid: μ ∈ {2,4,8} × τ=0.5, seed 731024; **λ ∈ {0, .02, .1, .5, 1}** at Node1,
  Node29, Node34; 1000 steps, T=2, mixture .9. 45 chains, 45,000 ProstT5
  chain-steps. λ=0 is the drift control the earlier runs deferred.
- Hardware: 2 × L40S + 1 × RTX 4090, RunPod Secure, one cell per pod (15 chains
  in one batch per step). 10.0–12.7 s/step.
- Raw data: `results/runpod/tlsample/grid` (local, ignored); pods terminated.
- Deviations: the reconstruction stages were not rerun — each pod was seeded
  with the `taulow` output via `fleet launch --push-outputs`, so `forward`,
  `encode` and `asr` were skipped and only `sample` ran. The AA/3Di MAP
  reconstructions are therefore byte-identical to `2026-11-02_tau-low`, and that
  record's reconstruction findings stand unchanged.
- Cost: $11.11 (3.5–3.7 h per pod, including bootstrap).

## Findings

### 1. λ does not improve AA recovery here either, and λ ≥ 0.5 hurts

Summed AA errors over the three nodes (1188 sites), AA-MAP start → best-scoring
sample, with the 3Di errors of the same sequences:

| μ | λ=0 | λ=.02 | λ=.1 | λ=.5 | λ=1 |
|---|---|---|---|---|---|
| 2 | 73 → 73 | 73 → 73 | 73 → 73 | 73 → **77** | 73 → **79** |
| 4 | 146 → 146 | 146 → 146 | 146 → **147** | 146 → **149** | 146 → **158** |
| 8 | 180 → 180 | 180 → 180 | 180 → 180 | 180 → **187** | 180 → **187** |

| μ | 3Di λ=0 | λ=.02 | λ=.1 | λ=.5 | λ=1 |
|---|---|---|---|---|---|
| 2 | 58 → 58 | 58 → 58 | 58 → 55 | 58 → 59 | 58 → 54 |
| 4 | 81 → 81 | 81 → 81 | 81 → 92 | 81 → 100 | 81 → 105 |
| 8 | 91 → 91 | 91 → 91 | 91 → 91 | 91 → 87 | 91 → 85 |

No λ at any μ beats its own λ=0 control, and the best AA result in the table is
always the start. λ=0 moves nothing at all: the drift control confirms that
with acceptance 0.60–0.67 the chains accept plenty of moves, but none ever
outscores the AA-MAP start under the AA term alone, which is what a
marginal-MAP start should do.

**No best λ is selected from this table.** Picking one by AA error would be
tuning on truth (scientific rule 1); the table is the result, and the result is
that the curve has no interior optimum.

### 2. This is not a mixing failure — the chains win the objective and lose the truth

`2026-11-02_divergence` finding 4 left this ambiguous: there the chains sat at
the start, so "λ did not help" and "the sampler never moved" were
indistinguishable. At λ ≥ 0.5 they do move. Joint-score gain of the best sample
over the start, and how far it moved:

| μ | λ | ΔS_λ (best − start) | AA sites moved | 3Di sites moved |
|---|---|---|---|---|
| 2 | .5 | +4.7 … +7.6 | 1–6 | 3–18 |
| 2 | 1 | +12.7 … +33.7 | 1–8 | 3–12 |
| 4 | .5 | +7.0 … +39.7 | 8–21 | 14–24 |
| 4 | 1 | +59.7 … +98.5 | 13–21 | 19–27 |
| 8 | .5 | +5.7 … +15.7 | 6–20 | 6–20 |
| 8 | 1 | +31.7 … +65.0 | 11–17 | 10–22 |

(ΔS_λ is exactly 0.00 for every chain at λ ≤ 0.1 except two, where it is +0.45
and +0.31.) So at λ ≥ 0.5 the sampler genuinely optimizes S_λ = ℓAA + λℓ3Di by
tens of nats, moving up to 21 of 396 residues, **and AA accuracy gets worse
while it does so.** The objective is misaligned with the endpoint, not
under-explored. λ ≥ 0.2 was the range `2026-11-02_divergence` finding 4
predicted the truth would need; it has now been scanned, and it moves the chains
away from the truth.

### 3. Only a third of the moves are right

`move_audit.py` (evaluation-only) classifies every site where the best sample
differs from the start, pairing nodes through `node_mapping.tsv`:

| λ | wrong→right | right→wrong | wrong→wrong | right / moved |
|---|---|---|---|---|
| .1 | 1 | 2 | 1 | 0.25 |
| .5 | 30 | 44 | 12 | 0.35 |
| 1 | 33 | 58 | 13 | 0.32 |

The structural term does carry signal — 30+ sites the AA MAP had wrong are
fixed, which a term with no information would not do — but it is outvoted
roughly two to one by sites it breaks. Per node the ratio varies: μ=8 Node29 at
λ=.5 is the one node/λ that nets an improvement (4 right against 1 wrong,
73 → 70 errors), which is also the node whose 3Di reconstruction was far better
than its AA one (19 vs 73 errors in `2026-11-02_tau-low`). That is a single node
out of 18 and is not evidence for a λ setting.

### 4. 3Di agreement with the true ancestor *did* improve, in 4 of 9 nodes

The summed 3Di column of finding 1 hides this. Per node, 3Di errors of
ProstT5(best AA sample) against the true ancestor's 3Di (`*` = improved):

| μ | node | λ=0 | λ=.02 | λ=.1 | λ=.5 | λ=1 |
|---|---|---|---|---|---|---|
| 2 | Node1 | 31→31 | 31→31 | 31→28* | 31→28* | 31→28* |
| 2 | Node29 | 11→11 | 11→11 | 11→11 | 11→9* | 11→9* |
| 2 | Node34 | 16→16 | 16→16 | 16→16 | 16→22 | 16→17 |
| 4 | Node1 | 21→21 | 21→21 | 21→21 | 21→33 | 21→33 |
| 4 | Node29 | 28→28 | 28→28 | 28→28 | 28→29 | 28→32 |
| 4 | Node34 | 32→32 | 32→32 | 32→43 | 32→38 | 32→40 |
| 8 | Node1 | 34→34 | 34→34 | 34→34 | 34→29* | 34→29* |
| 8 | Node29 | 24→24 | 24→24 | 24→24 | 24→23* | 24→23* |
| 8 | Node34 | 33→33 | 33→33 | 33→33 | 33→35 | 33→33 |

The term is therefore doing its job at μ=2 and μ=8: it moves the
reconstruction's predicted structure closer to the true ancestor's. μ=4 is
uniformly worse on both channels. How the improvements pair with AA:

- **μ=2 Node1, λ ≥ .1: 3Di 31→28 with AA errors unchanged at 30.** One residue
  moved, classed wrong→wrong by `move_audit.py` — the wrong amino acid, but a
  structurally better one. A structural gain at no AA cost.
- **μ=2 Node29 (3Di −2, AA +2) and μ=8 Node1 (3Di −5, AA +3):** a straight
  trade of AA accuracy for structural accuracy.
- **μ=8 Node29 at λ=.5: both improve** — AA 73→70, 3Di 24→23. The only such
  case, and the node whose direct 3Di reconstruction most outperformed its AA
  one (19 vs 73 errors in `2026-11-02_tau-low`).

This sharpens finding 3 rather than softening findings 1–2. The structural term
is not noise; it steers toward the true fold and often gets there. It fails on
the endpoint because ProstT5 is many-to-one — "closer to the true 3Di" is
satisfiable by residues that are structurally right and chemically wrong — and a
single global λ cannot separate those from the residues that are right on both.

## Interpretation

The conserved-fold regime was the strongest remaining case for the additive
objective, and the objective fails in it. A 3Di posterior that is *more*
accurate than the AA posterior overall (91.2% vs 85.3% at μ=8) still does not
improve AA recovery when added as ℓAA + λℓ3Di at any λ, because the term is
wrong in a correlated way at the sites where AA is also uncertain, and λ large
enough to move the chain is large enough to overwrite correct residues.

What this rules out: scanning λ further on this objective. The evidence now
covers λ from .02 to 1 across μ 1–8 and τ 0.5–2, with and without a conserved
fold, and with a drift control.

What it does not rule out: a different coupling. The failure is per-site —
structure helps at some sites and hurts at others, and finding 4 shows some
structural gains cost nothing on AA — so a formulation that
weights the 3Di term per site by its own confidence, rather than one global λ,
is untested. `2026-11-02_tau-low`'s complementarity check is the reason to
think there is something there: at μ=8 the 3Di MAP is right at 87% of the 2,155
node-sites where AA is wrong, with mean confidence 0.98. Converting that into a
per-site gate (and testing whether its confidence is calibrated where it is
wrong) is the next question, not another λ.

## Operational

- The L40S is no faster than the RTX 4090 on this workload and costs 23% more:
  10.0–10.9 s/step against 12.7, within the spread of the two L40S pods
  themselves. 4090s and 3090s were out of stock at launch, so this was
  availability, not a choice. Prefer 4090 on price; take L40S when it is the
  only ≥24 GB card free. Its 48 GB does remove the batch-32 encode OOM risk that
  `2026-11-02_tau-low` hit on a 24 GB card.
- `fleet launch --push-outputs` worked as intended: all three pods skipped
  `forward`/`encode`/`asr` and went straight to sampling, so the $11 bought
  sampling only.
- Two transient sync failures (`Network is unreachable` for ~15 min, and one
  `file has vanished: checkpoint.json.tmp` race) resolved on the next pass and
  cost nothing — ef1e130 terminates a pod only after a clean copy, and all three
  outputs arrived complete (45/45 chains at step 1000).
