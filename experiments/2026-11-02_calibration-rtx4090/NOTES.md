# 2026-11-02_calibration-rtx4090

- Question: does the full grid pipeline run on a real GPU, and what do ESM2 and
  ProstT5 cost per call, for sizing the μ/τ/λ grid?
- Code: commit fefe226 (`grid_plan.json`); fleet run `cal-1` (`fleet.json`).
- Hardware: 1 × RTX 4090 24 GB, RunPod Secure Cloud, $0.89/h, template
  `runpod-torch-v240` (torch 2.4.1+cu124, Python 3.11), transformers 4.44.2.
- Raw data: `results/runpod/cal-1/grid` (local, ignored); pod deleted.
- Deviations: grid ran via `fleet.py launch --attach` and `redeploy` while
  launcher bugs were fixed (see below). The ProstT5 batch sweep ran on the same pod
  after DONE, using `scripts/runpod/bench_prostt5.py`.

## Findings

Stage wall times (`pod_shard.log`), one μ=1, τ=1.5 cell:

| Stage | Wall time | Notes |
|---|---|---|
| bootstrap (first) | ~6 min | pip, 13 GB weights (ESM2 + ProstT5), Q.3Di.AF, IQ-TREE |
| forward | 63 s | 4,777 ESM2 calls in 733 sequential passes, median batch 3; 100 calls/s; ~65 ms per pass |
| encode | 58 s | 77 nodes, ProstT5 batch 32 |
| asr | 140 s | AA + 3Di IQ-TREE, ModelFinder |
| sample | 781 s | 20 steps × 74 chains (37 nodes × λ .02/.1); 29–34 s per step |

The cell is reconstructed as in the historical runs: AA MAP accuracy 97.83%
(PMB+R3), 3Di 89.87% (Q3DI_AF+I+G4), 37/37 nodes matched. The 20 sampler steps
are a timing run, not inference: summed AA errors over 37 nodes moved 317 → 318
(λ=.02) and 317 → 324 (λ=.1), with acceptance ≈ .6.

ProstT5 greedy <AA2fold> on 396-residue sequences (fp16, RTX 4090):

| Batch | Seconds | Sequences/s | Peak GPU memory |
|---|---|---|---|
| 8 | 9.5 | 0.85 | 11.0 GB |
| 32 | 19.3 | 1.66 | 22.4 GB |
| 64 | out of memory on 24 GB | | |

**ProstT5 decoding is the cost driver, and GPU memory limits batch size.**
Memory grows by about 0.4 GB per sequence in a batch (attention caches over 396
decoding steps), so a 24 GB card tops out near batch 32 and about 1.7
encodings/s. The earlier "~6 GB, memory is not the constraint" assumption was
wrong. ESM2 forward simulation is cheap and limited by sequential passes.

### Implication for `grids/example.toml` (270 chains × 1000 steps)

About 270k chain-steps. Shared λ starts are encoded once, but chains diverge
within a few steps (37 → 55 distinct sequences of 74 after 11 steps here), so
expect roughly 200–250k encodings.

- On 24 GB cards at ~1.7/s, that is ~35–40 GPU-hours. That costs about $12–14 at
  Community 4090 prices ($0.34/h, rarely in stock today) and about $35 at the
  Secure 4090 price ($0.89/h).
- Wall time is at least 1000 sequential steps. With 9 pods (30 chains each, one
  batch of ≤32 per step, ~19 s) it takes ~5–6 h. With 18 pods (15 chains, ~12 s
  per step) it takes ~3.5 h, at lower batch efficiency and higher cost.
- 80 GB cards do not help: the A100 plateaus at ~1.8 encodings/s at batch 64–128
  (`2026-11-02_calibration-a100`), so cheap 24 GB cards at batch 32 are the
  most cost-effective option.

## Launcher issues found and fixed during this run

1. Pods created from the bare image never booted within 10 min. Pods created
   from the official template booted in ~40 s.
2. `runpodctl ssh info` (v3.0.1) kept reporting "no container" for running pods.
   SSH details are now read from `pod get`.
3. Pod `/workspace` refuses chown, so rsync uses `-rlt` and tar uses `--no-same-owner`.
4. The detached job's log redirect ran before its directory existed, so the job
   silently never started. The backgrounded `&&` chain also held the SSH session
   open. Fixed, and the job string is now executed in a test.
5. Cheap Community stock (3090/4090) was exhausted for most of the session, so
   the launcher fell back to Secure Cloud.

Cost: about $1.50 in total, including ~35 min of pods that never started and probes.
