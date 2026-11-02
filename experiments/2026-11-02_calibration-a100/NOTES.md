# 2026-11-02_calibration-a100

- Question: do 80 GB cards (larger ProstT5 batches) give more encodings per
  dollar than 24 GB cards? Compare with `2026-11-02_calibration-rtx4090`.
- Code: commit c32d89b; fleet run `cal-a100`, `--model-batch-size 128`.
- Hardware: 1 × A100 80GB PCIe, RunPod Community Cloud, $1.19/h, template
  `runpod-torch-v240`. Launched cleanly by `fleet.py` with no intervention.
- Raw data: `results/runpod/cal-a100/grid` (local, ignored); pod deleted.
- Deviations: none in the grid run; ProstT5 batch sweep afterwards on the same pod.

## Findings

| Stage | A100 | RTX 4090 run |
|---|---|---|
| bootstrap (fresh pod) | 182 s | ~6 min |
| forward | 93 s | 63 s |
| encode | 88 s | 58 s |
| asr | 311 s | 140 s |
| sample (20 steps) | 754 s, ~37 s/step for ~65 distinct sequences in 1 batch | 781 s, ~29–34 s/step in 2 batches of ≤32 |

ProstT5 batch sweep, A100 80GB, fp16:

| Batch | Seconds | Sequences/s | Peak GPU memory |
|---|---|---|---|
| 8 | 16.5 | 0.48 | 11.0 GB |
| 32 | 21.0 | 1.52 | 22.4 GB |
| 64 | 36.5 | 1.75 | 37.5 GB |
| 128 | 69.8 | 1.83 | 67.6 GB |
| 192 | out of memory on 80 GB | | |

**Throughput plateaus at ~1.8 encodings/s per GPU on both cards.** Past batch
~32, time grows linearly with batch, so this transformers T5 greedy decode is
compute-bound per generated token. Per dollar, a 24 GB card at batch 32 is the
best option:

| Card | Price | Encodings per $ |
|---|---|---|
| 4090 Community | $0.34/h | ~17.6k |
| 3090 Community | $0.22/h | probably more, if similar speed (unmeasured) |
| 4090 Secure | $0.89/h | ~6.7k |
| A100 Community | $1.19/h | ~5.5k |

Reconstruction (this run's own simulation): AA MAP 97.93% (PMB+R3), 3Di 90.48%
(Q3DI_AF+I+G4), 37/37 nodes matched.

### Same seed, different GPU, different simulation

The forward simulation (same commit, seed and batch settings) differs between
the 4090 and the A100. In 5 edges whose parent sequence was still identical,
the same site was drawn but a different residue was accepted, e.g.
Root→Internal_0022 at event 125: A135C on the 4090 vs. A135A on the A100. That
is 5 flips in 4,777 draws (~0.1%). It comes from fp16 ESM2 probabilities
differing slightly between GPU architectures (METHODS §6 anticipates this). Each
flip changes the rest of that subtree's trajectory: 36 of 77 nodes differ, by
20 sites on average (max 77).

Seeds reproduce exactly only on the same GPU type and batch layout. Across GPU
types, treat runs as statistical replicates and record the GPU (`fleet.json`
records it; the simulation manifest does not yet).

Cost: ~$0.62.
