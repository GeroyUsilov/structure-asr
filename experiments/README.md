# Experiment records

One directory per real run (usually a RunPod grid), named `YYYY-MM-DD_<short-name>/`.
Records are small and committed. Raw outputs (simulations, traces, checkpoints)
stay on the RunPod network volume or in object storage, never in git.

Create a record from a finished (or partial) grid output directory:

```bash
bash scripts/record_run.sh grids/my_grid.toml 2026-11-02_calibration-a100 [--outdir <grid outdir>]
```

It copies:

- the grid TOML
- `grid_plan.json`
- `batch_timings_*.tsv` and `sample_timings_*.tsv`
- `evaluation/*_summary.tsv` and `evaluation/node_mapping.tsv`
- every cell's `simulation/manifest.json`, `three_di/manifest.json`,
  `asr/*.asr_manifest.json` and `sample/*/run_manifest.json`

Then fill in `NOTES.md`:

```markdown
# <name>

- Question: what this run was meant to answer
- Code: commit <sha> (from grid_plan.json / `git rev-parse HEAD` on the pod)
- Hardware: GPU type and count, pod image, wall time per stage
- Raw data: where the full outputs live (volume path / bucket)
- Deviations: anything run differently from the TOML or WORKFLOWS

## Findings

What the summaries show, with the numbers. Keep the claims within what was
measured: fidelity to the true ancestor (AA recovery and 3Di structural fidelity) is the endpoint, and score gains alone don't count.
```

Historical (pre-repository) runs are described in `docs/EXPERIMENTS.md` and the
bundled `examples/`; they are not re-recorded here.
