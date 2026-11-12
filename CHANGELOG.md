# Changelog

## 0.3.0 — package consolidation (unreleased, on `chore/restructure`)

- All code now lives in one import package, `structure_asr`. It has subpackages
  for the forward simulator (`simulation`), historical EF-Tu analyses
  (`analysis`), the OFS sweep (`ofs`) and the ESM2 smoke test (`smoke`).
- `forward_asr_benchmark` and the `runpod_*` packages are now alias shims.
  Old imports, `python -m` paths and saved config factory strings still work.
- Added `environment.yml` (conda, CPU), this changelog, and pytest settings.
  Rewrote the README around the scientific method and current findings.
- Endpoints now explicitly include structural fidelity to the true ancestor
  (3Di), alongside AA recovery (AGENTS.md rule 5, experiments/README.md).
- Committed the `2026-11-02_tau-low-lambda` experiment record.
- No algorithm, default or output format changed. Equivalence was checked against
  0.2.0 (byte-identical outputs; see `docs/VALIDATION.md`, "0.3.0 package consolidation").

## 0.2.0 and earlier

See the git history; experiment records are in `experiments/`.
