# Working on this repository

Research code for structure-informed ancestral sequence reconstruction (see
README). Reference docs, kept current: `docs/METHODS.md` (exact algorithms and
conventions), `docs/WORKFLOWS.md` (commands, RunPod, outputs),
`docs/VALIDATION.md` (what is tested; known limitations). Read the relevant one
before changing an algorithm or a workflow. `docs/archive/` is history.

## Scientific rules

1. Never use ground truth during inference, proposal or node/chain selection,
   or hyperparameter tuning. Evaluation is a separate post-inference stage, and
   only evaluation code reads truth files.
2. Pair AA and 3Di nodes (and inferred and true nodes) by topology, never by
   node number. Glu/Gln deliberately pairs AA Node168 with 3Di Node16; EF-Tu has
   its own mapping.
3. Keep historical provenance distinct: legacy vs. uncertainty-weighted sampling
   kernels, α vs. λ score scales, batch-1 vs. batched forward runs. Don't
   relabel a new result as a reproduction of an old one.
4. A change to numerical behavior, defaults or data needs a doc update
   (METHODS/WORKFLOWS/VALIDATION). A behavior-preserving refactor needs evidence:
   byte-identical fake-encoder outputs or an equivalence test.
5. Endpoints are fidelity to the true ancestor: AA recovery, and structural
   fidelity (3Di of the reconstruction vs. the true ancestor's 3Di). A gain in
   either is worth reporting; report both, and say when one trades against the
   other. Higher joint scores S_λ are not improvements by themselves.

## Engineering rules

6. GPU work (ESM2, ProstT5) runs on RunPod. Don't download or run those models,
   or train encoders, locally unless asked. CPU tests use deterministic fakes.
   New dependencies or model downloads get RunPod instructions, not local
   installs. Don't bypass model-loading security checks.
7. `examples/` is immutable input. Outputs and caches go in ignored `results/`
   and `.cache/`. Never modify the Desktop originals.
8. Long jobs must checkpoint and resume, validate inputs without loading GPU
   models (`--validate-only`), and show progress. Test resume when touching them.
9. All code lives in `structure_asr`; `forward_asr_benchmark` and `runpod_*`
   are thin alias shims (VALIDATION "0.3.0 package consolidation"). Keep the
   old `runpod_*` import paths and config factories working until the
   user retires them.
10. Before committing, run `bash scripts/check.sh` (tests + CPU example replay;
    `--quick` for tests only).

## Git and experiments

- `main` is the trunk. Do feature work on short branches (`feat/…`, `fix/…`,
  `chore/…`) and merge when `scripts/check.sh` passes. Commit only when asked.
- The remote is a private GitHub repo. RunPod grids run through
  `scripts/runpod/fleet.py`: self-contained pods (no network volume) get a git
  bundle of the committed HEAD; `grid_plan.json` records the commit. Confirm
  with the user before launching pods, and make sure none are left running.
- Each real run gets a small committed record in `experiments/YYYY-MM-DD_<name>/`
  (`scripts/record_run.sh`, see `experiments/README.md`). Raw outputs stay on
  the RunPod volume.
- Work on what the user asks for now. Ideas listed in the docs are not a queue.
