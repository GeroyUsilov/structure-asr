# Validation and inherited limitations

Validation performed locally on 2026-10-28, Python 3.12, without downloading or
loading ESM2/ProstT5 and without running IQ-TREE.

## Checked

- Original 21 CPU tests pass (evolution, per-branch resume, DCA/OFS scorer
  interfaces, Hamming distances, batching, configuration and timing recovery).
- Six additional tests cover shared posterior loading, non-self proposal
  probabilities, exact small-state detailed balance, both samplers' interrupted
  resume, completed resume, length rejection and historical benchmark replay
  (some checks are grouped inside one test method): 27 tests before the grid stage.
- Both samplers' fake-encoder traces, checkpoints, best FASTAs and score tables
  match the original Desktop implementations byte-for-byte in a fixed-seed
  smoke run. The fake encoder maps AA strings directly to test 3Di strings;
  it tests software behavior, not structural predictions.
- Known-tree mappings reproduce 37 matched inferred nodes in each condition
  and modality; historical AA results reproduce 68→75 errors across six nodes.
- Complete CPU figure replay succeeds, including 5000 posterior samples per
  node, nine-condition distance ECDFs, Glu/Gln 307-site entropy validation and
  the selected 163–172 logo window.
- Plan-only higher-rate generation and Node34 `--validate-only` succeed.
- The package wheel builds offline with `--no-deps --no-build-isolation`.
  The full checkout, rather than the wheel alone, supplies example data.

- Grid stages (11 tests in `tests/test_grid.py`, 38 main-suite tests total):
  - With a deterministic fake predictor, batched multi-lane forward simulation
    reproduces serial `ForwardSimulator` nodes, node order, events and scores
    for batch sizes 1/3/64.
  - An interrupted grid resumes to the uninterrupted result. Two shards equal
    one process. Plan counts equal `exact_planned_attempts` and the cap is enforced.
  - The ASR stage, run with an IQ-TREE stub, succeeds with every truth file
    deleted. It skips finished jobs and refuses outputs made from changed inputs.
  - The opt-in real-IQ-TREE test runs only when `ASR_IQTREE` is set. It passes
    with IQ-TREE 3.1.4 (official macOS arm64 release, sha256-verified).
- `asr-grid plan grids/example.toml` reproduces the historical μ=1 (4,777) and
  documented μ=5 (23,896) call counts.
- Real local IQ-TREE 3.1.4 run (`reconstruct` + `evaluate-dir`) on the bundled
  μ=1 τ=1.5/2 tips, using the reference topology and Q.3Di.AF for 3Di.
  - All 37 nodes matched in each condition and modality.
  - AA: ModelFinder chose PMB+R3 (τ=1.5) and PMB+G4 (τ=2). MAP accuracy was
    97.9047% for both, with mean p(truth) .97261 and .97095, equal to the
    historical values at the printed precision.
  - 3Di: ModelFinder chose Q3DI_AF+I+G4 and Q3DI_AF+R4. MAP accuracy was 88.70%
    and 79.60%, with mean p(truth) .88409 and .78786.
  - Per-site MAP states agree with the historical `.state` files at 100% for
    AA (both τ) and 3Di τ=1.5, and at 99.04% for 3Di τ=2.
  - This is consistent with, but does not prove, the historical runs having
    used ModelFinder for AA and Q.3Di.AF for 3Di. Outputs are in
    `results/iqtree_local/`. Runtime was about 1 minute per condition on 4 threads.

These checks do not establish GPU performance, new model compatibility, sampler
convergence, improved AA recovery or a completed new-data ASR benchmark.

## Known inherited limitations — retained to preserve experiment behavior

1. **Model dependencies.** `requirements-gpu.txt` preserves historical pins,
   including transformers 4.44.2. It is not a refreshed security/compatibility
   audit. `materialize_model` may download more than one weight format from a
   model repository. Use an existing verified snapshot and ample persistent
   disk; previous failures included disk quota, old torch `.bin` loading, and
   safetensors metadata incompatibility. No bypass or new loader hack was added.
2. **Simulation output interruption.** Branch checkpoints are robust to
   interrupted edges, but a crash during final output-directory creation may
   leave a partial `simulation/` directory that blocks the old runner. Preserve
   and inspect it before recovery; do not delete completed runs automatically.
3. **3Di JSONL tail.** Loading skips an incomplete last checkpoint line, but
   append-after-truncation recovery is not fully hardened. Keep a copy and repair
   the truncated tail before appending; add a regression test when fixing this.
4. **Tree serialization.** Historical Bio.Phylo Newick writing can combine
   generated internal names with confidence labels and rounds branch lengths.
   Use the original reference tree and `load_tree` for truth mappings and
   exact branch-length plans, not a round-tripped simulation tree. Changing
   serialization also changes legacy checkpoint fingerprints.
5. **Sampling resume identity.** Existing checkpoints compare configuration,
   not input-file contents. Do not edit posterior tables or swap model snapshots
   in place during a resumed chain. At completed resume, missing final files
   are not automatically rebuilt if a crash happened after checkpoint writing.
6. **Generic input hardening.** Shared sampling loaders preserve historical
   flooring/normalization behavior. They reject negative/nonfinite values and
   nonconsecutive sites, but do not comprehensively reject every malformed
   probability table (e.g. all-zero rows). Bundled inputs are validated and
   normalized. Strengthening this should be a separately tested change.
7. **Compatibility shims.** Old public imports and CLI/factory paths work;
   private module globals/monkeypatch targets are now in `structure_asr`.
   Update external tests that patched the old module's private objects.
8. **Blinding.** The samplers never read truth. This full research checkout
   contains truth for post-inference evaluation and is not an access-controlled
   blind experiment. A future inference-only job archive should exclude it.
9. **Stochastic reproducibility.** Saved seeds/RNG states preserve tested CPU
   control flow. Bitwise GPU results across different libraries/hardware are
   not guaranteed. Preserve environment and model revision in future runs.

## Refactor boundary

Moved active forward/model/evaluation modules to named `structure_asr` modules,
retained old import shims, removed duplicated posterior code, made package
installation include workflows, corrected Python requirement to 3.11+, replaced
Glu/Gln absolute data paths with relative paths and optional font fallback,
added replay entry points, examples, tests and documentation. The numerical
simulation/sampling kernels, RNG order and model generation settings are unchanged.
