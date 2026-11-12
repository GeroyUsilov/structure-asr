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

- Grid stages (13 tests in `tests/test_grid.py`; 44 main-suite tests total with
  the 4 fleet tests below):
  - With a deterministic fake predictor, batched multi-lane forward simulation
    reproduces serial `ForwardSimulator` nodes, node order, events and scores
    for batch sizes 1/3/64.
  - An interrupted grid resumes to the uninterrupted result. Two shards equal
    one process. Plan counts equal `exact_planned_attempts` and the cap is enforced.
  - The ASR stage, run with an IQ-TREE stub, succeeds with every truth file
    deleted. It skips finished jobs and refuses outputs made from changed inputs.
  - The sample stage, run on stub-ASR output with truth deleted, uses a 3Di tree
    whose node names differ from the AA tree's. The pairing is recovered by topology.
    After an interruption it refuses to restart without `--resume`, then resumes.
    Every (cell, λ) run's trace, chain summary, final/best FASTAs and site
    weights equal a standalone `structure_asr.sampling` run (fake encoder).
    Shared starts are encoded once; `sample_summary.tsv` maps every node.
  - `[sample]` validation rejects unknown keys, duplicate or negative λ,
    empty node lists and invalid start modes; `plan` counts chains and chain-steps.
  - The opt-in real-IQ-TREE test runs only when `ASR_IQTREE` is set. It passes
    with IQ-TREE 3.1.4 (official macOS arm64 release, sha256-verified).
- Structural selection (`[fold]`, METHODS 7b) is covered on CPU with a fake
  residue-wise 3Di map: a permissive threshold reproduces the legacy generator's
  sequences and events exactly, an impossible threshold reverts every block and
  leaves every node at the root, selection raises the tips' minimum 3Di identity
  above both the threshold and the unselected run, block size sets the number of
  predictions, `FoldSettings` separates checkpoints, and a `[fold]` grid without
  an encoder is rejected. **Never run on a GPU or with real ProstT5**, so the
  realized divergence, acceptance rates and cost of a real `[fold]` cell are
  unmeasured.
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

- 2026-11-01: refactoring `structure_asr.sampling.main` onto the shared
  `SamplerRun`/`advance_runs` code left its outputs byte-identical to the
  previous commit (fake encoder, 25 steps; two nodes x two chains at λ=.05 and
  model batch 3; `aa_sample` start at λ=0, seed 11), apart from manifest timestamps.

- RunPod fleet tooling (`tests/test_fleet.py`, fake runner, no RunPod calls):
  SSH-info parsing rejects the scp-incapable proxy; GPU/cloud fallback order
  skips unavailable types; the remote job is valid shell for the right shard;
  launch → sync deletes pods only after their `DONE` marker syncs; `--attach`
  adopts a running pod; the detached job string runs and returns at once.
- First GPU run, 2026-11-02 (`experiments/2026-11-02_calibration-rtx4090/`):
  bootstrap, forward, encode, asr and sample ran on an RTX 4090 pod via
  `fleet.py`; evaluation ran locally (37/37 nodes, AA MAP 97.83%). ProstT5
  greedy decoding costs ~19 s per batch of 32 at 22.4 GB peak memory; batch 64
  does not fit in 24 GB. On an A100 80GB (`experiments/2026-11-02_calibration-a100/`)
  throughput plateaus at ~1.8 encodings/s (batch 64–128), so cheap 24 GB cards
  give the most encodings per dollar.
- Cross-GPU reproducibility: the same seed and commit gave different forward
  simulations on RTX 4090 and A100. In 5 of 4,777 draws, with identical parent
  sequences, a different residue was accepted at the same site (fp16 ESM2
  numerics), and each flip changed the downstream subtree (36 of 77 nodes
  differ). Treat seeds as exactly reproducible only on the same GPU type and
  batch layout.

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

### 0.3.0 package consolidation

Code now lives in one import package, `structure_asr`. Before this release it
was spread across five top-level packages:

| Old location | New location |
|---|---|
| `forward_asr_benchmark.*` | `structure_asr.simulation.*` |
| `runpod_exact_mlm_ef_tu_sweep.analyze_3di_hamming` / `analyze_aa_hamming` / `analyze_temperature_sweep` | `structure_asr.analysis.hamming_3di` / `hamming_aa` / `temperature_sweep` |
| `runpod_exact_mlm_ef_tu_sweep.convert_ground_truth_nodes_to_3di` | `structure_asr.analysis.encode_truth_nodes` |
| `runpod_exact_mlm_smoke.run_smoke` (+ `root.fasta`, `tiny_tree.nwk`) | `structure_asr.smoke.run_smoke` |
| `runpod_paper_tree_ofs.*` | `structure_asr.ofs.*` |

Each old module path is still a shim. It replaces itself in `sys.modules` with
the new module, so imports, private names, monkeypatching and `python -m` keep
working. The earlier `import *` shims (`convert_tips_to_3di`,
`evaluate_asr_ground_truth`, `run_sweep`, `esm2_adapter`) are unchanged.

Generated configs deliberately keep writing the old factory strings
(`runpod_exact_mlm_smoke.esm2_adapter:make_site_predictor`,
`runpod_paper_tree_ofs.ofs_adapter:make_ofs_scorer`). Those strings feed the
forward-checkpoint fingerprint, so changing them would block resuming existing
checkpoints. No function bodies changed, apart from import lines.

Equivalence evidence (rule 4), comparing 0.2.0 (`73942b6`) with this
refactor:

- `scripts/check.sh` passes: 55 tests (1 skipped), plus the example replay.
- A fake-model run produced 240 files: a grid through forward, fake-encoder 3Di,
  stub IQ-TREE ASR, joint sampling over λ ∈ {0, .5} and evaluation; the
  standalone sampler; and the `forward-asr simulate` CLI with uniform and
  exact-MLM proposals, launched through the old `python -m
  forward_asr_benchmark.cli` path. 220 files are byte-identical. The other 20
  differ only in `created_utc` or timing columns. These are the same 20 files
  that differ between two runs of unchanged 0.2.0 code.
- CPU replay: every TSV and PNG is byte-identical. The 8 PDF/SVG files differ
  only in their embedded creation date.
