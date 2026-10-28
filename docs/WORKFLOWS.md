# Running and extending the workflows

Run commands from the repository root after the README installation. Outputs
below are ignored by Git. Set `HF_HOME` to persistent RunPod storage if desired;
reuse a known-good cached model through `--model /path/to/snapshot` rather than
downloading duplicate weights. Do not run GPU steps on the local machine.

## Forward evolution

```bash
bash scripts/forward_ef_tu.sh --outdir results/mu123_tau1p5 \
  --rates 1 2 3 --proposal-temperature 1.5 --plan-only
```

Remove `--plan-only` to execute. For τ=2 use a different output directory.
`--seed` defaults to 731024, shared across rates; model defaults to ESM2-650M.
`--max-attempts-per-run` defaults to 100000. The plan refuses larger jobs
unless that cap is raised deliberately. Rates 5/10/20 require respectively
23896/47802/95614 calls for this tree and seed. At the previously measured
.04 s/call, those three rates at one temperature imply about 1.86 hours
of forward passes alone; this is a sizing example, not a queued run or a new GPU measurement.

Per-rate outputs:

```text
results/<sweep>/
  root_sequence.fasta, sweep_plan.json, sweep_summary.json
  mu_1/
    config.toml, checkpoint.json, model_pass_timings.tsv
    simulation/
      ground_truth_nodes.fasta, leaves.fasta, tree.nwk
      events.tsv, node_scores.tsv, manifest.json
```

Rerun the same command to resume completed branches. Do not change the model,
seed, rate or temperature in an existing result directory. See `VALIDATION.md`
for inherited interruption edge cases.

## AA→3Di conversion (GPU)

Tip conversion reuses the working full ProstT5 path:

```bash
python -m structure_asr.prostt5 --results-dir results/mu123_tau1p5 \
  --rates 1 2 3 --output-archive results/mu123_tau1p5_3di.tar.gz --resume
```

For the historical μ=1 all-node truth cases:

```bash
python -m runpod_exact_mlm_ef_tu_sweep.convert_ground_truth_nodes_to_3di \
  --condition-input mu1_tau1p5=examples/ef_tu/mu1_tau1p5/truth_aa.fasta \
  --condition-input mu1_tau2=examples/ef_tu/mu1_tau2/truth_aa.fasta \
  --outdir results/truth_3di --resume
```

Those encodings are already bundled. Re-running conversion is only necessary
for new sequences or an explicit reproducibility study. The all-node CLI is
historically restricted to μ=1 condition names, 77 nodes and length 396.
Its restriction should not be silently bypassed.

## Multi-regime grids (forward → 3Di → IQ-TREE ASR → evaluation)

`asr-grid` (`structure_asr.grid`, wrapper `scripts/grid.sh`) runs a grid of
(μ, τ, seed) cells on one tree and root, defined in a TOML file such as
`grids/example.toml`. Paths in the TOML are relative to that file. Each stage
is a separate command, resumable, and skips completed cells:

| Stage | Where | Reads | Writes per cell |
|---|---|---|---|
| `plan` | CPU | grid, tree, root | `grid_plan.json` (exact call counts, cap check) |
| `forward` | GPU | — | `simulation/` (same layout as `asr-evolve`), `checkpoint.json` |
| `encode` | GPU | `simulation/ground_truth_nodes.fasta` | `three_di/leaves_3di.fasta` (inference input), `truth/ground_truth_nodes_3di.fasta` |
| `asr` | CPU | tip FASTAs, label-free topology only | `asr/{aa,3di}.{state,treefile,iqtree,log}`, `*.asr_manifest.json` |
| `evaluate` | CPU | ASR outputs + truth | `evaluation/{dataset,node}_summary.tsv`, mappings, site table |
| `status` | any | — | prints per-cell completion |

Cells live in `<outdir>/mu_<r>__tau_<t>__seed_<s>/`. `forward` batches one
masked ESM2 pass over up to `--max-batch` edges drawn from all cells (see
METHODS §6). `--shard i/n` splits cells round-robin across GPUs or pods that
share a network volume; shards write disjoint directories. `plan` reports a
per-cell lower bound on sequential passes (μ·L·deepest root-to-tip length), which
no batch size can beat. Use it with `batch_timings_*.tsv` to size runs.

RunPod recipe:

```bash
# Network volume holds the checkout, HF_HOME, .cache/ and results/.
bash scripts/grid.sh plan grids/my_grid.toml
bash scripts/grid.sh forward grids/my_grid.toml --validate-only
bash scripts/grid.sh forward grids/my_grid.toml --max-batch 64 [--shard 0/2]
bash scripts/grid.sh encode grids/my_grid.toml --batch-size 32 --resume
# CPU pod (or the GPU pod's spare cores), same volume:
bash scripts/fetch_3di_matrix.sh
#   install IQ-TREE 3.1.4 from the official release (Linux-intel or Linux-arm
#   tarball, verify the GitHub-listed sha256), unpack under .cache/tools/,
#   then export ASR_IQTREE=$PWD/.cache/tools/iqtree-3.1.4-Linux-intel/bin/iqtree3
bash scripts/grid.sh asr grids/my_grid.toml --jobs "$(nproc)"
bash scripts/grid.sh evaluate grids/my_grid.toml
```

Before a large grid, run a tiny one (e.g. `rates = [0.1]`, one τ and seed) end to
end. Compare `--max-batch 1` with `--max-batch 64` output directories to measure
how often batched fp16 numerics change a draw.

`reconstruct` and `evaluate-dir` run the ASR and evaluation stages on one
existing tip dataset, for example the bundled tips:

```bash
bash scripts/grid.sh reconstruct \
  --aa-tips examples/ef_tu/tips/aa/ef_tu_tips_mu1_tau2_aa.fasta \
  --three-di-tips examples/ef_tu/tips/3di/ef_tu_tips_mu1_tau2_3di.fasta \
  --tree examples/ef_tu/reference/AB_eftu_rerooted.tre \
  --three-di-matrix .cache/models/3di/Q.3Di.AF --outdir results/iqtree_local/mu1_tau2
bash scripts/grid.sh evaluate-dir --asr-dir results/iqtree_local/mu1_tau2 \
  --tree examples/ef_tu/reference/AB_eftu_rerooted.tre \
  --truth-aa examples/ef_tu/mu1_tau2/truth_aa.fasta \
  --truth-3di examples/ef_tu/mu1_tau2/truth_3di.fasta \
  --condition mu1_tau2 --tau 2 --outdir results/iqtree_local/mu1_tau2/evaluation
```

## ASR posterior inputs

Sampling consumes IQ-TREE-style `.state` tables: tab-separated `Node`, `Site`,
`State`, and 20 `p_<symbol>` columns. Each node needs consecutive 1-based sites.
AA and 3Di alignments must refer to the same positions. Tree files are needed
for benchmarking mappings; do not join independently reconstructed nodes by
number without verifying their topology.

`asr-grid asr`/`reconstruct` produce these files (METHODS §7). The *historical*
`examples/ef_tu/mu1_tau*` trees and posteriors predate that stage. Their IQ-TREE
command and model are unrecorded, so they are inputs, not evidence of the
current configuration.

## Current uncertainty-weighted sampling

```bash
bash scripts/sample_node34.sh 0.01 --validate-only
bash scripts/sample_node34.sh 0.01
```

Use .01/.02/.05/.1 on separate RunPod machines to reproduce the historical independent
1000-step, one-chain experiments. Each job uses the same seed/start by default
and changes only the structural weight. Results contain the starting AA and
predicted 3Di, best/final sequences, per-step scores, acceptance/proposal values,
site weights, checkpoint and manifest. Model-score selection happens inside
sampling; ground-truth evaluation is a subsequent operation.

For other nodes/chain counts, use `python -m structure_asr.sampling --help`.
Use `--nodes-file examples/ef_tu/representative_nodes.txt` for Node1/29/34.

## Historical sampler reproduction

```bash
python -m structure_asr.legacy_sampling \
  --aa-state examples/ef_tu/mu1_tau2/aa.state \
  --three-di-state examples/ef_tu/mu1_tau2/3di.state \
  --nodes-file examples/ef_tu/representative_nodes.txt \
  --condition mu1_tau2 --outdir results/legacy_mu1_tau2 \
  --alpha 0.5 --mcmc-temperature 2 --n-iter 3000 \
  --chains-per-node 1 --acceptance-mode legacy --resume
```

Use this only to reproduce the historical algorithm. The default modern entry
point `asr-sample` uses the uncertainty-weighted Hastings implementation.
Neither command evaluates truth. The curated checkout contains truth files;
omit them when constructing a physically blinded inference-only bundle.

## CPU result replay and figures

```bash
bash scripts/reproduce_examples.sh
```

Individual commands:

```bash
python -m structure_asr.benchmark_examples
python -m structure_asr.score_placement --example-root examples/ef_tu \
  --mapping results/reproduced/node_mapping.tsv --outdir results/score_placement
python -m structure_asr.example_distances
python examples/glugln/plot_glugln_model_entropies.py
python examples/glugln/plot_glugln_sequence_logos.py
```

The seven-rate τ=1 tip files are retained separately in
`examples/ef_tu/rate_sweep_tau1`; historical general Hamming analysis entry
points are `runpod_exact_mlm_ef_tu_sweep.analyze_aa_hamming` and
`runpod_exact_mlm_ef_tu_sweep.analyze_3di_hamming` (see their `--help`).

## Other sampling processes retained

- `examples/glugln/run_corrected_glugln.py`: draw independent AA posterior
  sequences, encode them, compare empirical 3Di frequencies with direct 3Di
  marginals. Run `--validate-only` locally; GPU execution on RunPod.
- `legacy/runpod_3di_to_aa_sampling_v3/`: preserved reverse `<fold2AA>` workflow,
  fixed 3Di marginal-MAP translated to stochastic AA samples with sequence-level
  rejection of invalid tokens. Its original README explains sampling settings
  and resume/import behavior. This is not joint phylogenetic MH.
- `legacy/protein_mcmc_utils.py`, `legacy/greedy_utils.py`: original candidate
  scoring, forward-chain, mutational scan and greedy-selection helpers.
- `runpod_paper_tree_ofs` plus core DCA/AlphaFold scorers: experimental historical
  options retained for reference, not active EF-Tu result provenance.
