# Import provenance

Imported 2026-10-28. The source directories were not Git repositories.
`main` records this curated baseline; `refactor/documented-handoff` records
the cleanup. Originals remain unchanged. No remote repository is configured.

Source root: `/Users/christopher/Desktop/aleks_claude`.

| Imported destination | Original, relative to source root |
|---|---|
| `src/forward_asr_benchmark` | `forward_asr_benchmark/src/forward_asr_benchmark` |
| `src/runpod_*` and `tests/test_*` | Corresponding directories in `forward_asr_benchmark` |
| `src/structure_asr/sampling.py` | `asr_synthetic_ground_truth_evaluation/runpod_uncertainty_weighted_node34_tau2/run_joint_asr_mcmc.py` |
| `src/structure_asr/legacy_sampling.py` | `asr_synthetic_ground_truth_evaluation/runpod_joint_asr_jobs/base/run_joint_asr_mcmc.py` |
| `src/structure_asr/score_placement.py` | `asr_synthetic_ground_truth_evaluation/score_placement/analyze_score_placement.py` |
| `examples/ef_tu/reference` | `forward_asr_benchmark/reference_data/ef_tu_uiuc/extracted` (tree and alignment only; README from parent) |
| `examples/ef_tu/mu1_tau*/{aa,3di}.{state,treefile}` | `asr_synthetic/ef_tu_tips_<condition>_<modality>.fasta.{state,treefile}` |
| `examples/ef_tu/mu1_tau*/truth_aa.fasta` | `asr_synthetic_ground_truth_evaluation/ground_truth_inputs/ground_truth_<condition>_aa.fasta` |
| `examples/ef_tu/mu1_tau*/truth_3di.fasta` | `asr_synthetic_ground_truth_evaluation/ground_truth_3di_tau1p5_tau2/ground_truth_<condition>_3di.fasta` |
| `examples/ef_tu/mu1_tau*/legacy_results` | `runpod_joint_asr_<condition>_results` (FASTAs, summary and manifest; large traces/checkpoints left in source) |
| `examples/ef_tu/node_mapping.tsv` | `asr_synthetic_ground_truth_evaluation/node_mapping.tsv` |
| `examples/glugln` scripts and inputs | `runpod_glugln_aa168_3di16_uncertainty` |
| `examples/glugln/site_uncertainty.tsv` | `results_glugln/GluGln_AA168_3Di16/site_uncertainty.tsv` |
| `legacy/{protein_mcmc_utils,greedy_utils}.py` | Same filenames in source root |

The two `plot_glugln_*.py` scripts came from the calling workspace:
`/Users/christopher/Documents/Codex/2026-08-19/can-you-navigate-to-the-aleks`.
Archived README files preserve historical instructions, not current entry points.

Excluded: model weights/caches, the 285 MB reference archive, duplicate job
bundles, notebooks duplicating these scripts, most large result tables and
unrelated datasets. Full source artifacts still reside under the source root.
This is a curated handoff, not a lossless backup of the Desktop directory.
