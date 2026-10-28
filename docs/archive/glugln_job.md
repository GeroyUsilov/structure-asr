# Corrected Glu/Gln uncertainty run

This bundle performs the intentional cross-tree comparison:

- amino-acid posterior: `AAseq.state`, **Node168** (`inputs/AAseq_Node168.csv`)
- direct 3Di posterior: `3Di.state`, **Node16** (`inputs/3Di_Node16.csv`)

Both inputs contain 307 matching alignment sites and 20 probability columns. This
bundle does not run Node11 or Node245, because their previous analyses did not use
the mistaken Glu/Gln AA-node mapping.

## Run on the pod

The pod does not need the `unzip` command. From `/workspace`:

```bash
cd /workspace \
&& python -m zipfile -e runpod_glugln_aa168_3di16_uncertainty.zip . \
&& cd runpod_glugln_aa168_3di16_uncertainty \
&& bash run_on_runpod.sh 2>&1 | tee run.log
```

The first run downloads/caches ProstT5 and installs Python dependencies. During
inference, progress lines look like:

```text
GluGln_AA168_3Di16: encoded 20/1000 samples
```

The run is resumable. If it stops, execute `bash run_on_runpod.sh` again in the
same directory. Set `N_SAMPLES`, `BATCH_SIZE`, `BOOTSTRAP`, or `SEED` before the
command to override defaults. Do not change `N_SAMPLES` inside an existing output
directory.

## Outputs

Results are written to the new, correction-specific `results/` directory. Key
files include:

- `node_summary.tsv`
- `all_sites_uncertainty.tsv`
- `high_disagreement_sites.tsv`
- `entropy_distributions.png`
- `entropy_scatter.png`
- `per_site_uncertainty.png`
- `js_divergence_by_site.png`
- `run_manifest.json`
- `GluGln_AA168_3Di16/empirical_3di_probabilities.tsv`
- `GluGln_AA168_3Di16/sampled_aa.fasta`
- `GluGln_AA168_3Di16/sampled_3di.fasta`

To make a result ZIP without installing `zip`:

```bash
cd /workspace
python -c "import shutil; shutil.make_archive('glugln_aa168_3di16_results', 'zip', root_dir='runpod_glugln_aa168_3di16_uncertainty/results')"
```

Download `/workspace/glugln_aa168_3di16_results.zip` when complete.

## Validate without loading the model

```bash
cd /workspace/runpod_glugln_aa168_3di16_uncertainty
python run_corrected_glugln.py --validate-only
```
