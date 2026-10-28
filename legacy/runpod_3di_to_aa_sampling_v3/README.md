# ProstT5 reverse uncertainty: fixed 3Di → sampled amino acids

This RunPod bundle performs the reverse of the earlier AA-posterior propagation
experiment for all three 3Di reconstructions:

| Analysis label | 3Di posterior input | Sites |
|---|---|---:|
| GluGln_3Di16 | `inputs/3Di_Node16.csv` | 307 |
| Node11 | `inputs/3Di_Node11.csv` | 661 |
| Node245 | `inputs/3Di_Node245.csv` | 661 |

For each node, the script takes the maximum-probability 3Di state at every site,
forming one fixed maximum-likelihood 3Di sequence. It then uses ProstT5's
`<fold2AA>` direction to generate 2,000 stochastic amino-acid translations and
calculates empirical amino-acid frequencies at every site.

## Is ProstT5 deterministic?

The model is deterministic under greedy decoding (`do_sample=False`, one beam),
apart from possible low-level hardware numerical differences. Running that mode
2,000 times would return the same AA sequence and produce only 0/1 site
frequencies. This workflow therefore uses stochastic decoding (`do_sample=True`).

Defaults follow the ProstT5 model-card inverse-folding example:

- temperature: 1.1
- top-p: 0.90
- top-k: 6
- repetition penalty: 1.2

The resulting sitewise values are Monte Carlo frequencies under those decoding
settings. They are not the raw one-step decoder softmax probabilities, and they
are not a phylogenetic amino-acid posterior.

ProstT5 occasionally emits a lowercase 3Di token even in `<fold2AA>` mode. This
workflow uses sequence-level rejection sampling: any generated sequence (or
long-sequence chunk) containing a non-AA token is rejected and retried. This
keeps the previously completed, valid uppercase-AA samples statistically
compatible with the continuation run.

## Run on RunPod

Upload `runpod_3di_to_aa_sampling_v3.zip` to `/workspace`. No `unzip` executable is
needed:

```bash
cd /workspace
python -m zipfile -e runpod_3di_to_aa_sampling_v3.zip .
cd runpod_3di_to_aa_sampling_v3
bash run_on_runpod.sh
```

The first run installs the small dependency set and downloads/caches
`Rostlab/ProstT5`. It reuses `/workspace/.cache/huggingface`, imports completed
samples from the original
`/workspace/runpod_3di_to_aa_sampling/results/` checkpoints, and writes the
combined continuation to `results_rejection_resume/` inside the v3 bundle. The
original results directory is not modified.

If GPU memory is insufficient, reduce the batch size:

```bash
BATCH_SIZE=4 bash run_on_runpod.sh
```

To override sampling settings:

```bash
N_SAMPLES=2000 TEMPERATURE=1.1 TOP_P=0.90 TOP_K=6 \
REPETITION_PENALTY=1.2 BATCH_SIZE=8 bash run_on_runpod.sh
```

Progress is printed after each completed batch, for example:

```text
Node11: accepted 640/2000 AA sequences (1 rejected in latest batch)
```

Rerun the same command after interruption. `--resume` is enabled by the wrapper,
so completed samples are read from `sampling_checkpoint.jsonl`. On its first
run, v3 automatically imports each valid unversioned checkpoint from the
original bundle. Subsequent runs resume from the v3 checkpoint.

## Validate inputs without loading ProstT5

```bash
python run_3di_to_aa_sampling.py --input-dir inputs --validate-only
```

## Outputs

Each node directory under `results_rejection_resume/` contains:

- `maximum_likelihood_3di.fasta`: the fixed row-wise argmax 3Di input.
- `sampled_aa.fasta`: all 2,000 ProstT5 AA translations.
- `sampling_checkpoint.jsonl`: resumable sample records.
- `empirical_aa_probabilities.tsv`: sitewise empirical probabilities, counts,
  entropy, top AA, and the corresponding maximum-likelihood 3Di state.

Top-level outputs:

- `all_nodes_empirical_aa_probabilities.tsv`
- `node_summary.tsv`
- `run_manifest.json`

For 2,000 samples, the smallest nonzero empirical probability is 0.0005.

## Long sequences

Node11 and Node245 exceed the model context limit. They are divided into two
balanced chunks (331 and 330 sites by default), translated separately within
each Monte Carlo sample, and concatenated back to the original 661-site order.
The per-site probabilities remain aligned to the original 3Di tables, but the
two chunks do not condition on one another across their boundary.
