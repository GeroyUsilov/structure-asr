# Ground-truth score placement

For each of the 37 mapped internal nodes, the IQ-TREE `.state` file is treated
as an independent-site product of marginal posterior distributions. A sampled
ancestral sequence has score

\[
\log P(x)=\sum_{i=1}^{L}\log p_i(x_i).
\]

The analysis draws 5,000 sequences per node, scores each draw under the same
posterior, and scores the corresponding simulated ground-truth sequence. The
dataset-level distribution sums one independent draw for every mapped internal
node and divides by the total number of node-sites. `Root` is excluded because
IQ-TREE did not report a marginal state distribution for it.

IQ-TREE prints probabilities to five decimal places. Sampling uses the printed
probabilities after row normalization. Scoring replaces a printed zero with
`5e-6`, the midpoint of the rounded-to-zero interval. Therefore censored scores
are finite, but the 3Di result remains far below every sampled score.

Files:

- `dataset_score_placement.tsv`: placement of the complete 37-node ancestral set.
- `node_score_placement.tsv`: placement for every mapped internal node.
- `representative_node_samples.tsv`: draws for Node1, Node29, and Node34.
- `dataset_score_histograms.{png,pdf}`: dataset-level histograms.
- `representative_node_histograms_*.{png,pdf}`: representative-node histograms.
- `histogram_data.json`: compact binned data used for interactive plotting.
- `analysis_manifest.json`: settings and reproducibility metadata.
- `analyze_score_placement.py`: reproducible analysis script.
