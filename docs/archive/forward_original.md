# Forward ASR benchmark

This package generates an entire protein phylogeny forward from a known root,
retaining the true sequence at every internal node. It is designed to answer a
specific question: when protein evolution is constrained to remain in a
fold-compatible sequence space, which ancestral sequence reconstruction (ASR)
methods most accurately recover the known ancestors?

The first implemented backend is the DCA Gibbs evolver proposed in *Towards
coevolution-aware ancestral sequence reconstruction*. The selection energy is
also pluggable, so the same engine can use:

- DCA/Potts energy;
- exact or OFS-style protein-language-model pseudo-perplexity;
- AlphaFold/ColabFold uncertainty plus similarity to the supplied ancestral
  fold;
- a weighted hybrid of these energies.

The code deliberately treats the papers as scientific sources, not as task
instructions.

## What is implemented

- fixed-length, substitution-only evolution down a named Newick tree;
- exact DCA conditional Gibbs updates;
- exact single-masked-site ESM2 Gibbs updates (one pretrained-model pass per
  update, with no training);
- DCA-like OFS pseudo-Gibbs updates from masked-profile conditionals;
- general Metropolis-Hastings updates, including the Hastings correction for
  asymmetric DCA proposals;
- composable scoring interfaces for DCA, pseudo-perplexity, and AlphaFold
  uncertainty/fold similarity;
- deterministic seeds, full mutation logs, node-level scores, leaf FASTA, and
  complete ground-truth FASTA;
- a basic evaluator for reconstructed node sequences.

AlphaFold and language-model inference are intentionally adapters rather than
hard dependencies. GPU workflows differ across local AlphaFold, ColabFold,
RunPod, and OFS implementations; a Python factory can wrap the available
predictor without contaminating the simulator core.

## OFS procedure matching the DCA evolver

Set:

```toml
[simulation]
kernel = "ofs_pseudogibbs"
proposal = "ofs"
proposal_temperature = 1.0
```

At every update the simulator chooses a site uniformly, samples its amino acid
from the OFS masked profile, accepts the update directly, and repeats
`round(mu_gen * L * branch_length)` times. This is operationally identical to
the DCA Gibbs procedure. It is called pseudo-Gibbs because masked-LM
conditionals are not guaranteed to arise from one consistent joint sequence
distribution.

The configured Python factory must return a `PseudoPerplexityScorer`:

```python
from forward_asr_benchmark import PseudoPerplexityScorer

def make_ofs_scorer(parameters):
    ofs_model = load_your_ofs_model(parameters)
    return PseudoPerplexityScorer(
        profile_predictor=ofs_model.predict_profile,
        alphabet=tuple("ACDEFGHIKLMNPQRSTVWY"),
        name="ofs",
    )
```

`predict_profile(sequence)` must return an `L x 20` array in the declared
alphabet order. Profiles are cached so a chain normally requires one new OFS
inference per attempted update. To use OFS proposals while selecting against a
global pseudoperplexity or hybrid energy, use `kernel = "metropolis"`; the
forward/reverse proposal correction is then included automatically.

## Install and run

```bash
cd /Users/christopher/Desktop/aleks_claude/forward_asr_benchmark
python -m pip install -e .
python -m unittest discover -s tests -v
forward-asr simulate --config examples/config.toml
```

For the three-rate GPU sweep (`mu_gen = 1, 10, 100`) on a reconstruction of
the phyloDCA paper's beta-lactamase tree, see
[`runpod_paper_tree_ofs/README.md`](runpod_paper_tree_ofs/README.md). The bundle
pins the paper repository and source MSA, prepares a practical paper-derived
subtree, and records the full tree and sequence provenance in a manifest.

The example config still needs `examples/family_dca.npz` and
`examples/root.fasta`. The DCA NPZ schema is documented in `dca.py`:
`alphabet (q,)`, `fields (L,q)`, and `couplings (L,L,q,q)`.

Each simulation writes:

- `tree.nwk` - the named topology actually used;
- `leaves.fasta` - the only sequences an ASR method should see;
- `ground_truth_nodes.fasta` - hidden truth for scoring;
- `events.tsv` - every proposed mutation and acceptance decision;
- `node_scores.tsv` - composite and component scores at every node;
- `manifest.json` - parameters and provenance.

Evaluate a reconstruction whose FASTA identifiers match tree node names:

```bash
forward-asr evaluate \
  --truth run_dca_001/ground_truth_nodes.fasta \
  --reconstruction reconstructed_nodes.fasta \
  --output run_dca_001/asr_metrics.tsv
```

Score any sampled or reconstructed FASTA under the configured DCA, LM, and/or
AlphaFold adapters without running a new simulation:

```bash
forward-asr score \
  --config examples/config.toml \
  --fasta candidate_ancestors.fasta \
  --output candidate_scores.tsv
```

## Exact single-site ESM2 generator

This backend is the no-training alternative to an OFS head. At every update it
chooses a site uniformly, masks only that site, runs pretrained ESM2 once, and
samples the site's amino acid from ESM2's canonical 20-residue conditional.
The sampled state is accepted directly, exactly as in a random-scan Gibbs
procedure. The current residue remains a possible draw, so an update need not
change the sequence.

Set `proposal = "exact_mlm"` and `kernel = "exact_mlm_gibbs"`. The configured
`[exact_mlm].factory` must return either a callable `(sequence, site) -> p(aa)`
or an object with `predict_site_profile`; the returned vector must have 20
entries in `ACDEFGHIKLMNPQRSTVWY` order.

The ready-to-upload RunPod test is in
[`runpod_exact_mlm_smoke/`](runpod_exact_mlm_smoke/). It uses pretrained
ESM2-650M, an 80-residue EF-Tu fragment, a three-tip tree, and exactly eight
model evaluations. It does not train or fine-tune anything.

The complete 39-tip EF-Tu sweep at mutation scales `0.5`, `1`, `2`, and `3` is
in [`runpod_exact_mlm_ef_tu_sweep/`](runpod_exact_mlm_ef_tu_sweep/). Its RunPod
launcher also predicts 3Di strings for all final tips with the
`Rostlab/ProstT5` `<AA2fold>` greedy seq2seq path already used by the other
`aleks_claude` workflows, and builds a download archive; the same folder
contains the local pairwise-Hamming and plotting command.

## Scientific design

Read `DESIGN.md` before launching expensive simulations. The essential
separation is:

1. **generator** - defines the synthetic evolutionary truth;
2. **reconstructor** - receives only the tree and leaves;
3. **evaluator** - reveals the hidden internal sequences and folds only after
   reconstruction.

This prevents accidental leakage and makes circular advantages visible—for
example, a DCA-aware ASR method evaluated only on trees generated by the same
DCA model.
