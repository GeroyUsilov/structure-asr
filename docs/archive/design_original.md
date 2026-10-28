# Experimental design: folded forward evolution with ASR ground truth

## Primary estimand

For every internal node, compare an ASR method's inferred sequence or posterior
ensemble with the exact sequence generated at that node. The primary metric is
normalized Hamming distance. Secondary metrics should measure posterior
calibration, top-k/ensemble coverage, structural agreement with the node's
predicted structure, and score gaps under evaluators not used by the generator.

Do not reduce the benchmark to “does a reconstructed sequence fold?” A wrong
ancestor can fold very confidently. Ground-truth sequence recovery and fold
preservation answer different questions and both must be reported.

## Generative regimes

Use the same rooted topology and root sequence across several generators:

| Regime | Proposal / acceptance | Purpose |
|---|---|---|
| Independent-site control | empirical site profile | exposes the easy, non-epistatic baseline |
| DCA Gibbs | exact Potts conditional, always accept | reproduces the proposed DCA evolver |
| LM | LM/OFS profile proposal, pseudo-perplexity MH energy | tests learned protein grammar |
| Fold-strict | uniform or DCA proposal, AF energy every attempt | clearest but most expensive fold constraint |
| Hybrid | DCA + LM + AF/reference-TM composite energy | strongest practical realism hypothesis |

For hybrid Metropolis-Hastings, component energies must be scaled before
weighting. Estimate each component's typical single-mutant delta on a pilot set
and use robust scales (for example median absolute delta). Otherwise a numeric
unit such as raw DCA energy can silently dominate pTM or pseudo-perplexity.

The AlphaFold term should include similarity to the ancestral reference fold
(TM-score or an equivalent topology measure), not only pLDDT/pTM/PAE. High
confidence alone does not imply preservation of the same fold.

## DCA regime from the paper

For each branch of length `t`, make `round(mu_gen * L * t)` site updates. At an
update, select a site uniformly and sample its new state from

`P(a_i | s_-i) proportional to exp(h_i(a_i) + sum_j J_ij(a_i, s_j))`.

This is a Gibbs update and is accepted directly. The DCA parameters should be
fit once to a curated natural-family MSA. Record the exact MSA, weighting,
regularization, gauge, software version, and conversion hash in the run
manifest. Select roots across context-dependent entropy (CDE) quantiles because
root mutability changes the recoverability regime.

## LM/OFS regime

For an encoder model, define per-residue pseudo-log-likelihood

`PLL(s) = mean_i log P(s_i | s_-i)`

and pseudo-perplexity `PP(s) = exp(-PLL(s))`; lower is more natural. OFS
approximates the full masked profile in one encoder pass. Its profile is useful
as an informed mutation proposal, while `-PLL` is an energy. If an asymmetric
LM proposal is used with Metropolis-Hastings, both forward and reverse proposal
probabilities must be included. The current core implements that logic for DCA;
an LM proposal adapter should follow the same `Proposal` contract.

Long LM-only chains need explicit guards for repeated-motif degeneration and
phylogenetic drift, both cautioned about in the OFS paper. Audit low-complexity,
repeat, composition, and clade-bias metrics at every saved node.

Two OFS regimes must be named separately. `ofs_pseudogibbs` mirrors the DCA
algorithm: uniform site choice, conditional OFS amino-acid draw, and automatic
acceptance. `metropolis` uses the same OFS proposal but accepts against an
explicit pseudoperplexity or composite energy with the Hastings correction.
The former is the cleanest like-for-like procedural comparison; the latter has
a declared sequence-level target energy.

## AlphaFold regime and compute strategy

Run two tiers:

1. a cheap tier (DCA and/or LM) for every proposal;
2. a structure tier (AlphaFold/ColabFold or ESMFold pilot) for all saved nodes
   and a predeclared fraction of proposals.

For a scientifically exact “fold-strict” generator, score structure on every
proposal. Screening AlphaFold only periodically creates a different,
path-dependent process and must be labeled as such. A correct delayed-acceptance
MH implementation is preferable if a cheap surrogate is used to avoid most
AlphaFold calls.

Store mean and per-residue pLDDT, PAE summaries, pTM/ipTM where applicable,
reference-fold TM-score, model/version, MSA/template mode, random seeds, and raw
prediction paths. Keep prediction settings constant across a comparison.

## Benchmark grid

At minimum cross:

- 5 root CDE quantiles;
- 4 divergence levels spanning shallow, recoverable, difficult, and saturated;
- 3 selection temperatures;
- at least 20 stochastic replicates per cell;
- several tree sizes/topologies, not only branch-length rescaling of one tree;
- all five generator regimes above where compute allows.

Hold the leaf count and topology fixed within paired comparisons. Calibrate
`mu_gen` by observed root-to-leaf Hamming distance, rather than assuming the
same numeric rate is comparable across DCA, LM, and AF selection landscapes.

## ASR methods to compare

- site-independent maximum a posteriori reconstruction;
- independent posterior samples;
- mixed/rate-heterogeneous standard ASR (for a strong conventional baseline);
- the DCA reshuffling method from the paper;
- LM-aware candidate reranking or sampling;
- structure-aware candidate reranking;
- hybrid reranking using evaluators that were not all used by the generator.

Reconstructors receive only `tree.nwk` and `leaves.fasta`. Any natural-family
MSA supplied to a method must be declared, frozen, and identical across paired
methods where possible.

## Metrics and leakage controls

Primary: node-wise normalized Hamming distance, summarized by node depth and
root-to-leaf divergence with replicate-level uncertainty.

Secondary:

- true-residue posterior probability, Brier score, log loss, and calibration;
- minimum and expected distance over posterior samples;
- reference-fold TM-score and RMSD;
- pLDDT, PAE, pTM, DCA energy, and LM pseudo-perplexity;
- diversity/entropy of reconstructed ensembles;
- functional-site recovery and contact-pair recovery where annotations exist.

Report metrics both under generator-aligned evaluators and orthogonal
evaluators. A DCA reconstructor on DCA-generated truth is an important positive
control, not sufficient evidence of general superiority. The decisive result is
the generator-by-reconstructor interaction across held-out scoring systems.
