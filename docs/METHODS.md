# Methods and conventions

## 1. Forward synthetic evolution

Input: the supplied `AB_eftu_rerooted.tre` (39 tips), with its existing rooting.
The code does not newly reroot it. A single sequence from accession `163840912`
in the supplied alignment seeds the simulation. Removing `-` and `.` from that
sequence gives L=396; this stripping is not deletion evolution or an alignment
procedure applied to descendants. All simulated sequences retain length 396.

For an edge of original length b, the number of update attempts is

\[
N_b=\lfloor\mu Lb\rfloor+
\operatorname{Bernoulli}(\mu Lb-\lfloor\mu Lb\rfloor).
\]

Edges are traversed in preorder with separate NumPy RNG streams spawned from
seed 731024. The tree's sum of branch lengths is about 12.0723694155; it is the
sum across edges, not a root-to-tip distance. Expected total calls are
μLΣb. Branch lengths are not reestimated by the generator.

At each attempt, choose a site i uniformly, mask it in the current sequence x,
and obtain conditional probabilities pθ(a|x\i) from
`facebook/esm2_t33_650M_UR50D`, restricted to 20 canonical amino acids. Draw

\[
x'_i\sim q_\tau(a\mid x_{\setminus i})
=\frac{p_\theta(a\mid x_{\setminus i})^{1/\tau}}
{\sum_c p_\theta(c\mid x_{\setminus i})^{1/\tau}}.
\]

Other sites are unchanged. The current residue remains a possible draw;
self-transitions count as attempts. Every draw is accepted. Thus μ is an
attempt-rate multiplier, not the number of realized substitutions per site.
One masked ESM2 forward pass is made per attempt, logged with timing.

The output saves all 77 node sequences (39 tips + 38 internal including the
degree-two root), branch events, tree and configuration. Branch checkpoints
retain RNG-reproducible completed edges; an interrupted edge is recomputed.

This generator supplies learned sequence-context constraints. It does not
certify folding, structural conservation, natural evolution or an ESMFold
stationary distribution. Learned masked conditionals need not be the
conditionals of a single consistent joint sequence model. OFS/DCA/AlphaFold
interfaces in the generic core are historical alternatives, not this generator.

## 2. AA and structural support

Let p_i(a) and r_i(d) be inferred ancestral AA and 3Di marginals for one mapped
node. They are treated as independent-site products for scoring:

\[
\ell_{AA}(x)=\sum_i\log p_i(x_i),\qquad
\ell_{3Di}(f(x))=\sum_i\log r_i(f(x)_i).
\]

f is the existing full ProstT5 seq2seq `<AA2fold>` encoder via deterministic
greedy generation (`do_sample=False`, `num_beams=1`), not the separate CNN
encoder-only predictor. Its output length equals AA length; sequences longer
than 500 residues are split by the historical balanced-chunk rule. The 396-AA
EF-Tu and 307-site Glu/Gln examples are unchunked. Store uppercase states
internally; lowercase 3Di in logos is display-only.

The 3Di evidence is predicted from AA, not independent experimental structural
data. The score is a weighted product-of-experts objective, not a fully specified
generative posterior coupling AA evolution and structural evolution.

## 3. Current uncertainty-weighted MH sampler

Define Gini uncertainty u_i=1−Σ_a p_i(a)², fixed throughout a run. With mixture
ρ (default .9) and power γ (default 1), sample sites from

\[
w_i=(1-\rho)/L+\rho\,u_i^\gamma/\sum_j u_j^\gamma.
\]

At site i, propose a different residue b from p_i conditioned on b≠x_i=a:

\[
q(x,y)=w_i\frac{p_i(b)}{1-p_i(a)},\quad
q(y,x)=w_i\frac{p_i(a)}{1-p_i(b)}.
\]

Score Sλ(x)=ℓAA(x)+λℓ3Di(f(x)). With chain temperature T (default 2), accept with

\[
\alpha(x,y)=\min\left\{1,
\exp\left[\frac{S_\lambda(y)-S_\lambda(x)}{T}\right]
\frac{q(y,x)}{q(x,y)}\right\}.
\]

This targets π(x)∝exp(Sλ(x)/T) when run as MH with a deterministic f.
Fixed site weights cancel in the Hastings ratio; conditional residue proposal
normalizers do not. Changing λ changes acceptance, not the fixed site weights.
τ is the forward-generator temperature and is distinct from T.

Chains start at AA marginal-MAP by default; alternative AA-posterior starts are
supported. Multiple chains advance in batched inference, not separate GPU
workers. Checkpoints store complete chain and RNG state every 10 iterations.
Best output is the highest-scoring visited state, not a posterior draw or a
ground-truth-selected reconstruction. No convergence claim follows from saving it.

## 4. Historical sampler — different algorithm and parameter scale

The completed representative-node runs used uniform site proposals and AA
posterior residue draws including self-transitions, score

\[
S_{\mathrm{old}}=\alpha\ell_{AA}+(1-\alpha)\ell_{3Di},\quad\alpha=.5,
\]

with T=2, 3000 steps, one chain per node, and `acceptance_mode=legacy`.
That mode omitted the Hastings correction for the asymmetric residue proposal;
it is not MH for the stated exp(Sold/T) target. The optional `hastings` mode
changes this and must not be labeled as a reproduction of those runs.

Equal α=.5 means equal AA/3Di weights, not λ=.5 in the current formula.
After dividing Sold by α, λ=(1−α)/α=1 and temperature becomes T/α=4;
even that rescaling does not remove the legacy acceptance-rule difference.

## 5. Benchmarking and score-histogram conventions

Map internal nodes by the partitions of terminal labels in the three incident
components. Suppress the simulation's degree-two root for this comparison;
37 inferred internal vertices then map uniquely. Recompute mappings separately
for AA and 3Di trees; names alone do not establish correspondence.

AA fidelity is sitewise identity / Hamming distance to simulated AA truth.
3Di fidelity uses predicted encodings of those true ancestral AAs as reference.
Direct 3Di marginal-MAP is different from f(AA-MAP).

Probability conventions retained from the original scripts:

- Sampling: floor each posterior entry at 5×10⁻⁶, then normalize the row.
- Dataset truth scores: use printed probabilities; clamp values below 5×10⁻⁶
  only for log scoring; count printed zeros separately.
- Score histograms: normalize printed rows, sample independent site states,
  and apply the same log floor to sampled and true sequences.

These are related but numerically distinct calculations. Rounded zeros are
censored probabilities, not known exact zeros. Histograms compare truth with
the inferred product distribution; samples are not ESM2-generated proteins.
Dataset-wide histograms aggregate log scores over mapped nodes and sites;
representative-node histograms describe a single node. Neither plot diagnoses
why the structural posterior disagrees with truth.

Aligned Hamming comparisons treat every aligned column equally (including gap
states in the empirical data). Empirical QE1.9d is a Glu/Gln comparison dataset,
not an independently sampled EF-Tu family; differing alignment lengths and
biological histories limit the interpretation of its distance comparison.

## 6. Batched forward execution (`asr-grid forward`)

The kernel of §1 is unchanged. Each preorder edge already has its own RNG stream
spawned from the cell seed. An edge's trajectory therefore depends only on its
parent sequence, that stream and the masked-site profiles, so edges whose
parents are complete can advance concurrently. This holds across edges of one
tree and across cells. Every step draws each active lane's site with its own
stream, makes one batched ESM2 pass (one masked position per row, equal lengths,
no padding), and then makes the same tempering and state draw as the serial
kernel. With a deterministic predictor the output, including node order and
event order, is identical to `ForwardSimulator` for any batch size or shard
split (tested). On a GPU, batched fp16 reductions can differ from batch-1 calls,
so a draw near a probability boundary can change. Grid outputs are labeled
`batched multi-lane` and are not reproductions of the historical batch-1 runs;
use `asr-evolve` for those. Batch size and scheduling order are recorded but are
not part of the checkpoint identity. An interrupted edge restarts from its own
fresh stream.

## 7. Fixed-topology IQ-TREE ASR (`asr-grid asr`)

Inputs are the tip alignment and the known topology, written without internal
labels, supports or branch lengths. IQ-TREE (`-te`) keeps the topology and
re-estimates branch lengths and model parameters. Simulated branch lengths count
update attempts, not substitutions, so they are not supplied. `--ancestral`
writes empirical-Bayes marginals.

- AA: `-st AA -m MFP` (ModelFinder, BIC).
- 3Di: states encoded as pseudo-amino-acids, `-st AA -m MFP -mdef <Q.3Di.AF> -mset Q3Di_AF`.
  The exchangeabilities are fixed to the published Q.3Di.AF matrix (Garg &
  Hochberg 2025, MBE 42:msaf124; CC0, doi:10.17617/3.1MJJBH). ModelFinder
  chooses only the frequency and rate-heterogeneity components.
- `--seed` is the cell seed. The command, IQ-TREE version, selected model and
  input hashes are written to `<modality>.asr_manifest.json`.

Using the true topology is a known-tree assumption that favors reconstruction
relative to practice. The ESM2 generator is not a Markov substitution process,
so every candidate model is misspecified; BIC picks the best available one.
