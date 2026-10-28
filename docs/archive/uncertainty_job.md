# Node34 tau=2 uncertainty-weighted AA/3Di sweep

This focused RunPod job tests weak structural guidance at `Node34` in the
`mu1_tau2` synthetic dataset. It is a blinded ASR job: the bundle contains no
ground-truth sequences and performs no ground-truth comparison.

The AA marginal-MAP sequence is the starting point. At every MCMC step, a site
is selected with fixed probability

```text
w_i = 0.1/L + 0.9*u_i/sum(u),  u_i = 1 - sum_a p_i(a)^2
```

and a residue different from the current residue is drawn from the site's AA
posterior. This removes no-op proposals while concentrating work on ambiguous
AA sites. The asymmetric residue proposal uses the formal Hastings correction.

The score is

```text
log P_AA(sequence) + lambda * log P_3Di(ProstT5(sequence))
```

The default sweep is:

```text
lambda:            0, 0.005, 0.01, 0.02, 0.05
chains per lambda: 3
steps per chain:   3000
MCMC temperature: 2.0
site focus:        90% uncertainty-weighted + 10% uniform
start:             AA marginal MAP
```

Here `tau=2` denotes the temperature used to generate the synthetic dataset;
it is distinct from the MCMC temperature.

Run from `/workspace`:

```bash
tar -xzf runpod_uncertainty_weighted_node34_tau2.tar.gz
cd runpod_uncertainty_weighted_node34_tau2
bash run_on_runpod.sh
```

The launcher resumes completed or interrupted lambda runs and creates:

```text
/workspace/runpod_uncertainty_weighted_node34_tau2_results.tar.gz
```

The results contain one `lambda_*` directory per weight. Each directory holds
checkpoints, complete traces, AA-MAP starting FASTAs, best and final FASTAs,
component-score summaries, and proposal-site weights. Ground-truth evaluation
is deliberately deferred until all ASR runs and model-based selection are
complete.

Optional environment overrides include `N_ITER`, `CHAINS_PER_NODE`, `LAMBDAS`,
`MCMC_TEMPERATURE`, `SITE_UNCERTAINTY_MIXTURE`, and `SITE_WEIGHT_POWER`.
