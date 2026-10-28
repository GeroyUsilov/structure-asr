# Representative-node joint AA/3Di ASR MCMC

This job runs the established sequence-space procedure on three ancestral
nodes from one synthetic dataset. At each step it proposes amino acids from
the node's IQ-TREE AA posterior, predicts the candidate's 3Di sequence with
deterministic ProstT5 `<AA2fold>`, and uses

```text
0.5 * log P_AA(sequence) + 0.5 * log P_3Di(ProstT5(sequence))
```

as the default joint score. Defaults reproduce the working
`protein_mcmc_utils.run_forward_chain` settings used in the project:

```text
MCMC iterations:  3000
MCMC temperature: 2.0
mutations/step:   1
alpha:            0.5
start:            AA MAP sequence
acceptance:       legacy project rule
seed:             731024
```

The synthetic-generation proposal temperature in the job name is distinct
from the MCMC temperature above.

The three chains are advanced together, so each MCMC iteration folds one
batch of three candidates. Checkpoints and output FASTAs are refreshed every
10 iterations. Rerunning the launcher resumes the job.

The primary outputs are `joint_asr_best_aa.fasta` and
`joint_asr_best_3di.fasta`. Final chain states, complete proposal traces,
component scores, and a manifest are also retained.

The optional environment variable `ACCEPTANCE_MODE=hastings` adds the formal
Hastings correction for the non-symmetric AA-posterior proposal. The default
`legacy` value matches the existing project implementation exactly.
