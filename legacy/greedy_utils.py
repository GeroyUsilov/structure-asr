import numpy as np
import pandas as pd
from tqdm import tqdm
from protein_mcmc_utils import (
    AA_LETTERS, yield_3di, split_3di, log_probability, sample_sequence
)


def batch_split_3di(sequences, tokenizer, model, device):
    """
    Batched version of split_3di for a list of equal-length sequences.
    For sequences > 512 residues, splits each at the midpoint and calls
    yield_3di twice (once per half), then concatenates results.
    """
    if len(sequences[0]) > 512:
        mid = len(sequences[0]) // 2
        structs_1 = yield_3di([s[:mid] for s in sequences], tokenizer, model, device)
        structs_2 = yield_3di([s[mid:] for s in sequences], tokenizer, model, device)
        return [s1 + s2 for s1, s2 in zip(structs_1, structs_2)]
    else:
        return yield_3di(sequences, tokenizer, model, device)


def greedy_site_step(seq, site_idx, df_aa, df_3di, alpha, tokenizer, model, device):
    """
    Evaluate all 20 amino acids at site_idx using half-caching.

    The half of the sequence that does NOT contain site_idx is identical
    across all 20 candidates, so it is evaluated only once (1 yield_3di call).
    The half containing site_idx is batched (1 yield_3di call for all 20).
    Total: 2 yield_3di calls per site regardless of sequence length.

    Returns:
        best_seq    : sequence with best-scoring amino acid at site_idx
        best_struct : corresponding 3Di structure (trimmed to len(df_3di))
        best_score  : corresponding joint log-probability
    """
    mid = len(seq) // 2

    candidates = []
    for aa in AA_LETTERS:
        s = list(seq)
        s[site_idx] = aa
        candidates.append(''.join(s))

    first_halves  = [s[:mid] for s in candidates]
    second_halves = [s[mid:] for s in candidates]

    if site_idx < mid:
        # Mutation in first half: batch first halves, cache second half
        structs_1 = yield_3di(first_halves, tokenizer, model, device)
        shared_2  = yield_3di([second_halves[0]], tokenizer, model, device)[0]
        structs = [s1 + shared_2 for s1 in structs_1]
    else:
        # Mutation in second half: cache first half, batch second halves
        shared_1  = yield_3di([first_halves[0]], tokenizer, model, device)[0]
        structs_2 = yield_3di(second_halves, tokenizer, model, device)
        structs = [shared_1 + s2 for s2 in structs_2]

    structs = [s[:len(df_3di)] for s in structs]
    scores = [
        alpha * log_probability(df_aa, cand)
        + (1 - alpha) * log_probability(df_3di, struct.upper())
        for cand, struct in zip(candidates, structs)
    ]

    best_idx = int(np.argmax(scores))
    return candidates[best_idx], structs[best_idx], scores[best_idx]


def run_greedy_ascent(
    starting_seq, df_aa, df_3di,
    tokenizer, model, device,
    alpha=0.5,
    n_sweeps=10,
    checkpoint_path=None,
    checkpoint_every=10,
):
    """
    Greedy coordinate ascent over amino acid sequence space.

    Each sweep visits all sites ordered by ascending per-site 3Di log-probability
    (worst sites first). At each site, all 20 amino acids are evaluated via
    greedy_site_step; the best is accepted if it strictly improves the objective.
    Stops early if a full sweep produces no improvement.

    Args:
        starting_seq     : initial amino acid sequence string
        df_aa            : DataFrame with per-site AA probabilities (p_A, p_R, ...)
        df_3di           : DataFrame with per-site 3Di probabilities
        tokenizer, model : ProstT5 tokenizer and model
        device           : torch device
        alpha            : weight on AA term (1-alpha on 3Di term)
        n_sweeps         : maximum number of full sweeps over all sites
        checkpoint_path  : CSV path for checkpointing; None = auto-named
        checkpoint_every : save checkpoint every N site steps

    Returns:
        best_seq    : best sequence found across all sweeps
        best_struct : corresponding 3Di structure
        best_log_p  : corresponding joint log-probability
        history     : list of (seq, struct, log_p) tuples at each improvement
    """
    if checkpoint_path is None:
        checkpoint_path = f"greedy_alpha{alpha}.csv"

    current_struct = split_3di(starting_seq, tokenizer, model, device)[:len(df_3di)]
    current_log_p = (
        alpha * log_probability(df_aa, starting_seq)
        + (1 - alpha) * log_probability(df_3di, current_struct.upper())
    )
    current_seq = starting_seq
    history = [(current_seq, current_struct, current_log_p)]

    for sweep in range(n_sweeps):
        improved = False

        # Visit sites in order of ascending 3Di log-prob (worst first)
        site_logps = [
            np.log(float(df_3di.iloc[i][f'p_{current_struct[i].upper()}']))
            for i in range(len(df_3di))
        ]
        site_order = np.argsort(site_logps)

        for step, site_idx in enumerate(tqdm(site_order, desc=f"Sweep {sweep + 1}")):
            cand, cand_struct, cand_score = greedy_site_step(
                current_seq, site_idx, df_aa, df_3di, alpha, tokenizer, model, device
            )

            if cand_score > current_log_p:
                current_seq    = cand
                current_struct = cand_struct
                current_log_p  = cand_score
                improved = True
                history.append((current_seq, current_struct, current_log_p))

            if checkpoint_path and (step + 1) % checkpoint_every == 0:
                pd.DataFrame({
                    'seq':    [h[0] for h in history],
                    'struct': [h[1] for h in history],
                    'log_p':  [h[2] for h in history],
                }).to_csv(checkpoint_path, index=False)

        print(f"Sweep {sweep + 1}: log_p={current_log_p:.2f}, total improvements={len(history) - 1}")
        if not improved:
            print("Converged — no improvement in full sweep.")
            break

    best = max(history, key=lambda h: h[2])
    return best[0], best[1], best[2], history
