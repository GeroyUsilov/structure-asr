import re
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

AA_LETTERS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def yield_3di(sequence_examples, tokenizer, model, device):
    min_len = min([len(s) for s in sequence_examples]) + 1
    max_len = max([len(s) for s in sequence_examples]) + 1

    prep_start = time.time()

    # replace all rare/ambiguous amino acids by X and introduce white-space
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    # add pre-fixes for AA to 3Di translation
    sequence_examples = ["<AA2fold>" + " " + s for s in sequence_examples]

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer(
        sequence_examples,
        add_special_tokens=True,
        padding="longest",
        return_tensors='pt'
    ).to(device)

    gen_kwargs_aa2fold = {
        "do_sample": False,
        "num_beams": 1,
    }

    # translate from AA to 3Di
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                translations = model.generate(
                    ids.input_ids,
                    attention_mask=ids.attention_mask,
                    max_length=max_len,
                    min_length=min_len,
                    early_stopping=True,
                    num_return_sequences=1,
                    **gen_kwargs_aa2fold
                )
        else:
            translations = model.generate(
                ids.input_ids,
                attention_mask=ids.attention_mask,
                max_length=max_len,
                min_length=min_len,
                early_stopping=True,
                num_return_sequences=1,
                **gen_kwargs_aa2fold
            )

    decoded_translations = tokenizer.batch_decode(translations, skip_special_tokens=True)
    structure_sequences = ["".join(ts.split(" ")) for ts in decoded_translations]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return structure_sequences


def yield_aa(structure_examples, tokenizer, model, device):
    min_len = min(len(s) for s in structure_examples) + 1
    max_len = max(len(s) for s in structure_examples) + 1

    # introduce white-space between characters
    structure_examples = [" ".join(list(structure.lower())) for structure in structure_examples]

    # add pre-fixes for 3Di to AA translation
    structure_examples = ["<fold2AA>" + " " + s for s in structure_examples]

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer(
        structure_examples,
        add_special_tokens=True,
        padding="longest",
        return_tensors='pt'
    ).to(device)

    gen_kwargs_fold2aa = {
        "do_sample": False,
        "num_beams": 1,
    }

    # translate from 3Di to AA
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                translations = model.generate(
                    ids.input_ids,
                    attention_mask=ids.attention_mask,
                    max_length=max_len,
                    min_length=min_len,
                    early_stopping=True,
                    num_return_sequences=1,
                    **gen_kwargs_fold2aa
                )
        else:
            translations = model.generate(
                ids.input_ids,
                attention_mask=ids.attention_mask,
                max_length=max_len,
                min_length=min_len,
                early_stopping=True,
                num_return_sequences=1,
                **gen_kwargs_fold2aa
            )

    decoded_translations = tokenizer.batch_decode(translations, skip_special_tokens=True)
    amino_acid_sequences = ["".join(ts.split(" ")) for ts in decoded_translations]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return amino_acid_sequences


def stitch_3di(sequence, tokenizer, model, device):
    if len(sequence) > 512:
        part_1_input = sequence[:512]
        part_2_input = sequence[-512:]
        structure_sequences = yield_3di([part_1_input, part_2_input], tokenizer, model, device)
        part_1 = structure_sequences[0][:int(len(sequence) / 2)]
        part_2 = structure_sequences[1][-len(sequence) + int(len(sequence) / 2):]
        return part_1 + part_2
    else:
        structure_sequences = yield_3di([sequence], tokenizer, model, device)
        return structure_sequences[0]


def split_3di(sequence, tokenizer, model, device):
    if len(sequence) > 512:
        part_1_input = sequence[:int(len(sequence) / 2)]
        part_2_input = sequence[int(len(sequence) / 2):]
        structure_sequences = yield_3di([part_1_input, part_2_input], tokenizer, model, device)
        return structure_sequences[0] + structure_sequences[1]
    else:
        structure_sequences = yield_3di([sequence], tokenizer, model, device)
        return structure_sequences[0]


def split_aa(sequence, tokenizer, model, device):
    if len(sequence) > 512:
        mid = len(sequence) // 2
        part_1_input = sequence[:mid]
        part_2_input = sequence[mid:]
        structure_sequences = yield_aa([part_1_input, part_2_input], tokenizer, model, device)
        return structure_sequences[0] + structure_sequences[1]
    else:
        structure_sequences = yield_aa([sequence], tokenizer, model, device)
        return structure_sequences[0]


def sample_sequence(df):
    """Sample a random sequence from the site-wise distributions."""
    sequence = []
    for _, row in df.iterrows():
        probs = row[[f'p_{aa}' for aa in AA_LETTERS]].values.astype(float)
        probs = probs / probs.sum()
        aa_idx = np.random.choice(len(AA_LETTERS), p=probs)
        sequence.append(AA_LETTERS[aa_idx])
    return ''.join(sequence)


def top_k_aa(df, site_idx, k=3):
    """Return the k highest-probability amino acids at a site as (aa, prob) pairs."""
    probs = df.iloc[site_idx][[f'p_{aa}' for aa in AA_LETTERS]].values.astype(float)
    probs = probs / probs.sum()
    order = np.argsort(probs)[::-1][:k]
    return [(AA_LETTERS[j], float(probs[j])) for j in order]


def log_probability(df, sequence):
    """Calculate log probability of a given sequence."""
    log_prob = 0.0
    for i, aa in enumerate(sequence):
        prob = float(df.iloc[i][f'p_{aa}'])
        log_prob += np.log(prob)
    return log_prob


def resample_site(df, sequence, site_idx):
    """Resample amino acid at specific site and return new sequence."""
    sequence_list = list(sequence)
    probs = df.iloc[site_idx][[f'p_{aa}' for aa in AA_LETTERS]].values.astype(float)
    probs = probs / probs.sum()
    aa_idx = np.random.choice(len(AA_LETTERS), p=probs)
    sequence_list[site_idx] = AA_LETTERS[aa_idx]
    return ''.join(sequence_list)


def hamming_distance(s1, s2):
    """Calculate Hamming distance between two strings of equal length."""
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def hamming_binary_list(s1, s2):
    """Return binary list: 1 for matches, 0 for mismatches."""
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length")
    return [1 if c1 == c2 else 0 for c1, c2 in zip(s1, s2)]


def sample_and_score(
    df_aa, df_3di,
    tokenizer, model, device,
    n_samples=1000,
    alpha=0.5,
    checkpoint_path=None,
    checkpoint_every=10,
):
    """
    Sample independent sequences from the site-wise AA distribution and score
    each under both the AA and 3Di models.

    Unlike run_forward_chain, this performs no MCMC: every sequence is an
    independent i.i.d. draw from df_aa (via sample_sequence). Each draw is
    folded to a 3Di structure with split_3di, then scored with the AA
    log-probability, the 3Di log-probability, and the alpha-weighted joint
    log-probability (alpha * AA + (1 - alpha) * 3Di), matching the objective
    used by run_forward_chain.

    Returns:
        seqs        : list of sampled AA sequences
        structs     : list of corresponding 3Di structures
        log_p_aa    : list of AA log-probabilities
        log_p_3di   : list of 3Di log-probabilities
        log_p_joint : list of alpha-weighted joint log-probabilities
    """
    if checkpoint_path is None:
        checkpoint_path = f"samples_alpha{alpha}_n{n_samples}.csv"

    seqs = []
    structs = []
    log_p_aa = []
    log_p_3di = []
    log_p_joint = []

    for i in tqdm(range(n_samples), desc="Sampling"):
        # Independent draw from the site-wise AA distribution
        seq = sample_sequence(df_aa)

        # Fold to 3Di structure (per-sequence, same convention as the MCMC chain)
        struct = split_3di(seq, tokenizer, model, device)
        struct = struct[:len(df_3di)]

        lp_aa = log_probability(df_aa, seq)
        lp_3di = log_probability(df_3di, struct.upper())
        lp_joint = alpha * lp_aa + (1 - alpha) * lp_3di

        seqs.append(seq)
        structs.append(struct)
        log_p_aa.append(lp_aa)
        log_p_3di.append(lp_3di)
        log_p_joint.append(lp_joint)

        if checkpoint_path and ((i + 1) % checkpoint_every == 0):
            pd.DataFrame({
                'seq': seqs,
                'struct': structs,
                'log_p_aa': log_p_aa,
                'log_p_3di': log_p_3di,
                'log_p_joint': log_p_joint,
            }).to_csv(checkpoint_path, index=False)

    return seqs, structs, log_p_aa, log_p_3di, log_p_joint


def mutational_scan(
    starting_seq, df_aa, df_3di,
    tokenizer, model, device,
    top_k=3,
    alpha=0.5,
    include_wildtype=False,
    checkpoint_path=None,
    checkpoint_every=50,
):
    """
    Single-site mutational scan of a starting sequence.

    Walks each position i in order and substitutes the top_k most-probable
    amino acids at that site (according to df_aa), one at a time, holding all
    other positions fixed at the starting sequence. Each single mutant is
    folded to 3Di with split_3di and scored under both the AA and 3Di models.

    Scores are reported as deltas relative to the starting sequence, so a
    positive d_log_p_* means the mutation improves that objective.

    Args:
        top_k            : number of top AAs per site to try (by df_aa prob).
        alpha            : AA / 3Di tradeoff for the joint log-p.
        include_wildtype : if False, skip candidates equal to the starting
                           residue at that position (only true mutations).

    Returns:
        pandas.DataFrame with one row per (position, candidate AA):
            position, wt_aa, mut_aa, aa_rank, p_aa_site,
            log_p_aa, log_p_3di, log_p_joint,
            d_log_p_aa, d_log_p_3di, d_log_p_joint,
            seq, struct
        sorted by descending d_log_p_joint.
    """
    if checkpoint_path is None:
        checkpoint_path = f"mutscan_top{top_k}_alpha{alpha}.csv"

    # Baseline scores for the starting sequence
    base_struct = split_3di(starting_seq, tokenizer, model, device)[:len(df_3di)]
    base_aa = log_probability(df_aa, starting_seq)
    base_3di = log_probability(df_3di, base_struct.upper())
    base_joint = alpha * base_aa + (1 - alpha) * base_3di

    rows = []
    for i in tqdm(range(len(starting_seq)), desc="Mut scan"):
        wt_aa = starting_seq[i]
        for rank, (mut_aa, p_site) in enumerate(top_k_aa(df_aa, i, k=top_k)):
            if mut_aa == wt_aa and not include_wildtype:
                continue

            mut_seq = starting_seq[:i] + mut_aa + starting_seq[i + 1:]
            mut_struct = split_3di(mut_seq, tokenizer, model, device)[:len(df_3di)]

            lp_aa = log_probability(df_aa, mut_seq)
            lp_3di = log_probability(df_3di, mut_struct.upper())
            lp_joint = alpha * lp_aa + (1 - alpha) * lp_3di

            rows.append({
                'position': i,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa,
                'aa_rank': rank,
                'p_aa_site': p_site,
                'log_p_aa': lp_aa,
                'log_p_3di': lp_3di,
                'log_p_joint': lp_joint,
                'd_log_p_aa': lp_aa - base_aa,
                'd_log_p_3di': lp_3di - base_3di,
                'd_log_p_joint': lp_joint - base_joint,
                'seq': mut_seq,
                'struct': mut_struct,
            })

            if checkpoint_path and (len(rows) % checkpoint_every == 0):
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    df_scan = pd.DataFrame(rows).sort_values('d_log_p_joint', ascending=False).reset_index(drop=True)
    if checkpoint_path:
        df_scan.to_csv(checkpoint_path, index=False)
    return df_scan


def combine_top_mutations(starting_seq, df_scan, n, score_col='d_log_p_joint'):
    """
    Apply the n best distinct-position mutations from a scan to a sequence.

    df_scan rows are ranked by score_col (descending). Only the highest-ranked
    mutation at each position is kept, so n counts distinct positions (rows that
    hit an already-used position are skipped). Returns (combined_seq, n_applied).
    """
    df_sorted = df_scan.sort_values(score_col, ascending=False)
    seq = list(starting_seq)
    used = set()
    applied = 0
    for _, row in df_sorted.iterrows():
        if applied >= n:          # check BEFORE applying, so n=0 applies nothing
            break
        pos = int(row['position'])
        if pos in used:
            continue
        seq[pos] = row['mut_aa']
        used.add(pos)
        applied += 1
    return ''.join(seq), applied


def evaluate_mutation_stacks(
    starting_seq, df_scan, df_aa, df_3di,
    tokenizer, model, device,
    n_list=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    alpha=0.5,
    score_col='d_log_p_joint',
    checkpoint_path=None,
):
    """
    Stack the top-ranked scan mutations and evaluate the combined sequence.

    For each n in n_list, apply the n best distinct-position mutations from
    df_scan (via combine_top_mutations) to starting_seq, fold the result to 3Di
    with split_3di, and score it under both models. n=0 (the starting sequence)
    is always included as a baseline.

    Returns a DataFrame with one row per n:
        n_requested, n_applied, log_p_aa, log_p_3di, log_p_joint,
        d_log_p_aa, d_log_p_3di, d_log_p_joint, seq, struct
    (deltas are relative to the starting sequence).
    """
    if checkpoint_path is None:
        checkpoint_path = f"mutstack_alpha{alpha}.csv"

    n_values = [0] + [int(n) for n in n_list]

    rows = []
    base_aa = base_3di = base_joint = None
    for n in tqdm(n_values, desc="Mut stacks"):
        combined_seq, applied = combine_top_mutations(starting_seq, df_scan, n, score_col=score_col)
        struct = split_3di(combined_seq, tokenizer, model, device)[:len(df_3di)]

        lp_aa = log_probability(df_aa, combined_seq)
        lp_3di = log_probability(df_3di, struct.upper())
        lp_joint = alpha * lp_aa + (1 - alpha) * lp_3di

        if n == 0:
            base_aa, base_3di, base_joint = lp_aa, lp_3di, lp_joint

        rows.append({
            'n_requested': n,
            'n_applied': applied,
            'log_p_aa': lp_aa,
            'log_p_3di': lp_3di,
            'log_p_joint': lp_joint,
            'd_log_p_aa': lp_aa - base_aa,
            'd_log_p_3di': lp_3di - base_3di,
            'd_log_p_joint': lp_joint - base_joint,
            'seq': combined_seq,
            'struct': struct,
        })

        if checkpoint_path:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(rows)


def run_forward_chain(
    starting_seq, df_aa, df_3di,
    tokenizer, model, device,
    n_iter=1000,
    T=1.0,
    n_muts=1,
    alpha=0.5,
    checkpoint_path=None,
    checkpoint_every=10,
    annealing_rate = 1
):
    """
    Run Metropolis-Hastings MCMC over amino acid sequence space.

    Each step proposes n_muts single-site mutations (without replacement),
    maps the candidate to 3Di structure via split_3di, and scores with a
    weighted combination of AA and 3Di log-probabilities.

    Returns:
        seqs_chain   : list of accepted AA sequences (including start)
        structs_chain: list of corresponding 3Di structures
        log_p_chain  : list of corresponding joint log-probabilities
    """
    if checkpoint_path is None:
        checkpoint_path = f"chain_T{T}_alpha{alpha}_nmuts{n_muts}_niter{n_iter}.csv"

    seqs_chain = []
    structs_chain = []
    log_p_chain = []

    current_seq = starting_seq
    current_struct = split_3di(current_seq, tokenizer, model, device)
    current_struct = current_struct[:len(df_3di)]
    log_p_current = (
        alpha * log_probability(df_aa, current_seq)
        + (1 - alpha) * log_probability(df_3di, current_struct.upper())
    )
    seqs_chain.append(current_seq)
    structs_chain.append(current_struct)
    log_p_chain.append(log_p_current)

    for _ in tqdm(range(n_iter), desc="MCMC"):
        # Propose: mutate n_muts distinct positions without replacement
        proposed_seq = current_seq
        sites = np.random.choice(len(proposed_seq), size=n_muts, replace=False)
        for site_idx in sites:
            proposed_seq = resample_site(df_aa, proposed_seq, site_idx)

        proposed_struct = split_3di(proposed_seq, tokenizer, model, device)
        proposed_struct = proposed_struct[:len(df_3di)]

        log_p_proposed = (
            alpha * log_probability(df_aa, proposed_seq)
            + (1 - alpha) * log_probability(df_3di, proposed_struct.upper())
        )

        # Metropolis-Hastings accept/reject
        log_acceptance = min(0, (log_p_proposed - log_p_current) / T)
        if np.log(np.random.rand()) < log_acceptance:
            current_seq = proposed_seq
            current_struct = proposed_struct
            log_p_current = log_p_proposed
            seqs_chain.append(current_seq)
            structs_chain.append(current_struct)
            log_p_chain.append(log_p_current)

        if checkpoint_path and ((_ + 1) % checkpoint_every == 0):
            pd.DataFrame({
                'seq': seqs_chain,
                'struct': structs_chain,
                'log_p': log_p_chain,
            }).to_csv(checkpoint_path, index=False)
        T = T*annealing_rate

    return seqs_chain, structs_chain, log_p_chain
