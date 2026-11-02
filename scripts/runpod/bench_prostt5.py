#!/usr/bin/env python3
"""Time ProstT5 greedy AA->3Di encoding against batch size on this GPU.

Uses the sampler's own path (structure_asr.sampling.encode_sequences), so the
numbers apply directly to `asr-grid sample --model-batch-size`. Sequences are
random single-site variants of a grid cell's simulated root, the same length
as real proposals. Prints one TSV row per batch size.

  python scripts/runpod/bench_prostt5.py --fasta results/grids/calibration/<cell>/root_sequence.fasta
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from forward_asr_benchmark.io import read_fasta
from structure_asr.prostt5 import load_prostt5
from structure_asr.sampling import encode_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--model", default="Rostlab/ProstT5")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 32, 64, 128, 256])
    parser.add_argument("--max-chunk-aa", type=int, default=500)
    args = parser.parse_args()
    root = next(iter(read_fasta(args.fasta).values()))
    rng = np.random.default_rng(0)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    torch, tokenizer, model, device, _resolved = load_prostt5(args.model, "cuda", "float16")
    encode_sequences([root, root], tokenizer, model, device, torch, 2, args.max_chunk_aa)  # warm-up
    print("batch_size\tsequences\tseconds\tseconds_per_batch\tsequences_per_second\tpeak_gb", flush=True)
    for batch in args.batch_sizes:
        sequences = []
        for _ in range(batch):
            site = int(rng.integers(len(root)))
            sequences.append(root[:site] + alphabet[int(rng.integers(20))] + root[site + 1:])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        encode_sequences(sequences, tokenizer, model, device, torch, batch, args.max_chunk_aa)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{batch}\t{batch}\t{seconds:.2f}\t{seconds:.2f}\t{batch / seconds:.2f}\t{peak:.1f}", flush=True)


if __name__ == "__main__":
    main()
