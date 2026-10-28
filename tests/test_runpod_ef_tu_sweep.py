import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest

import numpy as np
from Bio import Phylo

from forward_asr_benchmark.evolve import (
    ForwardSimulator,
    SimulationSettings,
    UniformSubstitutionProposal,
)
from forward_asr_benchmark.scoring import ConstantScorer
from runpod_exact_mlm_ef_tu_sweep.analyze_3di_hamming import hamming_matrices
from runpod_exact_mlm_ef_tu_sweep.convert_tips_to_3di import (
    balanced_chunk_slices,
    has_model_weights,
    load_checkpoint,
)
from runpod_exact_mlm_ef_tu_sweep.run_sweep import (
    exact_planned_attempts,
    reconcile_timings_with_checkpoint,
    write_config,
)


class EfTuSweepTests(unittest.TestCase):
    def test_prostt5_balanced_chunks_match_existing_workflow(self):
        self.assertEqual(balanced_chunk_slices(396, 500), [(0, 396)])
        self.assertEqual(balanced_chunk_slices(661, 500), [(0, 331), (331, 661)])

    def test_model_weight_detection_supports_bin_and_safetensors(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            self.assertFalse(has_model_weights(base))
            (base / "pytorch_model.bin").touch()
            self.assertTrue(has_model_weights(base))

    def test_3di_checkpoint_ignores_incomplete_final_line(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.jsonl"
            checkpoint.write_text(
                json.dumps(
                    {
                        "index": 0,
                        "identifier": "tip_a",
                        "aa": "ACDE",
                        "three_di": "ACDE",
                    }
                )
                + "\n{\"index\":",
                encoding="utf-8",
            )
            self.assertEqual(
                load_checkpoint(checkpoint, ["tip_a", "tip_b"], ["ACDE", "FGHI"]),
                {0: "ACDE"},
            )

    def test_pairwise_hamming_counts_and_normalization(self):
        encoded = np.asarray(
            [list("AAAA"), list("AACA"), list("CCCC")], dtype="U1"
        )
        counts, normalized = hamming_matrices(encoded)
        np.testing.assert_array_equal(
            counts,
            np.asarray([[0, 1, 4], [1, 0, 3], [4, 3, 0]], dtype=np.int32),
        )
        np.testing.assert_allclose(normalized, counts / 4.0)

    def test_generated_config_enables_progress_and_branch_checkpoints(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            config_path = base / "config.toml"
            write_config(
                config_path,
                tree=base / "tree.nwk",
                root=base / "root.fasta",
                output=base / "simulation",
                timings=base / "timings.tsv",
                checkpoint=base / "checkpoint.json",
                rate=0.5,
                planned_attempts=2391,
                seed=731024,
                model_name="facebook/esm2_t33_650M_UR50D",
                device="cuda",
                dtype="float16",
                proposal_temperature=1.0,
            )
            with config_path.open("rb") as handle:
                config = tomllib.load(handle)
            self.assertEqual(config["simulation"]["checkpoint_every_branches"], 1)
            self.assertEqual(config["exact_mlm"]["parameters"]["progress_total"], 2391)
            self.assertTrue(config["exact_mlm"]["parameters"]["resume_timings"])

    def test_exact_plan_matches_simulator_stochastic_rounding(self):
        tree = Phylo.read(StringIO("((x:0.13,y:0.27)n:0.41,z:0.09)Root;"), "newick")
        length = 7
        rate = 2.0
        seed = 731024
        planned = exact_planned_attempts(tree, length, rate, seed)
        result = ForwardSimulator(
            ConstantScorer(),
            UniformSubstitutionProposal(("A", "B")),
            SimulationSettings(mutation_scale=rate, seed=seed),
        ).simulate(tree, "A" * length)
        self.assertEqual(planned, len(result.events))

    def test_timing_reconciliation_discards_incomplete_branch_rows(self):
        header = (
            "pass\tsequence_length\tsite_zero_based\tforward_seconds\t"
            "end_to_end_seconds\n"
        )
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            checkpoint = base / "checkpoint.json"
            timings = base / "timings.tsv"
            checkpoint.write_text(json.dumps({"events": [{}, {}]}), encoding="utf-8")
            timings.write_text(
                header
                + "1\t396\t1\t0.04\t0.05\n"
                + "2\t396\t2\t0.04\t0.05\n"
                + "3\t396\t3\t0.04\t0.05\n",
                encoding="utf-8",
            )
            completed = reconcile_timings_with_checkpoint(checkpoint, timings)
            self.assertEqual(completed, 2)
            self.assertEqual(len(timings.read_text(encoding="utf-8").splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
