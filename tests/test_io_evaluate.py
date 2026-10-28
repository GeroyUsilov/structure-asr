from io import StringIO
import tempfile
import unittest
from pathlib import Path

from Bio import Phylo

from forward_asr_benchmark.evaluate import compare_reconstructions
from forward_asr_benchmark.evolve import EvolutionResult, MutationEvent
from forward_asr_benchmark.io import write_fasta, write_run
from forward_asr_benchmark.scoring import ScoreResult


class OutputTests(unittest.TestCase):
    def test_run_outputs_and_ground_truth_evaluation(self):
        tree = Phylo.read(StringIO("(tip:0.1)Root;"), "newick")
        result = EvolutionResult(
            sequences={"Root": "AAAA", "tip": "AABA"},
            scores={
                "Root": ScoreResult(0.0, {"x": 1.0}),
                "tip": ScoreResult(1.0, {"x": 2.0}),
            },
            events=[MutationEvent("Root", "tip", 1, 2, "A", "B", True, 0.0, 0.0, 1.0)],
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / "run"
            write_run(output, tree, result, {"schema_version": 1})
            reconstructed = base / "reconstructed.fasta"
            write_fasta(reconstructed, {"Root": "AAAB"})
            rows = compare_reconstructions(output / "ground_truth_nodes.fasta", reconstructed)
            self.assertEqual(rows[0]["hamming_distance"], 1)
            self.assertEqual(rows[0]["normalized_hamming"], 0.25)
            self.assertTrue((output / "leaves.fasta").read_text().startswith(">tip"))
