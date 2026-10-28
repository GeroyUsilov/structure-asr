import csv
import tempfile
import unittest
from pathlib import Path

from structure_asr.benchmark_examples import evaluate, errors

ROOT = Path(__file__).resolve().parents[1]


class WorkedExampleTests(unittest.TestCase):
    def test_reproduce_completed_ef_tu_benchmarks(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = evaluate(ROOT / "examples/ef_tu", Path(temp))
            self.assertEqual(summary["legacy_aa_errors_start"], 68)
            self.assertEqual(summary["legacy_aa_errors_best"], 75)
            with (Path(temp)/"dataset_summary.tsv").open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(int(row["matched_nodes"]) == 37 for row in rows))
            for row in rows:
                if row["modality"] == "aa":
                    self.assertAlmostEqual(float(row["map_accuracy"]), 0.9790, places=4)
            with (Path(temp)/"legacy_joint_comparison.tsv").open() as handle:
                comparisons = list(csv.DictReader(handle, delimiter="\t"))
            node34 = next(r for r in comparisons if r["condition"] == "mu1_tau2" and r["node"] == "Node34")
            self.assertEqual([int(node34[k]) for k in ("aa_start_errors", "aa_best_errors", "3di_start_errors", "3di_best_errors")], [7, 12, 81, 65])

    def test_hamming_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            errors("AAA", "AA")


if __name__ == "__main__":
    unittest.main()
