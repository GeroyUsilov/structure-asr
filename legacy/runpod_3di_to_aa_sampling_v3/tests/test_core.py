from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from run_3di_to_aa_sampling import (
    AA_STATES,
    PROBABILITY_COLUMNS,
    balanced_chunk_slices,
    load_3di_posterior,
    maximum_likelihood_3di,
    normalize_generated_aa,
    read_checkpoint,
    sample_node,
    summarize_samples,
)


class CoreTests(unittest.TestCase):
    def test_balanced_chunks_preserve_length(self):
        self.assertEqual(balanced_chunk_slices(307, 500), [(0, 307)])
        self.assertEqual(balanced_chunk_slices(661, 500), [(0, 331), (331, 661)])

    def test_load_uses_probability_argmax(self):
        rows = []
        for site, winner in [(1, "Q"), (2, "A")]:
            row = {"Node": "NodeX", "Site": site, "State": "-"}
            row.update({column: 0.0 for column in PROBABILITY_COLUMNS})
            row[f"p_{winner}"] = 2.0
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            frame = load_3di_posterior(path, "NodeX")
        self.assertEqual(maximum_likelihood_3di(frame), "QA")
        self.assertTrue(np.allclose(frame["ML_3Di_Posterior"], 1.0))

    def test_sample_probabilities_sum_to_one(self):
        frame = pd.DataFrame(
            {
                "Site": [1, 2],
                "ML_3Di_State": ["A", "Q"],
                "ML_3Di_Posterior": [0.9, 0.8],
            }
        )
        table, summary = summarize_samples("NodeX", frame, ["AA", "AR", "XR", "AA"])
        probability_columns = [f"p_{state}" for state in AA_STATES + ["X"]]
        self.assertTrue(np.allclose(table[probability_columns].sum(axis=1), 1.0))
        self.assertEqual(table.loc[0, "p_A"], 0.75)
        self.assertEqual(table.loc[0, "p_X"], 0.25)
        self.assertEqual(table.loc[1, "p_R"], 0.5)
        self.assertEqual(summary["samples"], 4)

    def test_normalize_generated_aa_maps_ambiguous_to_x(self):
        self.assertEqual(normalize_generated_aa("A U Z O B J", 6), "AXXXXX")
        with self.assertRaisesRegex(RuntimeError, "lowercase 3Di tokens"):
            normalize_generated_aa("Aa", 2)

    def test_legacy_checkpoint_can_be_imported_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            path.write_text(json.dumps({"index": 0, "aa": "AA"}) + "\n")
            with self.assertRaisesRegex(ValueError, "incompatible checkpoint schema"):
                read_checkpoint(path, 2)
            self.assertEqual(read_checkpoint(path, 2, allow_legacy_unversioned=True), ["AA"])

    def test_sample_node_imports_legacy_without_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old" / "NodeX"
            target = root / "new" / "NodeX"
            source.mkdir(parents=True)
            target.mkdir(parents=True)
            records = [
                {"index": 0, "aa": "AA"},
                {"index": 1, "aa": "AR"},
            ]
            (source / "sampling_checkpoint.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            samples = sample_node(
                "NodeX", "AQ", target, None, None, None, None,
                2, 1, 42, 500, 1.1, 0.9, 6, 1.2, True, source,
            )
            self.assertEqual(samples, ["AA", "AR"])
            self.assertEqual(
                read_checkpoint(target / "sampling_checkpoint.jsonl", 2),
                ["AA", "AR"],
            )


if __name__ == "__main__":
    unittest.main()
