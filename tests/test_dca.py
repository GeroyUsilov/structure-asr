import tempfile
import unittest
from pathlib import Path

import numpy as np

from forward_asr_benchmark.dca import DcaModel


def make_model() -> DcaModel:
    fields = np.asarray([[0.3, -0.2], [0.1, 0.4], [-0.1, 0.2]])
    couplings = np.zeros((3, 3, 2, 2))
    couplings[0, 1] = np.asarray([[0.5, -0.1], [-0.2, 0.6]])
    couplings[0, 2] = np.asarray([[0.2, -0.3], [0.1, 0.4]])
    couplings[1, 2] = np.asarray([[0.0, 0.2], [-0.4, 0.5]])
    return DcaModel(("A", "B"), fields, couplings)


class DcaModelTests(unittest.TestCase):
    def test_conditional_log_odds_match_full_energy_difference(self):
        model = make_model()
        sequence = "ABA"
        for site in range(model.length):
            logits = model.conditional_logits(sequence, site)
            variants = [sequence[:site] + state + sequence[site + 1 :] for state in model.alphabet]
            energies = np.asarray([model.energy(variant) for variant in variants])
            # Conditional logits equal -energy up to a site-independent constant.
            np.testing.assert_allclose(logits - logits[0], -energies + energies[0])

    def test_save_load_round_trip(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = DcaModel.load(path)
        self.assertEqual(loaded.alphabet, model.alphabet)
        np.testing.assert_allclose(loaded.fields, model.fields)
        np.testing.assert_allclose(loaded.couplings, model.couplings)
        self.assertEqual(loaded.energy("ABA"), model.energy("ABA"))
