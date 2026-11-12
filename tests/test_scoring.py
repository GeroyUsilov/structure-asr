import unittest

import numpy as np

from structure_asr.simulation.scoring import (
    AlphaFoldUncertaintyScorer,
    CachedProfilePredictor,
    PseudoPerplexityScorer,
)


class ScoringTests(unittest.TestCase):
    def test_pseudo_perplexity_uses_observed_residue_probabilities(self):
        def profile(_sequence):
            return np.asarray([[0.8, 0.2], [0.25, 0.75]])

        result = PseudoPerplexityScorer(profile, alphabet=("A", "B")).score("AB")
        expected_energy = -np.mean(np.log([0.8, 0.75]))
        self.assertAlmostEqual(result.energy, expected_energy)
        self.assertAlmostEqual(result.metrics["pseudo_perplexity"], np.exp(expected_energy))

    def test_alphafold_energy_rewards_confidence_and_reference_fold(self):
        good = AlphaFoldUncertaintyScorer(
            lambda _sequence: {"mean_plddt": 95, "ptm": 0.9, "mean_pae": 2, "reference_tm": 0.95}
        ).score("AAAA")
        bad = AlphaFoldUncertaintyScorer(
            lambda _sequence: {"mean_plddt": 60, "ptm": 0.5, "mean_pae": 15, "reference_tm": 0.4}
        ).score("AAAA")
        self.assertLess(good.energy, bad.energy)

    def test_alphafold_requires_same_fold_metric_by_default(self):
        scorer = AlphaFoldUncertaintyScorer(lambda _sequence: {"mean_plddt": 99})
        with self.assertRaisesRegex(ValueError, "reference_tm"):
            scorer.score("AAAA")

    def test_cached_profile_predictor_reuses_recent_sequence(self):
        calls = []

        def predictor(sequence):
            calls.append(sequence)
            return np.ones((len(sequence), 2))

        cached = CachedProfilePredictor(predictor, max_size=2)
        cached("AA")
        cached("AA")
        self.assertEqual(calls, ["AA"])
