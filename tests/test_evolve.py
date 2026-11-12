from io import StringIO
import unittest

import numpy as np
from Bio import Phylo

from structure_asr.simulation.dca import DcaModel
from structure_asr.simulation.evolve import (
    DcaConditionalProposal,
    ForwardSimulator,
    MaskedSiteConditionalProposal,
    ProfileConditionalProposal,
    SimulationSettings,
)
from structure_asr.simulation.scoring import (
    CachedProfilePredictor,
    ConstantScorer,
    DcaScorer,
    PseudoPerplexityScorer,
)


class ForwardEvolutionTests(unittest.TestCase):
    def test_forward_simulation_records_every_node_and_is_reproducible(self):
        model = DcaModel.independent(("A", "B"), np.zeros((4, 2)))
        tree = Phylo.read(StringIO("((x:0.5,y:0.5)n:0.5,z:1.0)Root;"), "newick")
        settings = SimulationSettings(mutation_scale=2.0, seed=42, kernel="dca_gibbs")
        first = ForwardSimulator(DcaScorer(model), DcaConditionalProposal(model), settings).simulate(tree, "AAAA")
        second = ForwardSimulator(DcaScorer(model), DcaConditionalProposal(model), settings).simulate(tree, "AAAA")
        self.assertEqual(set(first.sequences), {"Root", "n", "x", "y", "z"})
        self.assertEqual(first.sequences, second.sequences)
        self.assertEqual(first.events, second.events)
        self.assertTrue(all(len(sequence) == 4 for sequence in first.sequences.values()))

    def test_ofs_pseudogibbs_mirrors_uniform_site_conditional_updates(self):
        calls = []

        def raw_profile(sequence):
            calls.append(sequence)
            # Every context overwhelmingly proposes B at every site.
            return np.tile(np.asarray([0.0, 1.0]), (len(sequence), 1))

        cached = CachedProfilePredictor(raw_profile, max_size=2)
        scorer = PseudoPerplexityScorer(cached, alphabet=("A", "B"), name="ofs")
        proposal = ProfileConditionalProposal(cached, alphabet=("A", "B"))
        tree = Phylo.read(StringIO("(x:1.0)Root;"), "newick")
        settings = SimulationSettings(mutation_scale=1.0, seed=3, kernel="ofs_pseudogibbs")
        result = ForwardSimulator(scorer, proposal, settings).simulate(tree, "AAAA")
        self.assertEqual(len(result.events), 4)
        self.assertTrue(all(event.accepted for event in result.events))
        self.assertTrue(set(result.sequences["x"]) <= {"A", "B"})
        # Root plus at most one new model evaluation per update; cache prevents
        # separate proposal, reverse-proposal, and scorer calls.
        self.assertLessEqual(len(calls), len(result.events) + 1)

    def test_exact_masked_mlm_uses_one_predictor_call_per_update(self):
        calls = []

        def site_profile(sequence, site):
            calls.append((sequence, site))
            return np.asarray([0.0, 1.0])

        proposal = MaskedSiteConditionalProposal(site_profile, alphabet=("A", "B"))
        tree = Phylo.read(StringIO("(x:1.0)Root;"), "newick")
        settings = SimulationSettings(mutation_scale=1.0, seed=3, kernel="exact_mlm_gibbs")
        result = ForwardSimulator(ConstantScorer(), proposal, settings).simulate(tree, "AAAA")
        self.assertEqual(len(result.events), 4)
        self.assertEqual(len(calls), len(result.events))
        self.assertTrue(all(event.accepted for event in result.events))
        self.assertTrue(all(event.new_state == "B" for event in result.events))
        self.assertIn("B", result.sequences["x"])
        self.assertTrue(set(result.sequences["x"]) <= {"A", "B"})

    def test_branch_checkpoint_resume_matches_uninterrupted_simulation(self):
        model = DcaModel.independent(("A", "B"), np.zeros((4, 2)))
        tree = Phylo.read(StringIO("(x:0.75,y:0.75)Root;"), "newick")
        settings = SimulationSettings(mutation_scale=2.0, seed=9, kernel="dca_gibbs")
        simulator = ForwardSimulator(DcaScorer(model), DcaConditionalProposal(model), settings)
        uninterrupted = simulator.simulate(tree, "AAAA")
        partial = None

        class PlannedInterruption(Exception):
            pass

        def stop_after_first_edge(completed_edges, _total_edges, result):
            nonlocal partial
            partial = result
            if completed_edges == 1:
                raise PlannedInterruption

        with self.assertRaises(PlannedInterruption):
            simulator.simulate(tree, "AAAA", edge_completed=stop_after_first_edge)
        resumed = simulator.simulate(
            tree,
            "AAAA",
            initial_result=partial,
            start_edge=1,
        )
        self.assertEqual(resumed.sequences, uninterrupted.sequences)
        self.assertEqual(resumed.events, uninterrupted.events)
