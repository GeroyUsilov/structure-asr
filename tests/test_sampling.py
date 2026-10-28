"""CPU tests: no model download; fake deterministic encoding is not biology."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from structure_asr import sampling, legacy_sampling
from structure_asr.posteriors import NodePosterior, load_state

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/ef_tu/mu1_tau2"


class SamplingTests(unittest.TestCase):
    def test_shared_posterior_loader(self):
        self.assertIs(sampling.load_state, legacy_sampling.load_state)
        posterior = load_state(EXAMPLE / "aa.state", 5e-6)["Node34"]
        self.assertEqual(posterior.probabilities.shape, (396, 20))
        np.testing.assert_allclose(posterior.probabilities.sum(axis=1), 1.0)
        self.assertTrue(np.all(posterior.probabilities > 0))

    def test_nonself_proposal_and_recorded_q(self):
        posterior = load_state(EXAMPLE / "aa.state", 5e-6)["Node34"]
        sequence = sampling.aa_map_sequence(posterior)
        state = SimpleNamespace(current_seq=sequence, rng=np.random.default_rng(731024))
        uncertainty, weights = sampling.uncertainty_site_weights(posterior, 0.9, 1)
        self.assertAlmostEqual(float(weights.sum()), 1)
        self.assertTrue(np.all(weights >= 0.1 / len(sequence) - 1e-15))
        self.assertTrue(np.all(uncertainty >= 0))
        for _ in range(50):
            candidate, _, forward, reverse, site = sampling.propose_sequence(state, posterior, weights)
            self.assertEqual(sum(a != b for a, b in zip(sequence, candidate)), 1)
            a = posterior.alphabet.index(sequence[site])
            b = posterior.alphabet.index(candidate[site])
            p = posterior.probabilities[site]
            self.assertAlmostEqual(math.exp(forward), weights[site] * p[b] / (1-p[a]))
            self.assertAlmostEqual(math.exp(reverse), weights[site] * p[a] / (1-p[b]))

    def test_hastings_detailed_balance_tempered_target(self):
        # A one-site finite state space allows an exact probability-flow check.
        p = np.array([0.6, 0.3, 0.1])
        structural = np.array([-4., -1., -2.])
        score = np.log(p) + 0.02 * structural
        target = np.exp(score / 2.)
        target /= target.sum()
        posterior = NodePosterior(tuple("ACD"), p[None, :], np.log(p[None, :]))
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                q_ab, q_ba = p[b]/(1-p[a]), p[a]/(1-p[b])
                ratio = (score[b]-score[a])/2. + math.log(q_ba/q_ab)
                flow_ab = target[a] * q_ab * min(1., math.exp(ratio))
                flow_ba = target[b] * q_ba * min(1., math.exp(-ratio))
                self.assertAlmostEqual(flow_ab, flow_ba)
        # A real call with this tiny posterior still makes a single change.
        state = SimpleNamespace(current_seq="A", rng=np.random.default_rng(1))
        self.assertNotEqual(sampling.propose_sequence(state, posterior, np.ones(1))[0], "A")

    def run_fake(self, module, output, fail_checkpoint=None):
        args = ["sample", "--aa-state", str(EXAMPLE / "aa.state"),
                "--three-di-state", str(EXAMPLE / "3di.state"),
                "--nodes-file", str(ROOT / "examples/ef_tu/node34.txt"),
                "--condition", "test", "--outdir", str(output),
                "--n-iter", "8", "--checkpoint-every", "2", "--resume"]
        def encode(sequences, *_args, **_kwargs):
            return list(sequences)

        writes = 0
        original_append = module.append_trace

        def append_trace(path, rows):
            nonlocal writes
            original_append(path, rows)
            writes += 1
            if writes == fail_checkpoint:
                raise RuntimeError("simulated interruption")

        with patch("sys.argv", args), patch.object(module, "load_prostt5", return_value=(
            None, None, None, None, Path("/fake/model")
        )), patch.object(module, "encode_sequences", encode), patch.object(
            module, "append_trace", append_trace
        ):
            module.main()

    def test_interruption_resume_matches_uninterrupted_for_both_samplers(self):
        for module in (sampling, legacy_sampling):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temp:
                full, resumed = Path(temp)/"full", Path(temp)/"resumed"
                self.run_fake(module, full)
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    self.run_fake(module, resumed, fail_checkpoint=2)
                self.run_fake(module, resumed)
                for filename in ("checkpoint.json", "trace.tsv", "joint_asr_best_aa.fasta",
                                 "joint_asr_best_3di.fasta", "chain_summary.tsv"):
                    self.assertEqual((full/filename).read_bytes(), (resumed/filename).read_bytes())
                self.run_fake(module, resumed)  # Completed resume must not duplicate trace rows.
                with (resumed/"trace.tsv").open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle, delimiter="\t"))), 8)
                self.assertEqual(json.loads((resumed/"run_manifest.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
