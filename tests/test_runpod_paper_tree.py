from io import StringIO
import tempfile
import unittest
from pathlib import Path

from Bio import Phylo

from runpod_paper_tree_ofs.prepare_paper_tree import (
    deterministic_subtree,
    mean_root_to_leaf,
    normalize_mean_root_to_leaf,
)
from runpod_paper_tree_ofs.run_sweep import planned_attempts


class PaperTreeRunpodTests(unittest.TestCase):
    def test_normalization_sets_mean_root_to_leaf_to_one(self):
        tree = Phylo.read(StringIO("((a:1,b:2):1,(c:3,d:4):1)Root;"), "newick")
        normalize_mean_root_to_leaf(tree)
        self.assertAlmostEqual(mean_root_to_leaf(tree), 1.0)

    def test_deterministic_subtree_is_stable_and_normalized(self):
        source = "(((a:1,b:1):1,(c:1,d:1):1):1,((e:1,f:1):1,(g:1,h:1):1):1)Root;"
        first = deterministic_subtree(Phylo.read(StringIO(source), "newick"), 4)
        second = deterministic_subtree(Phylo.read(StringIO(source), "newick"), 4)
        self.assertEqual([x.name for x in first.get_terminals()], [x.name for x in second.get_terminals()])
        self.assertEqual(len(first.get_terminals()), 4)
        self.assertAlmostEqual(mean_root_to_leaf(first), 1.0)

    def test_planned_attempts_uses_every_branch(self):
        tree = Phylo.read(StringIO("(a:0.5,b:0.5)Root;"), "newick")
        self.assertEqual(planned_attempts(tree, sequence_length=100, rate=2), 200)


if __name__ == "__main__":
    unittest.main()

