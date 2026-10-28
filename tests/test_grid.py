import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
import zlib
from io import StringIO
from pathlib import Path

import numpy as np
from Bio import Phylo

from forward_asr_benchmark.evolve import (
    ForwardSimulator,
    MaskedSiteConditionalProposal,
    SimulationSettings,
)
from forward_asr_benchmark.io import read_fasta, write_fasta
from structure_asr import grid as grid_module
from structure_asr import iqtree
from structure_asr.batched_forward import (
    Cell,
    finished_result,
    historical_scorer,
    load_checkpoint,
    simulate_cells,
    write_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
NEWICK = "((a:0.3,(b:0.2,c:0.5):0.1):0.4,(d:0.7,e:0.05):0.2);"


def profile(sequence: str, site: int) -> np.ndarray:
    """Deterministic stand-in for a masked-site LM conditional."""
    rng = np.random.default_rng(zlib.crc32(f"{sequence}|{site}".encode()))
    return rng.dirichlet(np.full(20, 0.3))


def batched(sequences, sites):
    return np.stack([profile(sequence, site) for sequence, site in zip(sequences, sites)])


def tree():
    from forward_asr_benchmark.tree import load_tree

    handle = tempfile.NamedTemporaryFile("w", suffix=".nwk", delete=False)
    handle.write(NEWICK)
    handle.close()
    try:
        return load_tree(handle.name)
    finally:
        os.unlink(handle.name)


def serial(root: str, mu: float, tau: float, seed: int):
    proposal = MaskedSiteConditionalProposal(profile, proposal_temperature=tau)
    settings = SimulationSettings(mutation_scale=mu, seed=seed, kernel="exact_mlm_gibbs")
    return ForwardSimulator(historical_scorer(), proposal, settings).simulate(tree(), root)


class BatchedForwardTests(unittest.TestCase):
    root = "MKTAYIAKQRQISFVKSHFSRQ"
    regimes = [(mu, tau, seed) for mu in (1.0, 3.0) for tau in (1.0, 2.0) for seed in (5, 6)]

    def cells(self):
        return [Cell(f"c{i}", tree(), self.root, mu, tau, seed)
                for i, (mu, tau, seed) in enumerate(self.regimes)]

    def test_batched_lanes_reproduce_serial_simulator_for_any_batch_size(self):
        expected = [serial(self.root, *regime) for regime in self.regimes]
        for max_batch in (1, 3, 64):
            cells = self.cells()
            states = simulate_cells(cells, batched, max_batch=max_batch)
            for cell, state, reference in zip(cells, states, expected):
                result = finished_result(cell, state)
                self.assertEqual(list(result.sequences.items()), list(reference.sequences.items()))
                self.assertEqual(result.events, reference.events)
                self.assertEqual(result.scores, reference.scores)

    def test_interrupted_grid_resumes_to_uninterrupted_result(self):
        expected = [serial(self.root, *regime) for regime in self.regimes]
        cells = self.cells()
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{cell.label}.json" for cell in cells]
            completed = 0

            class Interrupt(Exception):
                pass

            def save(index, state):
                nonlocal completed
                write_checkpoint(paths[index], cells[index], state)
                completed += 1
                if completed == 13:
                    raise Interrupt

            with self.assertRaises(Interrupt):
                simulate_cells(cells, batched, max_batch=5, edge_completed=save)
            resumed_cells = self.cells()
            progress = [load_checkpoint(path, cell) for path, cell in zip(paths, resumed_cells)]
            self.assertTrue(any(state is not None and 0 < len(state.completed_edges) < 10
                                for state in progress))
            states = simulate_cells(resumed_cells, batched, max_batch=2, progress=progress)
            for cell, state, reference in zip(resumed_cells, states, expected):
                result = finished_result(cell, state)
                self.assertEqual(result.sequences, reference.sequences)
                self.assertEqual(result.events, reference.events)

    def test_checkpoint_rejects_a_different_cell(self):
        cell, other = self.cells()[:2]
        state = simulate_cells([cell], batched)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            write_checkpoint(path, cell, state)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path, other)


def write_grid(directory: Path, **asr) -> Path:
    (directory / "tree.nwk").write_text(NEWICK, encoding="utf-8")
    write_fasta(directory / "aln.fasta", {"src|42": "MKTAYIAKQR-QISFVKSHFSRQ"})
    asr_lines = "\n".join(f"{key} = {json.dumps(value)}" for key, value in asr.items())
    path = directory / "grid.toml"
    path.write_text(textwrap.dedent(f"""\
        [grid]
        name = "toy"
        outdir = "out"
        tree = "tree.nwk"
        alignment = "aln.fasta"
        root_accession = "42"
        rates = [1, 2]
        proposal_temperatures = [1.5]
        seeds = [3, 4]
        max_attempts_per_cell = 1000

        [asr]
        """) + asr_lines + "\n", encoding="utf-8")
    return path


def fake_encoder(identifiers, sequences, _checkpoint, _resume):
    return [sequence[::-1] for sequence in sequences]


class GridStageTests(unittest.TestCase):
    def test_plan_counts_match_exact_planned_attempts_and_cap_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            grid = grid_module.load_grid(write_grid(Path(directory)))
            payload = grid_module.plan(grid)
            self.assertEqual(len(payload["cells"]), 4)
            self.assertEqual(
                [cell["planned_attempts"] for cell in payload["cells"]],
                [grid.planned_attempts(spec) for spec in grid.cells],
            )
            grid.max_attempts_per_cell = 1
            with self.assertRaises(RuntimeError):
                grid_module.plan(grid)

    def test_shards_partition_cells_and_match_one_process_output(self):
        with tempfile.TemporaryDirectory() as directory:
            config = write_grid(Path(directory))
            whole = grid_module.load_grid(config, Path(directory) / "whole")
            grid_module.run_forward(whole, batched, specs=whole.cells, max_batch=4)
            sharded = grid_module.load_grid(config, Path(directory) / "sharded")
            labels = []
            for shard in ("0/2", "1/2"):
                specs = sharded.shard(shard)
                labels.extend(spec.label for spec in specs)
                grid_module.run_forward(sharded, batched, specs=specs, max_batch=1)
            self.assertEqual(sorted(labels), sorted(spec.label for spec in whole.cells))
            for spec in whole.cells:
                for name in ("ground_truth_nodes.fasta", "leaves.fasta", "events.tsv"):
                    self.assertEqual(
                        (whole.cell_dir(spec) / "simulation" / name).read_text(),
                        (sharded.cell_dir(spec) / "simulation" / name).read_text(),
                    )
            spec = whole.cells[0]
            reference = serial(whole.root_sequence, spec.mutation_scale,
                               spec.proposal_temperature, spec.seed)
            self.assertEqual(
                read_fasta(whole.cell_dir(spec) / "simulation" / "ground_truth_nodes.fasta"),
                reference.sequences,
            )
            # A completed grid is skipped on rerun.
            self.assertEqual(grid_module.run_forward(whole, batched, specs=whole.cells, max_batch=4), [])

    def test_encode_writes_tip_inputs_and_separate_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            grid = grid_module.load_grid(write_grid(Path(directory)))
            with self.assertRaises(RuntimeError):
                grid_module.run_encode(grid, fake_encoder, specs=grid.cells, shard_name="all",
                                       batch_size=2, model_label="fake", model_info={}, resume=False)
            grid_module.run_forward(grid, batched, specs=grid.cells, max_batch=8)
            grid_module.run_encode(grid, fake_encoder, specs=grid.cells, shard_name="all",
                                   batch_size=2, model_label="fake", model_info={}, resume=False)
            cell = grid.cell_dir(grid.cells[0])
            tips = read_fasta(cell / "three_di" / "leaves_3di.fasta")
            truth = read_fasta(cell / "truth" / "ground_truth_nodes_3di.fasta")
            self.assertEqual(set(tips), {"a", "b", "c", "d", "e"})
            self.assertEqual(len(truth), 9)
            self.assertTrue(all(truth[name] == sequence for name, sequence in tips.items()))


STUB_IQTREE = r'''#!{python}
"""Minimal IQ-TREE stand-in: records argv, writes .state/.treefile/.iqtree."""
import json, re, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("IQ-TREE version 0.0-stub"); sys.exit(0)
opt = {{args[i]: args[i + 1] for i in range(0, len(args) - 1) if args[i].startswith("-")}}
prefix = opt["--prefix"]
seqs, name = {{}}, None
for line in open(opt["-s"]):
    line = line.strip()
    if line.startswith(">"):
        name = line[1:]; seqs[name] = ""
    elif line:
        seqs[name] += line
# Like IQ-TREE, report an unrooted tree with NodeN internal labels (toy tips only).
assert open(opt["-te"]).read().strip() == "((a,(b,c)),(d,e));"
topology = "(a,(b,c)Node2,(d,e)Node3)Node1;"
open(prefix + ".treefile", "w").write(topology + "\n")
length = len(next(iter(seqs.values())))
cols = "ARNDCQEGHILKMFPSTWYV"
with open(prefix + ".state", "w") as h:
    h.write("# stub\nNode\tSite\tState\t" + "\t".join("p_" + c for c in cols) + "\n")
    for node in sorted(set(re.findall(r"Node\d+", topology))):
        for site in range(1, length + 1):
            h.write("%s\t%d\tA\t" % (node, site) + "\t".join("1.00000" if c == "A" else "0.00000" for c in cols) + "\n")
model = "Q3DI_AF+G4" if "-mdef" in opt else "LG+G4"
open(prefix + ".iqtree", "w").write("Best-fit model according to BIC: %s\n" % model)
open(prefix + ".log", "w").write(json.dumps(args))
'''


class IqtreeStageTests(unittest.TestCase):
    def stub(self, directory: Path) -> str:
        path = directory / "iqtree_stub"
        path.write_text(STUB_IQTREE.format(python=sys.executable), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return str(path)

    def test_topology_is_label_and_length_free(self):
        newick = iqtree.topology_newick(tree())
        self.assertEqual(newick, "((a,(b,c)),(d,e));")

    def test_matrix_definition_round_trips_published_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            values = [str(i + 1) for i in range(210)]
            matrix = Path(directory) / "Q.test"
            matrix.write_text(" ".join(values), encoding="utf-8")
            definition = iqtree.write_matrix_definition(matrix, Path(directory) / "m.nex", "QT")
            text = definition.read_text()
            self.assertIn("model QT =", text)
            body = text.split("=", 1)[1].split(";")[0].split()
            self.assertEqual(body, values)
            matrix.write_text("1 2 3", encoding="utf-8")
            with self.assertRaises(ValueError):
                iqtree.write_matrix_definition(matrix, Path(directory) / "bad.nex")

    def test_grid_asr_is_blind_resumable_and_evaluable(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            matrix = base / "Q.test"
            matrix.write_text(" ".join(["1"] * 210), encoding="utf-8")
            grid = grid_module.load_grid(write_grid(base, three_di_matrix=str(matrix)))
            grid_module.run_forward(grid, batched, specs=grid.cells, max_batch=8)
            grid_module.run_encode(grid, fake_encoder, specs=grid.cells, shard_name="all",
                                   batch_size=2, model_label="fake", model_info={}, resume=False)
            # Blinding: ASR must succeed with every truth file absent.
            hidden = {}
            for spec in grid.cells:
                cell = grid.cell_dir(spec)
                for path in (cell / "simulation" / "ground_truth_nodes.fasta",
                             cell / "truth" / "ground_truth_nodes_3di.fasta"):
                    hidden[path] = path.read_bytes()
                    path.unlink()
            executable = self.stub(base)
            manifests = grid_module.run_asr(grid, specs=grid.cells, executable=executable, workers=3)
            self.assertEqual(len(manifests), 8)
            by_modality = {manifest["modality"]: manifest for manifest in manifests}
            aa_command = by_modality["aa"]["command"]
            self.assertIn("-te", aa_command)
            self.assertIn("--ancestral", aa_command)
            self.assertEqual(aa_command[aa_command.index("-m") + 1], "MFP")
            self.assertNotIn("-blfix", aa_command)
            three_di_command = by_modality["3di"]["command"]
            self.assertEqual(three_di_command[three_di_command.index("-mset") + 1], "Q3DI_AF")
            self.assertEqual(by_modality["3di"]["best_fit_model"], "Q3DI_AF+G4")
            for path, content in hidden.items():
                path.write_bytes(content)
            # Rerun skips finished jobs; IQ-TREE outputs are not rewritten.
            state = grid.cell_dir(grid.cells[0]) / "asr" / "aa.state"
            before = state.stat().st_mtime_ns
            grid_module.run_asr(grid, specs=grid.cells, executable=executable, workers=1)
            self.assertEqual(state.stat().st_mtime_ns, before)
            rows = grid_module.run_evaluate(grid, specs=grid.cells, outdir=base / "evaluation")
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["matched_nodes"] == 3 for row in rows))
            self.assertTrue((base / "evaluation" / "dataset_summary.tsv").exists())

    def test_changed_inputs_block_silent_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            topology = iqtree.write_topology(tree(), base / "topology.nwk")
            alignment = base / "aa.fasta"
            write_fasta(alignment, {name: "MKT" for name in "abcde"})
            job = iqtree.AsrJob("aa", alignment, topology, base / "asr" / "aa", seed=1)
            executable = self.stub(base)
            iqtree.run_job(job, executable, "stub")
            write_fasta(alignment, {name: "MKS" for name in "abcde"})
            with self.assertRaises(RuntimeError):
                iqtree.run_job(job, executable, "stub")


@unittest.skipUnless(os.environ.get("ASR_IQTREE"), "set ASR_IQTREE to run real IQ-TREE")
class RealIqtreeTests(unittest.TestCase):
    def test_real_aa_reconstruction_on_bundled_tips_maps_all_nodes(self):
        from forward_asr_benchmark.tree import load_tree
        from structure_asr.evaluation import read_state, topology_mapping

        reference = load_tree(ROOT / "examples/ef_tu/reference/AB_eftu_rerooted.tre")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            topology = iqtree.write_topology(reference, base / "topology.nwk")
            job = iqtree.AsrJob(
                "aa", ROOT / "examples/ef_tu/tips/aa/ef_tu_tips_mu1_tau2_aa.fasta",
                topology, base / "aa", seed=731024, threads=1, model="LG+G4",
            )
            executable = os.environ["ASR_IQTREE"]
            manifest = iqtree.run_job(job, executable, iqtree.iqtree_version(executable))
            self.assertTrue(manifest["best_fit_model"].startswith("LG"))
            inferred = Phylo.read(str(base / "aa.treefile"), "newick")
            mapping = topology_mapping(reference, inferred)
            self.assertEqual(len(mapping), 37)
            self.assertTrue(all(row["matched"] for row in mapping))
            _columns, states = read_state(base / "aa.state")
            self.assertEqual(len(states), 37)
            self.assertTrue(all(len(rows) == 396 for rows in states.values()))


if __name__ == "__main__":
    unittest.main()
