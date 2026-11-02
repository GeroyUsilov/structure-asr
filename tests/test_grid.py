import csv
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
from unittest.mock import patch

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
from structure_asr.fold_conservation import FoldFilter, FoldSettings, three_di_identity
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


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
THREEDI = "ACDEFGHIKLMNPQRSTVWY"


def fake_three_di(sequences):
    """Residue-wise AA->3Di map, so 3Di identity tracks AA identity."""
    return ["".join(THREEDI[ALPHABET.index(residue) % len(THREEDI)] for residue in sequence)
            for sequence in sequences]


class FoldConservationTests(unittest.TestCase):
    root = "MKTAYIAKQRQISFVKSHFSRQ"

    def cell(self, fold, mu=3.0, tau=1.0, seed=5):
        return Cell("fold", tree(), self.root, mu, tau, seed, fold=fold)

    def run_cell(self, fold, encode=fake_three_di, **kwargs):
        cell = self.cell(fold, **kwargs)
        state = simulate_cells([cell], batched, max_batch=4, fold_encode=encode)[0]
        return cell, finished_result(cell, state)

    def test_an_impossible_threshold_reverts_every_block(self):
        _cell, result = self.run_cell(FoldSettings(min_identity=1.0, block=2))
        self.assertTrue(all(sequence == self.root for sequence in result.sequences.values()))
        self.assertTrue(result.events and not any(event.accepted for event in result.events))

    def test_a_permissive_threshold_reproduces_the_legacy_generator(self):
        reference = serial(self.root, 3.0, 1.0, 5)
        _cell, result = self.run_cell(FoldSettings(min_identity=0.0, block=2))
        self.assertEqual(result.sequences, reference.sequences)
        self.assertEqual(result.events, reference.events)

    def test_structural_selection_keeps_tips_closer_to_the_root_fold(self):
        def identity_to_root(result):
            root_3di, = fake_three_di([self.root])
            tips = [result.sequences[name] for name in ("a", "b", "c", "d", "e")]
            return min(three_di_identity(encoded, root_3di)
                       for encoded in fake_three_di(tips))

        free = identity_to_root(serial(self.root, 3.0, 1.0, 5))
        _cell, selected = self.run_cell(FoldSettings(min_identity=0.8, block=1))
        self.assertGreaterEqual(identity_to_root(selected), 0.8)
        self.assertGreater(identity_to_root(selected), free)

    def test_a_rejected_block_keeps_its_draws_in_the_record_as_unaccepted(self):
        _cell, result = self.run_cell(FoldSettings(min_identity=0.95, block=4))
        self.assertTrue(any(not event.accepted for event in result.events))
        self.assertTrue(any(event.accepted for event in result.events))
        # Rejections come in whole blocks, so a node's realized changes only
        # ever reflect accepted events.
        for name, sequence in result.sequences.items():
            if name == "Root":
                continue
            self.assertEqual(len(sequence), len(self.root))

    def test_block_size_sets_the_number_of_3di_predictions(self):
        counts = {}
        for block in (1, 5):
            calls = []

            def counting(sequences, calls=calls):
                calls.extend(sequences)
                return fake_three_di(sequences)

            self.run_cell(FoldSettings(min_identity=0.5, block=block), encode=counting)
            counts[block] = len(calls)
        self.assertLess(counts[5], counts[1])

    def test_metropolis_always_keeps_a_block_that_improves_identity(self):
        fold = FoldSettings(criterion="metropolis", beta=50.0, block=1)
        filter_ = FoldFilter(fold, fake_three_di)
        better, worse = self.root, "A" + self.root[1:]
        filter_.prepare([self.root, better, worse])
        rng = np.random.default_rng(0)
        self.assertTrue(filter_.accepts(better, worse, self.root, rng)[0])

    def test_fold_settings_are_part_of_the_cell_fingerprint(self):
        legacy = self.cell(None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            cell, _result = self.run_cell(FoldSettings(min_identity=0.5, block=2))
            state = simulate_cells([cell], batched, max_batch=4, fold_encode=fake_three_di)[0]
            write_checkpoint(path, cell, state)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path, legacy)

    def test_a_fold_cell_without_an_encoder_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate_cells([self.cell(FoldSettings())], batched)

    def test_invalid_fold_settings_are_rejected(self):
        for payload in ({"reference": "tip"}, {"criterion": "softmax"}, {"block": 0},
                        {"min_identity": 1.5}, {"beta": -1.0}, {"predictor": "cnn"},
                        {"unknown": 1}):
            with self.assertRaises(ValueError):
                FoldSettings.from_toml(payload)

    def test_three_di_identity_requires_matching_lengths(self):
        with self.assertRaises(ValueError):
            three_di_identity("AAA", "AA")



def write_grid(directory: Path, sample: str = "", **asr) -> Path:
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
        """) + asr_lines + "\n" + sample, encoding="utf-8")
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


def write_stub(directory: Path) -> str:
    path = directory / "iqtree_stub"
    path.write_text(STUB_IQTREE.format(python=sys.executable), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


class IqtreeStageTests(unittest.TestCase):
    def stub(self, directory: Path) -> str:
        return write_stub(directory)

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


STATE_COLUMNS = "ARNDCQEGHILKMFPSTWYV"
SAMPLE_TOML = textwrap.dedent("""\
    [sample]
    structural_lambdas = [0, 0.5]
    nodes = ["Node1", "Node2"]
    n_iter = 6
    chains_per_node = 2
    checkpoint_every = 2
    """)


def write_state(path: Path, posteriors: dict) -> None:
    lines = ["# test posteriors", "Node\tSite\tState\t" + "\t".join("p_" + c for c in STATE_COLUMNS)]
    for node, matrix in posteriors.items():
        for site, row in enumerate(matrix, start=1):
            lines.append(f"{node}\t{site}\t{STATE_COLUMNS[int(row.argmax())]}\t"
                         + "\t".join(f"{value:.5f}" for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fake_3di(sequences):
    """Deterministic AA->3Di stand-in with local context dependence."""
    return ["".join(AA[zlib.crc32(sequence[max(0, i - 2):i + 3].encode()) % 20]
                    for i in range(len(sequence))) for sequence in sequences]


class GridSampleTests(unittest.TestCase):
    def build(self, base: Path):
        """Grid through ASR, with informative posteriors and a relabeled 3Di tree."""
        matrix = base / "Q.test"
        matrix.write_text(" ".join(["1"] * 210), encoding="utf-8")
        grid = grid_module.load_grid(write_grid(base, SAMPLE_TOML, three_di_matrix=str(matrix)))
        grid_module.run_forward(grid, batched, specs=grid.cells, max_batch=8)
        grid_module.run_encode(grid, fake_encoder, specs=grid.cells, shard_name="all",
                               batch_size=2, model_label="fake", model_info={}, resume=False)
        grid_module.run_asr(grid, specs=grid.cells, executable=write_stub(base), workers=2)
        length = len(grid.root_sequence)
        for spec in grid.cells:
            asr = grid.cell_dir(spec) / "asr"
            rng = np.random.default_rng(spec.index)
            aa = {f"Node{i}": rng.dirichlet(np.full(20, 0.2), size=length) for i in (1, 2, 3)}
            three_di = {f"Node{i}": rng.dirichlet(np.full(20, 0.2), size=length) for i in (1, 2, 3)}
            write_state(asr / "aa.state", aa)
            # The 3Di tree names AA Node2's clade Node3: correspondence must come from topology.
            (asr / "3di.treefile").write_text("(a,(b,c)Node3,(d,e)Node2)Node1;\n", encoding="utf-8")
            write_state(asr / "3di.state", {"Node1": three_di["Node1"], "Node3": three_di["Node2"],
                                            "Node2": three_di["Node3"]})
            # Standalone-sampler input: the same 3Di posteriors under AA names.
            write_state(asr / "3di_by_aa_name.state", three_di)
        return grid

    def standalone(self, grid, spec, structural_lambda, outdir: Path) -> None:
        from structure_asr import sampling

        asr = grid.cell_dir(spec) / "asr"
        nodes = outdir.parent / "nodes.txt"
        nodes.write_text("Node1\nNode2\n", encoding="utf-8")
        argv = ["sample", "--aa-state", str(asr / "aa.state"),
                "--three-di-state", str(asr / "3di_by_aa_name.state"),
                "--nodes-file", str(nodes), "--condition", "standalone", "--outdir", str(outdir),
                "--n-iter", "6", "--chains-per-node", "2", "--checkpoint-every", "2",
                "--structural-lambda", str(structural_lambda), "--seed", str(spec.seed)]
        with patch("sys.argv", argv), patch.object(
            sampling, "load_prostt5", return_value=(None, None, None, None, Path("/fake/model"))
        ), patch.object(sampling, "encode_sequences", lambda sequences, *_args: fake_3di(sequences)):
            sampling.main()

    def test_grid_sample_is_blind_resumable_and_matches_standalone_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            grid = self.build(base)
            hidden = {}
            for spec in grid.cells:
                cell = grid.cell_dir(spec)
                for path in (cell / "simulation" / "ground_truth_nodes.fasta",
                             cell / "truth" / "ground_truth_nodes_3di.fasta"):
                    hidden[path] = path.read_bytes()
                    path.unlink()
            calls = 0

            def interrupted(sequences):
                nonlocal calls
                calls += 1
                if calls == 5:
                    raise RuntimeError("simulated interruption")
                return fake_3di(sequences)

            options = dict(specs=grid.cells, model="fake", resolved_model="/fake/model",
                           shard_name="all")
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                grid_module.run_sample(grid, interrupted, resume=False, **options)
            with self.assertRaisesRegex(RuntimeError, "requires --resume"):
                grid_module.run_sample(grid, fake_3di, resume=False, **options)
            written = grid_module.run_sample(grid, fake_3di, resume=True, **options)
            self.assertEqual(len(written), 8)
            self.assertTrue(all(grid_module.sample_complete(grid, spec) for spec in grid.cells))
            self.assertEqual(grid_module.run_sample(grid, fake_3di, resume=True, **options), [])
            # 4 cells x 2 lambdas x 2 nodes x 2 chains start at AA-MAP: one encoding per cell node.
            with (grid.outdir / "sample_timings_all.tsv").open() as handle:
                first = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual((int(first["sequences"]), int(first["unique_sequences"])), (32, 8))

            for spec in grid.cells[:2]:
                for value in (0, 0.5):
                    run = grid_module.sample_dir(grid, spec, value)
                    config = json.loads((run / "run_manifest.json").read_text())["configuration"]
                    self.assertEqual(config["three_di_nodes"], {"Node1": "Node1", "Node2": "Node3"})
                    self.assertEqual(config["seed"], spec.seed)
                    reference = base / f"standalone_{spec.index}_{value}" / "run"
                    reference.parent.mkdir()
                    self.standalone(grid, spec, value, reference)
                    for name in ("trace.tsv", "chain_summary.tsv", "final_aa.fasta",
                                 "joint_asr_best_aa.fasta", "joint_asr_best_3di.fasta",
                                 "aa_map_start_3di.fasta", "site_proposal_weights.tsv"):
                        self.assertEqual((run / name).read_text(), (reference / name).read_text(),
                                         f"{spec.label} lambda={value} {name}")

            for path, content in hidden.items():
                path.write_bytes(content)
            grid_module.run_evaluate(grid, specs=grid.cells, outdir=base / "evaluation")
            with (base / "evaluation" / "sample_summary.tsv").open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 4 * 2 * 2)
            self.assertTrue(all(row["ground_truth_node"] and row["status"] == "complete" for row in rows))
            self.assertTrue(all(int(row["best_aa_errors"]) <= int(row["sites"]) for row in rows))

    def test_sample_settings_validation_and_plan(self):
        settings = grid_module.SampleSettings.from_toml({"structural_lambdas": [0.01], "nodes": "all"})
        self.assertIsNone(settings.nodes)
        self.assertEqual(settings.n_iter, 3000)
        for bad in ({"structural_lambdas": [0.01]},
                    {"structural_lambdas": [0.01, 0.01], "nodes": ["Node1"]},
                    {"structural_lambdas": [-1], "nodes": ["Node1"]},
                    {"structural_lambdas": [0.01], "nodes": []},
                    {"structural_lambdas": [0.01], "nodes": ["Node1"], "lambda": 2},
                    {"structural_lambdas": [0.01], "nodes": ["Node1"], "start_mode": "truth"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                grid_module.SampleSettings.from_toml(bad)
        with tempfile.TemporaryDirectory() as directory:
            grid = grid_module.load_grid(write_grid(Path(directory), SAMPLE_TOML.replace(
                'nodes = ["Node1", "Node2"]', 'nodes = "all"')))
            sample = grid_module.plan(grid)["sample"]
            self.assertEqual(sample["nodes_per_cell"], 3)
            self.assertEqual(sample["total_chains"], 4 * 2 * 3 * 2)
            self.assertEqual(sample["total_chain_steps"], 4 * 2 * 3 * 2 * 6)


class EvaluationMapStateTests(unittest.TestCase):
    def test_unlabelled_iqtree_state_falls_back_to_argmax(self):
        from collections import OrderedDict
        from structure_asr.evaluation import score_dataset

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "3di.state"
            alphabet = "ARNDCQEGHILKMFPSTWYV"

            def row(site, state, probabilities):
                values = [f"{probabilities.get(letter, 0.0):.5f}" for letter in alphabet]
                return "\t".join(["Node1", str(site), state, *values]) + "\n"

            path.write_text(
                "# IQ-TREE header\n"
                + "\t".join(["Node", "Site", "State", *(f"p_{letter}" for letter in alphabet)]) + "\n"
                + row(1, "D", {"A": 0.1, "D": 0.8, "P": 0.1})
                + row(2, "-", {"A": 0.2, "D": 0.3, "P": 0.5})
            )
            sites = StringIO()
            writer = csv.DictWriter(sites, fieldnames=None, extrasaction="ignore")
            writer.fieldnames = ["inferred_node", "site", "map_state", "truth_state", "map_correct"]
            mapping = [{"matched": True, "inferred_node": "Node1", "ground_truth_node": "Internal_0001"}]
            nodes, dataset = score_dataset(
                condition="c", tau=1.5, modality="3di", state_path=path,
                truth=OrderedDict(Internal_0001="DP"), mapping_rows=mapping, site_writer=writer,
            )
            self.assertEqual(dataset["map_accuracy"], 1.0)
            self.assertEqual(nodes[0]["map_accuracy"], 1.0)


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
