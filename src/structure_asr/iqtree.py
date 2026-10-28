"""IQ-TREE marginal ancestral reconstruction on a fixed topology.

Only tip alignments and a label-free, length-free topology are passed to
IQ-TREE; this module never opens ground-truth files. IQ-TREE re-estimates
branch lengths and model parameters on the fixed topology (``-te``) and
writes empirical-Bayes marginals (``--ancestral``). AA models are chosen by
ModelFinder (BIC by default). 3Di states are encoded as pseudo-amino-acids and
use a published 3Di exchangeability matrix supplied through ``-mdef``;
ModelFinder then chooses only frequency and rate-heterogeneity components.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from forward_asr_benchmark.io import read_fasta

ASR_OUTPUT_SUFFIXES = (".state", ".treefile", ".iqtree", ".log")
DEFAULT_3DI_MODEL_NAME = "Q3Di_AF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def topology_newick(tree) -> str:
    """Newick with tip names only: no internal labels, supports or lengths."""

    def render(clade) -> str:
        if clade.is_terminal():
            return str(clade.name)
        return "(" + ",".join(render(child) for child in clade.clades) + ")"

    return render(tree.root) + ";"


def write_topology(tree, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(topology_newick(tree) + "\n", encoding="utf-8")
    return path


def write_matrix_definition(matrix: Path, path: Path, name: str = DEFAULT_3DI_MODEL_NAME) -> Path:
    """Wrap a PAML-format 20-state matrix file as an IQ-TREE ``-mdef`` model."""
    values = matrix.read_text(encoding="utf-8").split()
    expected = 190 + 20
    if len(values) != expected:
        raise ValueError(f"{matrix} has {len(values)} numbers; expected {expected} (PAML 20-state)")
    for value in values:
        if not float(value) >= 0:
            raise ValueError(f"{matrix} contains an invalid rate or frequency: {value}")
    lines, index = [], 0
    for row in range(1, 20):
        lines.append(" ".join(values[index : index + row]))
        index += row
    lines.extend(["", " ".join(values[index:])])
    body = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#nexus\nbegin models;\nmodel {name} =\n{body};\nend;\n", encoding="utf-8"
    )
    return path


@dataclass(frozen=True)
class AsrJob:
    modality: str  # "aa" or "3di"
    alignment: Path
    topology: Path
    prefix: Path
    seed: int
    threads: int = 1
    model: str = "MFP"
    model_definition: Path | None = None
    model_set: str | None = None

    def command(self, iqtree: str) -> list[str]:
        command = [
            iqtree,
            "-s", str(self.alignment),
            "-st", "AA",
            "-te", str(self.topology),
            "-m", self.model,
            "--ancestral",
            "-T", str(self.threads),
            "--seed", str(self.seed),
            "--prefix", str(self.prefix),
        ]
        if self.model_definition is not None:
            command.extend(["-mdef", str(self.model_definition)])
        if self.model_set is not None:
            command.extend(["-mset", self.model_set])
        return command

    def outputs(self) -> dict[str, Path]:
        return {
            suffix.lstrip("."): Path(str(self.prefix) + suffix) for suffix in ASR_OUTPUT_SUFFIXES
        }

    def provenance(self, iqtree: str) -> dict[str, object]:
        record: dict[str, object] = {
            "modality": self.modality,
            "command": self.command(iqtree),
            "alignment_sha256": sha256(self.alignment),
            "topology_sha256": sha256(self.topology),
        }
        if self.model_definition is not None:
            record["model_definition_sha256"] = sha256(self.model_definition)
        return record


def validate_inputs(job: AsrJob) -> None:
    records = read_fasta(job.alignment)
    lengths = {len(sequence) for sequence in records.values()}
    if len(lengths) != 1:
        raise ValueError(f"{job.alignment} is not an equal-length alignment")
    tips = set(re.findall(r"[(,]([^(),;:]+)", job.topology.read_text(encoding="utf-8")))
    if tips != set(records):
        raise ValueError(
            f"{job.alignment} tips do not match topology tips "
            f"({len(set(records) - tips)} extra, {len(tips - set(records))} missing)"
        )


def iqtree_version(iqtree: str) -> str:
    completed = subprocess.run(
        [iqtree, "--version"], capture_output=True, text=True, check=True
    )
    for line in completed.stdout.splitlines():
        if "IQ-TREE" in line:
            return line.strip()
    return completed.stdout.strip().splitlines()[0]


def best_model(report: Path) -> str:
    text = report.read_text(encoding="utf-8", errors="replace")
    for pattern in (
        r"Best-fit model according to \w+:\s*(\S+)",
        r"Model of substitution:\s*(\S+)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    raise ValueError(f"no selected model found in {report}")


def manifest_path(job: AsrJob) -> Path:
    return Path(str(job.prefix) + ".asr_manifest.json")


def is_complete(job: AsrJob, iqtree: str) -> bool:
    """True for a finished, matching run; raise if finished outputs disagree."""
    path = manifest_path(job)
    if not path.exists():
        return False
    recorded = json.loads(path.read_text(encoding="utf-8"))
    current = job.provenance(iqtree)
    for key in ("alignment_sha256", "topology_sha256", "model_definition_sha256"):
        if recorded.get(key) != current.get(key):
            raise RuntimeError(f"{path} was produced from different inputs ({key}); use a new output directory")
    if recorded.get("command")[1:] != current["command"][1:]:
        raise RuntimeError(f"{path} was produced with different IQ-TREE options")
    return all(output.exists() for name, output in job.outputs().items() if name != "log")


def run_job(job: AsrJob, iqtree: str, version: str) -> dict[str, object]:
    """Run (or resume) one reconstruction; return its manifest."""
    if is_complete(job, iqtree):
        return json.loads(manifest_path(job).read_text(encoding="utf-8"))
    validate_inputs(job)
    job.prefix.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(job.command(iqtree), capture_output=True, text=True)
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-20:])
        raise RuntimeError(f"IQ-TREE failed for {job.prefix} ({job.modality}):\n{tail}")
    outputs = job.outputs()
    missing = [str(path) for name, path in outputs.items() if name != "log" and not path.exists()]
    if missing:
        raise RuntimeError(f"IQ-TREE finished without expected outputs: {missing}")
    manifest = {
        **job.provenance(iqtree),
        "iqtree_version": version,
        "best_fit_model": best_model(outputs["iqtree"]),
        "state_sha256": sha256(outputs["state"]),
        "outputs": {name: path.name for name, path in outputs.items()},
        "topology_policy": "fixed known topology (-te); branch lengths and model parameters re-estimated",
    }
    manifest_path(job).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
