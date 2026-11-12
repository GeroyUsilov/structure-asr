"""Tree loading and deterministic internal-node naming."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from Bio import Phylo


def load_tree(path: str | Path):
    tree = Phylo.read(str(path), "newick")
    seen: set[str] = set()
    internal_counter = 0
    for clade in tree.find_clades(order="preorder"):
        if not clade.name:
            if clade is tree.root:
                clade.name = "Root"
            else:
                internal_counter += 1
                clade.name = f"Internal_{internal_counter:04d}"
        if clade.name in seen:
            raise ValueError(f"duplicate tree node name: {clade.name!r}")
        seen.add(clade.name)
        if clade is not tree.root and clade.branch_length is None:
            raise ValueError(f"branch leading to {clade.name!r} has no length")
        if clade.branch_length is not None and clade.branch_length < 0:
            raise ValueError(f"branch leading to {clade.name!r} has negative length")
    return tree


def tree_to_newick(tree) -> str:
    handle = StringIO()
    Phylo.write(tree, handle, "newick")
    return handle.getvalue().strip()

