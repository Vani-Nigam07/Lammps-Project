"""Fixed workdir enforcement for LAMMPS runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

WORKDIR_ROOT = "/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/"


def get_workdir() -> str:
    """Return the fixed root workdir, ensuring it exists."""
    path = Path(WORKDIR_ROOT)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def safe_join(root: str, *parts: Iterable[str]) -> str:
    """Join paths and ensure the result stays under root."""
    root_path = Path(root).resolve()
    candidate = root_path.joinpath(*parts).resolve()
    if root_path not in candidate.parents and candidate != root_path:
        raise ValueError(f"Path escapes workdir root: {candidate}")
    return str(candidate)


def validate_filename(name: str) -> str:
    """Reject path separators; only allow basename to prevent traversal."""
    if name != os.path.basename(name):
        raise ValueError("Filename must be a basename without path separators")
    if name.strip() == "":
        raise ValueError("Filename must be non-empty")
    return name
