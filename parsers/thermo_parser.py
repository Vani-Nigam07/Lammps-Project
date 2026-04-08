"""Parse thermodynamic data from a LAMMPS log file."""

from __future__ import annotations

import sys
from typing import Dict, List


def parse_thermo(log_path: str) -> Dict[str, List[float]]:
    """Return thermo data as a dict of column -> list of floats."""
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
    except Exception as exc:
        print(f"[parse_thermo] Failed to read log: {exc}", file=sys.stderr)
        return {}

    thermo_data: List[List[float]] = []
    headers: List[str] = []
    in_thermo_section = False

    for line in lines:
        line = line.strip()
        if line.startswith("thermo_style"):
            parts = line.split()
            if len(parts) > 1:
                headers = parts[1:]

        if line.startswith("Step"):
            in_thermo_section = True
            if not headers:
                headers = line.split()
            continue

        if in_thermo_section and (line.startswith("Loop") or not line):
            in_thermo_section = False
            continue

        if in_thermo_section and line and not line.startswith("#"):
            try:
                values = [float(x) for x in line.split()]
                if len(values) == len(headers):
                    thermo_data.append(values)
            except ValueError:
                continue

    if not thermo_data or not headers:
        print(f"[parse_thermo] No thermo data found in {log_path}", file=sys.stderr)
        return {}

    cols: Dict[str, List[float]] = {h: [] for h in headers}
    for row in thermo_data:
        for h, v in zip(headers, row):
            cols[h].append(v)

    return cols
