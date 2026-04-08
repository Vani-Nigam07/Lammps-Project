"""Parsing and writing LAMMPS data files."""

from __future__ import annotations

import re
from typing import Dict, Any


def parse_lammps_data(filepath: str) -> Dict[str, Any]:
    """Parse a LAMMPS data file into structured sections."""
    with open(filepath) as f:
        raw = f.read()

    lines = raw.splitlines()
    header_comment = lines[0] if lines else ""

    # Parse counts from header
    counts: Dict[str, int] = {}
    for line in lines[1:20]:
        m = re.match(r"\s*(\d+)\s+(atoms|bonds|angles|atom types|bond types|angle types)", line)
        if m:
            counts[m.group(2)] = int(m.group(1))

    # Parse box bounds
    box: Dict[str, tuple[float, float]] = {}
    for line in lines[1:20]:
        for tag in ["xlo xhi", "ylo yhi", "zlo zhi"]:
            if tag in line:
                vals = line.split()
                box[tag] = (float(vals[0]), float(vals[1]))

    # Find section start lines
    section_starts: Dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in (
            "Masses",
            "Atoms",
            "Bonds",
            "Angles",
            "Velocities",
            "Dihedrals",
            "Impropers",
            "Pair Coeffs",
            "Bond Coeffs",
            "Angle Coeffs",
        ):
            section_starts[stripped] = i

    # Parse Masses
    masses: Dict[int, float] = {}
    if "Masses" in section_starts:
        idx = section_starts["Masses"] + 2  # skip blank line
        while idx < len(lines) and lines[idx].strip():
            parts = lines[idx].split()
            masses[int(parts[0])] = float(parts[1])
            idx += 1

    # Parse Atoms  (id mol type charge x y z)
    atoms = []
    if "Atoms" in section_starts:
        idx = section_starts["Atoms"] + 2
        while idx < len(lines) and lines[idx].strip():
            parts = lines[idx].split()
            if len(parts) >= 7:
                atoms.append(
                    {
                        "id": int(parts[0]),
                        "mol": int(parts[1]),
                        "type": int(parts[2]),
                        "charge": float(parts[3]),
                        "x": float(parts[4]),
                        "y": float(parts[5]),
                        "z": float(parts[6]),
                    }
                )
            idx += 1

    # Parse Bonds (id type a1 a2)
    bonds = []
    if "Bonds" in section_starts:
        idx = section_starts["Bonds"] + 2
        while idx < len(lines) and lines[idx].strip():
            parts = lines[idx].split()
            if len(parts) >= 4:
                bonds.append(
                    {
                        "id": int(parts[0]),
                        "type": int(parts[1]),
                        "a1": int(parts[2]),
                        "a2": int(parts[3]),
                    }
                )
            idx += 1

    # Parse Angles (id type a1 a2 a3)
    angles = []
    if "Angles" in section_starts:
        idx = section_starts["Angles"] + 2
        while idx < len(lines) and lines[idx].strip():
            parts = lines[idx].split()
            if len(parts) >= 5:
                angles.append(
                    {
                        "id": int(parts[0]),
                        "type": int(parts[1]),
                        "a1": int(parts[2]),
                        "a2": int(parts[3]),
                        "a3": int(parts[4]),
                    }
                )
            idx += 1

    return {
        "header": header_comment,
        "counts": counts,
        "box": box,
        "masses": masses,
        "atoms": atoms,
        "bonds": bonds,
        "angles": angles,
    }


def write_lammps_data(data: Dict[str, Any], header_comment: str | None = None) -> str:
    """Serialize parsed data back to LAMMPS data file format."""
    lines = []
    hdr = header_comment or data.get("header", "")
    lines.append(hdr)
    lines.append(f"{data['counts']['atoms']} atoms")
    lines.append(f"{data['counts']['bonds']} bonds")
    lines.append(f"{data['counts']['angles']} angles")
    lines.append("")
    lines.append(f"{data['counts'].get('atom types', 6)} atom types")
    lines.append(f"{data['counts'].get('bond types', 1)} bond types")
    lines.append(f"{data['counts'].get('angle types', 1)} angle types")
    lines.append("")
    for tag in ["xlo xhi", "ylo yhi", "zlo zhi"]:
        lo, hi = data["box"][tag]
        lines.append(f"{lo:f} {hi:f} {tag}")
    lines.append("")
    lines.append("Masses")
    lines.append("")
    for tid in sorted(data["masses"].keys()):
        lines.append(f"{tid} {data['masses'][tid]}")
    lines.append("")
    lines.append("Atoms")
    lines.append("")
    for a in data["atoms"]:
        lines.append(
            f"{a['id']:>8d} {a['mol']:>8d} {a['type']:>8d} "
            f"{a['charge']:>12.4f} {a['x']:>12.4f} {a['y']:>12.4f} {a['z']:>12.4f}"
        )
    lines.append("")
    lines.append("Bonds")
    lines.append("")
    for b in data["bonds"]:
        lines.append(f"{b['id']:>8d} {b['type']:>8d} {b['a1']:>8d} {b['a2']:>8d}")
    lines.append("")
    lines.append("Angles")
    lines.append("")
    for ang in data["angles"]:
        lines.append(
            f"{ang['id']:>8d} {ang['type']:>8d} {ang['a1']:>8d} {ang['a2']:>8d} {ang['a3']:>8d}"
        )
    return "\n".join(lines) + "\n"
