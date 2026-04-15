"""
Graphene / h-BN Pore Editor — Streamlit app for interactive pore design.

Reads a LAMMPS data file, reconstructs a full (no-pore) filter sheet from
the piston, lets you lasso/box/brush-select atoms for deletion, then writes
a new data file + input script with the chosen material (Graphene or h-BN).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re, copy, os, warnings
from collections import deque

# ── Material definitions ──────────────────────────────────────────────
#
# Internally the filter sheet always uses atom type 2 (graphene placeholder).
# At export time, types are remapped according to the chosen material.
#
# h-BN LJ parameters (ε kcal/mol, σ Å) provenance chain:
#   Primary source : Mayo, Olafson & Goddard, J. Phys. Chem. 1990, 94, 8897
#                    (DREIDING/A force field)
#   Adopted by     : Kang & Hwang, J. Phys.: Condens. Matter 2004, 16, 3901
#                    εB = 0.004116 eV = 0.0949 kcal/mol, σB = 3.453 Å
#                    εN = 0.006281 eV = 0.1448 kcal/mol, σN = 3.365 Å
#   Used in water  : Won & Aluru, J. Am. Chem. Soc. 2007, 129, 2748
#                    (SI Ref. S11; uncharged LJ model, SPC/E water)

MATERIALS = {
    "Graphene": {
        "filter_types": [2],
        "type_labels":  {2: "C"},
        "extra_masses": {},
        "extra_pairs":  {},
        "atom_types_total": 6,
        "neigh_exclude": "exclude type 1 1 exclude type 2 2",
        "pore_group_types": "2",
        "description": "Single-layer graphene (C, type 2)",
    },
    "h-BN": {
        # B → type 7, N → type 8  (appended after the existing 6 fluid types)
        "filter_types": [7, 8],
        "type_labels":  {7: "B", 8: "N"},
        "extra_masses": {7: 10.811, 8: 14.007},
        "extra_pairs":  {
            # DREIDING/A via Kang & Hwang 2004 via Won & Aluru 2007
            7: (0.0949, 3.453),   # B: ε = 0.3971 kJ/mol, σ = 3.453 Å
            8: (0.1448, 3.365),   # N: ε = 0.6060 kJ/mol, σ = 3.365 Å
        },
        "atom_types_total": 8,
        "neigh_exclude": (
            "exclude type 1 1 "
            "exclude type 7 7 "
            "exclude type 8 8 "
            "exclude type 7 8"
        ),
        "pore_group_types": "7 8",
        "description": "Hexagonal boron nitride (B type 7, N type 8)",
    },
    "MoS2": {
        # Mo → type 7, S → type 8  (mutually exclusive with h-BN in this app)
        "filter_types": [7, 8],
        "type_labels":  {7: "Mo", 8: "S"},
        "extra_masses": {7: 95.95, 8: 32.06},
        "extra_pairs":  {
            7: (0.0135, 4.2000),  # Mo–Mo
            8: (0.4612, 3.1300),  # S–S
        },
        "atom_types_total": 8,
        "neigh_exclude": (
            "exclude type 1 1 "
            "exclude type 7 7 "
            "exclude type 8 8 "
            "exclude type 7 8"
        ),
        "pore_group_types": "7 8",
        "description": "Monolayer MoS2 (Mo type 7, S type 8)",
    },
    "Ti2C MXene": {
        # Ti → type 7, C_carbide → type 8
        # Three-layer structure: Ti(top)–C(mid)–Ti(bot)
        # LJ parameters: UFF for Ti, Steele-1973 for C (see TI2C_* constants above)
        "filter_types": [7, 8],
        "type_labels":  {7: "Ti", 8: "C"},
        "extra_masses": {7: 47.867, 8: 12.011},
        "extra_pairs":  {
            7: (0.017,   2.8236),  # Ti–Ti  (UFF, Rappé 1992)
            8: (0.0556,  3.4000),  # C–C    (Steele 1973 / graphene)
        },
        "atom_types_total": 8,
        "neigh_exclude": (
            "exclude type 1 1 "
            "exclude type 7 7 "
            "exclude type 8 8 "
            "exclude type 7 8"
        ),
        "pore_group_types": "7 8",
        "description": "Ti₂C MXene — Ti(top/bot) type 7, C(carbide) type 8",
    },
}

# ── h-BN partial-charge models ────────────────────────────────────────
#
# The LAMMPS input already uses lj/cut/coul/long + kspace pppm, so any
# non-zero charge written to the data file is automatically handled.
#
# Three literature options:
#   0.00  — uncharged (Won & Aluru, JACS 2007; bare-edge LJ-only model)
#   ±0.30 — QMC/RPA charges (Wu, Wagner & Aluru, J. Chem. Phys. 2016, 144, 164118)
#            recommended for pores ≥7 Å where electrostatics affect ion rejection
#   ±0.37 — DREIDING-derived (used in several desalination MD papers, e.g. Wang et al.)
#            slightly higher polarization; conservative upper bound
#
BN_CHARGE_MODELS = {
    "0.00 e  — uncharged (Won & Aluru 2007)":           0.00,
    "±0.30 e — QMC/RPA (Wu, Wagner & Aluru JCP 2016)": 0.30,
    "±0.37 e — DREIDING-derived (desalination papers)": 0.37,
}

MOS2_CHARGE_MODELS = {
    "0.00 e  — uncharged (LJ-only)": (0.00, 0.00),
    "+0.50 / −0.25 e — MoSu-CHARMM (RPA-based)": (0.50, -0.25),
    "+0.76 / −0.38 e — MoS2–water FF (Morita/Varshney)": (0.76, -0.38),
    "+0.70 / −0.35 e — MoS2–water FF (Becker)": (0.70, -0.35),
}

MOS2_DEFAULT_A = 3.19
MOS2_DEFAULT_SS = 3.13  # S–S layer spacing (DFT)

# Target z-coordinate (Å) for the topmost atomic layer of trilayer sheets
# (top S of MoS2, top Ti of Ti2C).  The mid-plane is placed at
# TRILAYER_TOP_Z − half_thickness so the membrane is centred correctly
# between the two solvent regions.
TRILAYER_TOP_Z = 102.0

# ── Ti₂C MXene structural defaults ───────────────────────────────────────
#
# Lattice constant from DFT (Naguib et al., Adv. Mater. 2012; Khazaei et al.,
# Adv. Funct. Mater. 2013):
#   a = 3.07 Å  (in-plane hexagonal lattice constant)
#
# Ti plane z-offset from C mid-plane (DFT, 1T-octahedral structure):
#   z_Ti ≈ 1.04 Å  → total Ti–Ti slab thickness ≈ 2.08 Å
#
# Force field parameters — Lennard-Jones ε (kcal/mol), σ (Å):
#   Ti : UFF (Rappé et al., JACS 1992, Table 1)
#          x_i = 2.8236 Å used directly as σ, D_i = 0.017 kcal/mol
#          (common practice in MXene MD literature, e.g. Zhao et al.,
#           ACS Appl. Nano Mater. 2021, 4, 2, 1970–1978)
#   C  : graphite/Steele 1973 carbon — same parameters as graphene piston
#          ε = 0.0556 kcal/mol, σ = 3.400 Å
#          (used instead of UFF C_3 because carbide C interacts with water
#           similarly to graphene C in the absence of a bespoke Ti₂C FF)
#
# Partial charge models:
#   ±0.00 e  — uncharged baseline (LJ-only)
#   Ti +0.48 / C -0.96 — Bader charges from DFT-PBE (charge-neutral per formula
#                         unit: 2×0.48 − 0.96 = 0);
#                         Zhao et al. ACS Appl. Nano Mater. 2021 & related work
#   Ti +0.80 / C -1.60 — higher AIMD estimate (Osti et al., Chem. Mater. 2016)
#
TI2C_DEFAULT_A    = 3.07   # Å, in-plane lattice constant
TI2C_DEFAULT_TI_Z = 1.04   # Å, Ti plane offset above/below C mid-plane

TI2C_CHARGE_MODELS = {
    "0.00 / 0.00 e — uncharged (LJ-only)":                     (0.00,  0.00),
    "Ti +0.48 / C −0.96 e — Bader DFT (Zhao et al. 2021)":    (0.48, -0.96),
    "Ti +0.80 / C −1.60 e — AIMD estimate (Osti et al. 2016)": (0.80, -1.60),
}


# ── Parsing ───────────────────────────────────────────────────────────

def parse_lammps_data(filepath):
    with open(filepath) as f:
        raw = f.read()
    lines = raw.splitlines()
    header_comment = lines[0]

    counts = {}
    for line in lines[1:20]:
        m = re.match(r'\s*(\d+)\s+(atoms|bonds|angles|atom types|bond types|angle types)', line)
        if m:
            counts[m.group(2)] = int(m.group(1))

    box = {}
    for line in lines[1:20]:
        for tag in ['xlo xhi', 'ylo yhi', 'zlo zhi']:
            if tag in line:
                vals = line.split()
                box[tag] = (float(vals[0]), float(vals[1]))

    section_starts = {}
    for i, line in enumerate(lines):
        s = line.strip()
        if s in ('Masses', 'Atoms', 'Bonds', 'Angles'):
            section_starts[s] = i

    masses = {}
    if 'Masses' in section_starts:
        idx = section_starts['Masses'] + 2
        while idx < len(lines) and lines[idx].strip():
            p = lines[idx].split()
            masses[int(p[0])] = float(p[1])
            idx += 1

    atoms = []
    if 'Atoms' in section_starts:
        idx = section_starts['Atoms'] + 2
        while idx < len(lines) and lines[idx].strip():
            p = lines[idx].split()
            if len(p) >= 7:
                atoms.append({'id': int(p[0]), 'mol': int(p[1]), 'type': int(p[2]),
                               'charge': float(p[3]), 'x': float(p[4]),
                               'y': float(p[5]), 'z': float(p[6])})
            idx += 1

    bonds = []
    if 'Bonds' in section_starts:
        idx = section_starts['Bonds'] + 2
        while idx < len(lines) and lines[idx].strip():
            p = lines[idx].split()
            if len(p) >= 4:
                bonds.append({'id': int(p[0]), 'type': int(p[1]),
                               'a1': int(p[2]), 'a2': int(p[3])})
            idx += 1

    angles = []
    if 'Angles' in section_starts:
        idx = section_starts['Angles'] + 2
        while idx < len(lines) and lines[idx].strip():
            p = lines[idx].split()
            if len(p) >= 5:
                angles.append({'id': int(p[0]), 'type': int(p[1]),
                                'a1': int(p[2]), 'a2': int(p[3]), 'a3': int(p[4])})
            idx += 1

    return {'header': header_comment, 'counts': counts, 'box': box,
            'masses': masses, 'atoms': atoms, 'bonds': bonds, 'angles': angles}


# ── ASE-based sheet builder ───────────────────────────────────────────
#
# Replaces the old piston-clone approach.  The primitive h-BN / graphene
# unit cell is repeated to fill the orthorhombic simulation box, then atom
# x/y coordinates are folded back into [xlo, xhi] × [ylo, yhi] via the
# lattice periodicity.  Because (nx·a, 0) is an exact lattice vector of
# the hexagonal cell, this wrapping is lossless — no duplicate positions.

def build_ase_filter_sheet(box, filter_z, a=2.50):
    """
    Build a 2-D hexagonal filter sheet with correct atomic coordinates.

    Parameters
    ----------
    box      : dict from parse_lammps_data  ('xlo xhi', 'ylo yhi', …)
    filter_z : z-coordinate (Å) to assign to all sheet atoms
    a        : lattice constant (Å).  2.50 for h-BN, 2.46 for graphene.

    Returns
    -------
    list of atom dicts, type=2, charge=0, IDs starting at 1 (temporary).
    Even-indexed atoms (0, 2, 4, …) are sublattice-A (→ B in h-BN).
    Odd-indexed atoms  (1, 3, 5, …) are sublattice-B (→ N in h-BN).
    """
    from ase import Atoms as AseAtoms

    lx  = box['xlo xhi'][1] - box['xlo xhi'][0]
    ly  = box['ylo yhi'][1] - box['ylo yhi'][0]
    xlo = box['xlo xhi'][0]
    ylo = box['ylo yhi'][0]

    # Rectangular 4-atom unit cell — no oblique shear, no wrapping needed.
    #
    # This is two primitive 2-atom cells stacked along y:
    #   B at (0,       0           )   N at (a/2, √3a/6)
    #   B at (a/2,     √3a/2       )   N at (0,   √3a·2/3)
    #
    # Cell dimensions:  ax = a,  ay = √3·a
    # Sublattice pattern: indices 0,2,4,… → B;  1,3,5,… → N  (i % 2)
    ay_rect = np.sqrt(3) * a          # y-period of rectangular cell

    uc = AseAtoms(
        'BNBN',
        positions=[
            (0,       0,                      0),
            (a / 2,   np.sqrt(3) * a / 6,     0),
            (a / 2,   np.sqrt(3) * a / 2,     0),
            (0,       np.sqrt(3) * a * 2 / 3, 0),
        ],
        cell=[[a, 0, 0], [0, ay_rect, 0], [0, 0, 20.0]],
        pbc=[True, True, False],
    )

    # Repeat slightly beyond the box, then crop. This avoids lattice strain.
    nx = max(1, int(np.ceil(lx / a)) + 1)
    ny = max(1, int(np.ceil(ly / ay_rect)) + 1)
    sheet = uc.repeat((nx, ny, 1))

    pos = sheet.get_positions()         # shape (4·nx·ny, 3)
    pos[:, 0] = pos[:, 0] + xlo
    pos[:, 1] = pos[:, 1] + ylo

    # Crop to the box bounds. Use half-open interval to avoid duplicate
    # atoms on periodic boundaries (x==xhi or y==yhi).
    eps = 1e-6
    mask = (
        (pos[:, 0] >= xlo - eps) & (pos[:, 0] < xlo + lx - eps) &
        (pos[:, 1] >= ylo - eps) & (pos[:, 1] < ylo + ly - eps)
    )
    pos = pos[mask]

    atom_list = []
    for idx, p in enumerate(pos):
        atom_list.append({
            'id':     idx + 1,       # temporary; caller reassigns
            'mol':    0,
            'type':   2,             # internal filter placeholder
            'charge': 0.0,
            'x': float(p[0]),
            'y': float(p[1]),
            'z': float(filter_z),
        })
    return atom_list


def build_ase_mos2_sheet(box, filter_z, a=3.19, ss=3.13):
    """
    Build a MoS2 monolayer (S–Mo–S) sheet using a rectangular cell.

    Parameters
    ----------
    box      : dict from parse_lammps_data  ('xlo xhi', 'ylo yhi', …)
    filter_z : z-coordinate (Å) of the Mo plane
    a        : lattice constant (Å)
    ss       : S–S vertical spacing (Å) between the two sulfur planes

    Returns
    -------
    list of atom dicts, type=2, charge=0, IDs starting at 1 (temporary).
    """
    from ase import Atoms as AseAtoms

    lx  = box['xlo xhi'][1] - box['xlo xhi'][0]
    ly  = box['ylo yhi'][1] - box['ylo yhi'][0]
    xlo = box['xlo xhi'][0]
    ylo = box['ylo yhi'][0]

    ay_rect = np.sqrt(3) * a
    z0 = ss / 2.0

    # Rectangular cell containing two MoS2 formula units (2 Mo, 4 S).
    mo_pos = [
        (0.0,     0.0,        0.0),
        (a / 2.0, ay_rect / 2.0, 0.0),
    ]

    s_offsets = [
        (2.0 * a / 3.0, ay_rect / 3.0,  +z0),
        (5.0 * a / 6.0, ay_rect / 6.0,  -z0),
    ]

    positions = []
    symbols = []
    for mx, my, _ in mo_pos:
        positions.append((mx, my, 0.0))
        symbols.append("Mo")
        for ox, oy, oz in s_offsets:
            x = (mx + ox) % a
            y = (my + oy) % ay_rect
            positions.append((x, y, oz))
            symbols.append("S")

    uc = AseAtoms(
        symbols,
        positions=positions,
        cell=[[a, 0, 0], [0, ay_rect, 0], [0, 0, 20.0]],
        pbc=[True, True, False],
    )

    # Repeat slightly beyond the box, then crop. This avoids lattice strain.
    nx = max(1, int(np.ceil(lx / a)) + 1)
    ny = max(1, int(np.ceil(ly / ay_rect)) + 1)
    sheet = uc.repeat((nx, ny, 1))

    pos = sheet.get_positions()         # shape (n, 3)
    pos[:, 0] = pos[:, 0] + xlo
    pos[:, 1] = pos[:, 1] + ylo
    pos[:, 2] = pos[:, 2] + filter_z

    # Crop to the box bounds. Use half-open interval to avoid duplicates.
    eps = 1e-6
    mask = (
        (pos[:, 0] >= xlo - eps) & (pos[:, 0] < xlo + lx - eps) &
        (pos[:, 1] >= ylo - eps) & (pos[:, 1] < ylo + ly - eps)
    )
    pos = pos[mask]

    atom_list = []
    for idx, p in enumerate(pos):
        atom_list.append({
            'id':     idx + 1,       # temporary; caller reassigns
            'mol':    0,
            'type':   2,             # internal filter placeholder
            'charge': 0.0,
            'x': float(p[0]),
            'y': float(p[1]),
            'z': float(p[2]),
        })
    return atom_list


def build_ase_ti2c_sheet(box, filter_z, a=3.07, ti_z=1.04):
    """
    Build a Ti₂C MXene monolayer (Ti–C–Ti) using a rectangular supercell.

    The structure is 1T-octahedral (P3̄m1), the ground-state polymorph from DFT.
    C atoms form the central hexagonal layer; Ti atoms sit in the triangular
    hollows above and below, at two distinct in-plane positions per formula unit
    (same offset logic as build_ase_mos2_sheet).

    Parameters
    ----------
    box      : dict from parse_lammps_data  ('xlo xhi', 'ylo yhi', …)
    filter_z : z-coordinate (Å) of the C mid-plane
    a        : in-plane lattice constant (Å), default 3.07 (DFT PBE)
    ti_z     : vertical offset (Å) of each Ti plane from C mid-plane, default 1.04

    Returns
    -------
    list of atom dicts — all assigned type=2 (internal filter placeholder).
    Z-coordinate encodes layer identity:
      filter_z          → C  (mid-plane)
      filter_z ± ti_z   → Ti (top / bottom planes)
    apply_material() uses the z-position to assign the correct LAMMPS types.
    """
    from ase import Atoms as AseAtoms

    lx  = box['xlo xhi'][1] - box['xlo xhi'][0]
    ly  = box['ylo yhi'][1] - box['ylo yhi'][0]
    xlo = box['xlo xhi'][0]
    ylo = box['ylo yhi'][0]

    ay_rect = np.sqrt(3) * a   # y-period of the rectangular 4-atom C cell

    # ── Rectangular unit cell with 2 formula units (2C + 2Ti_top + 2Ti_bot) ──
    #
    # C sublattice (same as h-BN / graphene rectangular cell, 2 atoms):
    #   C_1 at (0,         0,          0)
    #   C_2 at (a/2,       ay_rect/2,  0)
    #
    # Ti_top sublattice (offset into triangular hollow above C plane):
    #   Ti_top_1 at (2a/3,    ay_rect/3,  +ti_z)
    #   Ti_top_2 at (a/6,     5ay_rect/6, +ti_z)   ← = Ti_top_1 + (a/2, ay/2) mod cell
    #
    # Ti_bot sublattice (opposite hollow site below C plane):
    #   Ti_bot_1 at (5a/6,    ay_rect/6,  -ti_z)
    #   Ti_bot_2 at (a/3,     2ay_rect/3, -ti_z)   ← = Ti_bot_1 + (a/2, ay/2) mod cell
    #
    # This matches the 1T stacking of Ti₂C (Naguib et al. 2012; Khazaei et al. 2013).
    ay = ay_rect
    positions = [
        # C mid-plane
        (0.0,          0.0,       0.0),
        (a / 2.0,      ay / 2.0,  0.0),
        # Ti top plane
        (2.0*a/3.0,    ay / 3.0,  +ti_z),
        (a / 6.0,      5.0*ay/6.0, +ti_z),
        # Ti bottom plane
        (5.0*a/6.0,    ay / 6.0,  -ti_z),
        (a / 3.0,      2.0*ay/3.0, -ti_z),
    ]
    symbols = ['C', 'C', 'Ti', 'Ti', 'Ti', 'Ti']

    uc = AseAtoms(
        symbols,
        positions=positions,
        cell=[[a, 0, 0], [0, ay, 0], [0, 0, 20.0]],
        pbc=[True, True, False],
    )

    # Repeat beyond box, then crop — same strategy as MoS₂ builder
    nx = max(1, int(np.ceil(lx / a)) + 1)
    ny = max(1, int(np.ceil(ly / ay)) + 1)
    sheet = uc.repeat((nx, ny, 1))

    pos = sheet.get_positions()
    pos[:, 0] += xlo
    pos[:, 1] += ylo
    pos[:, 2] += filter_z   # shift so C mid-plane lands at filter_z

    eps = 1e-6
    mask = (
        (pos[:, 0] >= xlo - eps) & (pos[:, 0] < xlo + lx - eps) &
        (pos[:, 1] >= ylo - eps) & (pos[:, 1] < ylo + ly - eps)
    )
    pos = pos[mask]

    atom_list = []
    for idx, p in enumerate(pos):
        atom_list.append({
            'id':     idx + 1,
            'mol':    0,
            'type':   2,       # internal placeholder; apply_material() retypes
            'charge': 0.0,
            'x': float(p[0]),
            'y': float(p[1]),
            'z': float(p[2]),
        })
    return atom_list


def reconstruct_filter_with_ase(data, a=2.50, commensurate_box=False, sheet_kind="graphene",
                                sheet_z_shift=0.0):
    """
    Replace type-2 (filter) atoms in *data* with an ASE-generated hexagonal
    sheet that has geometrically correct positions for lattice constant *a*.

    Bonds and angles in the LAMMPS data reference only fluid atoms (types
    3-6) so they are remapped from the non-filter subset only — same logic
    as the original piston-clone approach.

    sheet_z_shift : float, Å
        Rigid shift added to filter_z before building the sheet.  For trilayer
        materials (MoS2, Ti2C) the outer atomic planes extend above and below
        the mid-plane, so a positive shift moves the whole sheet up and
        prevents the lower surface from overlapping with solvent atoms.
    """
    working = copy.deepcopy(data)

    if commensurate_box:
        xlo, xhi = working['box']['xlo xhi']
        ylo, yhi = working['box']['ylo yhi']
        lx = xhi - xlo
        ly = yhi - ylo
        ay_rect = np.sqrt(3) * a
        nx = max(1, int(np.ceil(lx / a)))
        ny = max(1, int(np.ceil(ly / ay_rect)))
        new_lx = nx * a
        new_ly = ny * ay_rect
        working['box']['xlo xhi'] = (xlo, xlo + new_lx)
        working['box']['ylo yhi'] = (ylo, ylo + new_ly)

    non_filter  = [at for at in working['atoms'] if at['type'] != 2]
    existing_t2 = [at for at in working['atoms'] if at['type'] == 2]
    filter_z    = (existing_t2[0]['z'] if existing_t2 else 96.5) + sheet_z_shift

    if sheet_z_shift != 0.0:
        print(f"[reconstruct] sheet_z_shift = {sheet_z_shift:+.3f} Å  →  filter_z = {filter_z:.3f} Å")

    if sheet_kind == "mos2":
        new_filter = build_ase_mos2_sheet(working['box'], filter_z, a=a, ss=MOS2_DEFAULT_SS)
    elif sheet_kind == "ti2c":
        new_filter = build_ase_ti2c_sheet(working['box'], filter_z, a=a,
                                          ti_z=TI2C_DEFAULT_TI_Z)
    else:
        new_filter = build_ase_filter_sheet(working['box'], filter_z, a=a)

    return replace_filter_sheet(working, non_filter, new_filter)


def replace_piston_sheet(working, a=2.46):
    """
    Rebuild the piston (type 1) sheet using the same commensurate box.
    Uses graphene lattice constant by default.
    """
    piston_atoms = [at for at in working['atoms'] if at['type'] == 1]
    if not piston_atoms:
        return working
    piston_z = piston_atoms[0]['z']
    new_piston = build_ase_filter_sheet(working['box'], piston_z, a=a)
    # Keep only non-piston, then append new piston with type=1
    non_piston = [at for at in working['atoms'] if at['type'] != 1]
    # Retype to piston type
    for p in new_piston:
        p['type'] = 1
    return replace_filter_sheet(working, non_piston, new_piston)


def replace_filter_sheet(data, non_filter, new_filter):

    # ── Geometry sanity check ─────────────────────────────────────────────
    _xs = np.array([at['x'] for at in new_filter])
    _ys = np.array([at['y'] for at in new_filter])
    _xlo, _xhi = data['box']['xlo xhi']
    _ylo, _yhi = data['box']['ylo yhi']
    _out = int(np.sum((_xs < _xlo) | (_xs > _xhi) | (_ys < _ylo) | (_ys > _yhi)))
    # Check ALL atoms by computing each atom's nearest neighbour in the sheet.
    # Vectorised slab-scan: compare every atom against its x-neighbours only.
    _nn = np.full(len(new_filter), np.inf)
    _order = np.argsort(_xs)
    for _ii, _i in enumerate(_order):
        for _jj in range(_ii + 1, len(new_filter)):
            _j = _order[_jj]
            if _xs[_j] - _xs[_i] > 1.6:
                break
            _dy = abs(_ys[_i] - _ys[_j])
            if _dy > 1.6:
                continue
            _d = np.sqrt((_xs[_i]-_xs[_j])**2 + _dy**2)
            if _d < _nn[_i]: _nn[_i] = _d
            if _d < _nn[_j]: _nn[_j] = _d
    print(f"[filter sheet] {len(new_filter)} atoms")
    print(f"  x range : [{_xs.min():.3f}, {_xs.max():.3f}]  box x: [{_xlo:.3f}, {_xhi:.3f}]  gap={_xhi-_xs.max():.4f} Å")
    print(f"  y range : [{_ys.min():.3f}, {_ys.max():.3f}]  box y: [{_ylo:.3f}, {_yhi:.3f}]  gap={_yhi-_ys.max():.4f} Å")
    print(f"  atoms outside box : {_out}")
    _nn_bad = np.sum(_nn > 1.6)   # atoms with no neighbour found (isolated)
    print(f"  NN dist  min/max  : {_nn.min():.4f} / {_nn.max():.4f} Å")
    print(f"  atoms with no NN < 1.6 Å : {_nn_bad}  (should be 0)")
    if _nn_bad > 0:
        bad_idx = np.where(_nn > 1.6)[0]
        for _bi in bad_idx[:5]:
            print(f"    atom idx={_bi}  x={_xs[_bi]:.3f}  y={_ys[_bi]:.3f}  NN={_nn[_bi]:.4f}")
    # ─────────────────────────────────────────────────────────────────────

    # Assign IDs continuing from the highest non-filter ID
    next_id = max(at['id'] for at in non_filter) + 1
    for fa in new_filter:
        fa['id'] = next_id
        next_id += 1

    all_atoms = non_filter + new_filter

    # Save original non-filter IDs BEFORE the sequential re-ID mutates them.
    # (Previously id_map was built here from the same atom objects, making it an
    # identity map that broke remapping whenever original IDs were not 1…M.)
    _orig_non_filter_ids = [at['id'] for at in non_filter]

    # Sequential re-ID across the full system
    for i, at in enumerate(all_atoms, start=1):
        at['id'] = i

    # Build the correct map: original_id → new sequential id
    id_map = {orig: j + 1 for j, orig in enumerate(_orig_non_filter_ids)}
    print(f"[id_map] {len(id_map)} non-filter atoms remapped; "
          f"ID range {min(id_map)}-{max(id_map)} → 1-{len(non_filter)}")

    def remap_bonds(bs):
        out = []
        for b in bs:
            if b['a1'] in id_map and b['a2'] in id_map:
                out.append({'id': len(out) + 1, 'type': b['type'],
                            'a1': id_map[b['a1']], 'a2': id_map[b['a2']]})
        return out

    def remap_angles(angs):
        out = []
        for ang in angs:
            if all(ang[k] in id_map for k in ('a1', 'a2', 'a3')):
                out.append({'id': len(out) + 1, 'type': ang['type'],
                            'a1': id_map[ang['a1']], 'a2': id_map[ang['a2']],
                            'a3': id_map[ang['a3']]})
        return out

    nd = copy.deepcopy(data)
    nd['atoms']           = all_atoms
    nd['bonds']           = remap_bonds(data['bonds'])
    nd['angles']          = remap_angles(data['angles'])
    nd['counts']['atoms'] = len(all_atoms)
    nd['counts']['bonds'] = len(nd['bonds'])
    nd['counts']['angles'] = len(nd['angles'])
    return nd


# ── BFS sublattice coloring ───────────────────────────────────────────

def assign_bn_sublattices(filter_atoms, box=None):
    """
    BFS graph-color the hexagonal lattice into two sublattices.
    Returns dict {atom_id: 0} for sublattice-A (→ B in h-BN)
                 {atom_id: 1} for sublattice-B (→ N in h-BN).

    Two atoms are neighbours if their 2-D distance < 1.55 Å (C-C bond ~1.42 Å).

    Parameters
    ----------
    box : optional dict from parse_lammps_data.  When supplied, atoms near the
          periodic boundaries are also tested against their images on the opposite
          side so that edge atoms are coloured correctly.
    """
    n = len(filter_atoms)
    xs = np.array([a['x'] for a in filter_atoms])
    ys = np.array([a['y'] for a in filter_atoms])
    ids = [a['id'] for a in filter_atoms]

    # Build neighbour list with a fast distance check
    CUTOFF = 1.55
    neighbors = {i: [] for i in range(n)}
    # Sort by x for a narrow band scan (faster than O(n²) for a sheet)
    order = np.argsort(xs)
    for ii, i in enumerate(order):
        for jj in range(ii + 1, n):
            j = order[jj]
            if xs[j] - xs[i] > CUTOFF:
                break
            dy = abs(ys[i] - ys[j])
            if dy > CUTOFF:
                continue
            dx = xs[i] - xs[j]
            if dx*dx + dy*dy <= CUTOFF*CUTOFF:
                neighbors[i].append(j)
                neighbors[j].append(i)

    # ── Periodic boundary neighbours ─────────────────────────────────────
    if box is not None:
        lx = box['xlo xhi'][1] - box['xlo xhi'][0]
        ly = box['ylo yhi'][1] - box['ylo yhi'][0]
        xlo_b, xhi_b = box['xlo xhi']
        ylo_b, yhi_b = box['ylo yhi']
        # Atoms within CUTOFF of a boundary can have neighbours across it.
        near_xlo = np.where(xs - xlo_b < CUTOFF)[0]
        near_xhi = np.where(xhi_b - xs < CUTOFF)[0]
        near_ylo = np.where(ys - ylo_b < CUTOFF)[0]
        near_yhi = np.where(yhi_b - ys < CUTOFF)[0]
        # Check each (boundary-hi, boundary-lo) pair with the periodic shift.
        for (hi_set, lo_set, dx_off, dy_off) in [
            (near_xhi, near_xlo,  -lx,   0),
            (near_yhi, near_ylo,    0, -ly),
            (near_xhi, near_ylo,  -lx,  ly),   # corners
            (near_yhi, near_xlo,   lx, -ly),
        ]:
            for i in hi_set:
                for j in lo_set:
                    if i == j:
                        continue
                    dx = xs[i] - xs[j] + dx_off
                    dy = ys[i] - ys[j] + dy_off
                    if abs(dx) <= CUTOFF and abs(dy) <= CUTOFF and dx*dx + dy*dy <= CUTOFF*CUTOFF:
                        if j not in neighbors[i]:
                            neighbors[i].append(j)
                            neighbors[j].append(i)

    # ── BFS colouring ────────────────────────────────────────────────────
    sublattice = [-1] * n
    sublattice[0] = 0
    queue = deque([0])
    while queue:
        i = queue.popleft()
        for j in neighbors[i]:
            if sublattice[j] == -1:
                sublattice[j] = 1 - sublattice[i]
                queue.append(j)

    uncoloured = sublattice.count(-1)
    if uncoloured:
        print(f"[sublattice] WARNING: {uncoloured}/{n} atoms uncoloured "
              f"(disconnected graph — check sheet connectivity)")
    print(f"[sublattice] A(→B): {sublattice.count(0)}  B(→N): {sublattice.count(1)}  "
          f"uncoloured: {uncoloured}")

    return {ids[i]: sublattice[i] for i in range(n)}


# ── Deletion ──────────────────────────────────────────────────────────

def delete_atoms_and_rewrite(data, ids_to_delete):
    # Keep a snapshot of positions by original ID for verification.
    orig_pos = {a['id']: (a['x'], a['y'], a['z']) for a in data['atoms']}
    remove_set = set(ids_to_delete)
    kept = [a for a in data['atoms'] if a['id'] not in remove_set]

    id_map = {}
    for i, a in enumerate(kept, start=1):
        id_map[a['id']] = i
        a['id'] = i

    kept_bonds = []
    for b in data['bonds']:
        if b['a1'] in id_map and b['a2'] in id_map:
            kept_bonds.append({'id': len(kept_bonds)+1, 'type': b['type'],
                                'a1': id_map[b['a1']], 'a2': id_map[b['a2']]})

    kept_angles = []
    for ang in data['angles']:
        if all(ang[k] in id_map for k in ('a1','a2','a3')):
            kept_angles.append({'id': len(kept_angles)+1, 'type': ang['type'],
                                 'a1': id_map[ang['a1']], 'a2': id_map[ang['a2']],
                                 'a3': id_map[ang['a3']]})

    nd = copy.deepcopy(data)
    nd['atoms']  = kept
    nd['bonds']  = kept_bonds
    nd['angles'] = kept_angles
    nd['counts']['atoms']  = len(kept)
    nd['counts']['bonds']  = len(kept_bonds)
    nd['counts']['angles'] = len(kept_angles)

    # Verify positions of kept atoms are unchanged (by original ID).
    max_disp = 0.0
    for a in kept:
        orig_id = next((k for k, v in id_map.items() if v == a['id']), None)
        if orig_id is None or orig_id not in orig_pos:
            continue
        ox, oy, oz = orig_pos[orig_id]
        dx = a['x'] - ox
        dy = a['y'] - oy
        dz = a['z'] - oz
        d = np.sqrt(dx*dx + dy*dy + dz*dz)
        if d > max_disp:
            max_disp = d
    print(f"[delete] kept atoms: {len(kept)}  removed: {len(remove_set)}  max position delta: {max_disp:.6e} Å")
    return nd, id_map


# ── Material conversion (applied to export copy only) ─────────────────

def apply_material(data, material_name, sublattice_map=None, bn_charge=0.0,
                   mos2_charges=(0.0, 0.0), ti2c_charges=(0.0, 0.0)):
    """
    Convert type-2 filter atoms to the correct types for the chosen material.
    For h-BN, sublattice_map {atom_id: 0/1} must be provided;
      sublattice 0 → B (type 7, charge = +bn_charge)
      sublattice 1 → N (type 8, charge = -bn_charge)
    bn_charge=0.0 gives the uncharged (LJ-only) model.
    Returns a new data dict ready for writing.
    """
    mat = MATERIALS[material_name]
    nd  = copy.deepcopy(data)

    if material_name == "Graphene":
        # No type changes needed; just confirm atom types count
        nd['counts']['atom types'] = 6
        return nd

    if material_name == "MoS2":
        mo_q, s_q = mos2_charges
        # Infer Mo plane from the median of unique z-values (Mo plane lies between S planes).
        fz = [a['z'] for a in nd['atoms'] if a['type'] == 2]
        if fz:
            uniq = sorted(set(round(z, 3) for z in fz))
            if len(uniq) >= 3:
                mo_z = uniq[len(uniq) // 2]
            else:
                mo_z = float(np.median(fz))
        else:
            mo_z = 0.0

        for a in nd['atoms']:
            if a['type'] == 2:
                if abs(a['z'] - mo_z) < 1e-2:
                    a['type'] = 7   # Mo
                    a['charge'] = mo_q
                else:
                    a['type'] = 8   # S
                    a['charge'] = s_q

        # Add masses for new types
        for tid, mass in mat['extra_masses'].items():
            nd['masses'][tid] = mass

        nd['counts']['atom types'] = mat['atom_types_total']
        return nd

    if material_name == "Ti2C MXene":
        ti_q, c_q = ti2c_charges
        # The C mid-plane is at the median z of all filter atoms.
        # Ti planes are above and below by TI2C_DEFAULT_TI_Z.
        # We identify C atoms as those closest to the median z.
        fz = [a['z'] for a in nd['atoms'] if a['type'] == 2]
        if fz:
            c_z = float(np.median(fz))
        else:
            c_z = 0.0
        # Tolerance: atoms within 0.3 Å of the median z are the C mid-plane.
        # (Ti planes are ≈1.04 Å away, so 0.3 Å is a safe discriminator.)
        C_PLANE_TOL = 0.3

        for a in nd['atoms']:
            if a['type'] == 2:
                if abs(a['z'] - c_z) < C_PLANE_TOL:
                    a['type'] = 8   # C carbide (mid-plane)
                    a['charge'] = c_q
                else:
                    a['type'] = 7   # Ti (top or bottom plane)
                    a['charge'] = ti_q

        for tid, mass in mat['extra_masses'].items():
            nd['masses'][tid] = mass

        nd['counts']['atom types'] = mat['atom_types_total']
        return nd

    # h-BN: retype each filter atom based on its sublattice
    for a in nd['atoms']:
        if a['type'] == 2:
            sl = sublattice_map.get(a['id'], 0)
            if sl == 0:
                a['type']   = 7          # B
                a['charge'] = +bn_charge
            else:
                a['type']   = 8          # N
                a['charge'] = -bn_charge

    # Add masses for new types
    for tid, mass in mat['extra_masses'].items():
        nd['masses'][tid] = mass

    nd['counts']['atom types'] = mat['atom_types_total']
    return nd


# ── Writing ───────────────────────────────────────────────────────────

def write_lammps_data(data, header_comment=None):
    lines = []
    lines.append(header_comment or data['header'])
    lines.append(f"{data['counts']['atoms']} atoms")
    lines.append(f"{data['counts']['bonds']} bonds")
    lines.append(f"{data['counts']['angles']} angles")
    lines.append("")
    lines.append(f"{data['counts'].get('atom types', 6)} atom types")
    lines.append(f"{data['counts'].get('bond types', 1)} bond types")
    lines.append(f"{data['counts'].get('angle types', 1)} angle types")
    lines.append("")
    for tag in ['xlo xhi', 'ylo yhi', 'zlo zhi']:
        lo, hi = data['box'][tag]
        lines.append(f"{lo:f} {hi:f} {tag}")
    lines.append("")
    lines.append("Masses")
    lines.append("")
    for tid in sorted(data['masses'].keys()):
        lines.append(f"{tid} {data['masses'][tid]}")
    lines.append("")
    lines.append("Atoms")
    lines.append("")
    for a in data['atoms']:
        lines.append(f"{a['id']:>8d} {a['mol']:>8d} {a['type']:>8d} "
                     f"{a['charge']:>12.4f} {a['x']:>12.4f} {a['y']:>12.4f} {a['z']:>12.4f}")
    lines.append("")
    lines.append("Bonds")
    lines.append("")
    for b in data['bonds']:
        lines.append(f"{b['id']:>8d} {b['type']:>8d} {b['a1']:>8d} {b['a2']:>8d}")
    lines.append("")
    lines.append("Angles")
    lines.append("")
    for ang in data['angles']:
        lines.append(f"{ang['id']:>8d} {ang['type']:>8d} "
                     f"{ang['a1']:>8d} {ang['a2']:>8d} {ang['a3']:>8d}")
    return "\n".join(lines) + "\n"


# ── PDB export ───────────────────────────────────────────────────────

def export_to_pdb(data, material_name, output_path):
    """
    Write the system to PDB format using MDAnalysis.

    Residue scheme:
      - Membrane atoms (mol=0): one residue per atom type
          GRP = graphene piston (type 1)
          GRF = graphene filter (type 2)
          BNF = h-BN filter     (types 7/8, grouped together)
      - Fluid molecules (mol>0): one residue per molecule ID
          SOL = water (types 3/4)
          K   = potassium (type 5)
          CL  = chloride  (type 6)

    Atom names and elements are set from the LAMMPS type numbers.
    Box dimensions are written to the CRYST1 record.
    """
    import MDAnalysis as mda

    if material_name == "MoS2":
        TYPE_ELEMENT  = {1: 'C',  2: 'C',  3: 'O',  4: 'H',  5: 'K',  6: 'CL', 7: 'Mo', 8: 'S'}
        TYPE_ATOMNAME = {1: 'CPT', 2: 'CFT', 3: 'OW', 4: 'HW', 5: 'K',  6: 'CL', 7: 'MO', 8: 'S'}
        TYPE_RESNAME  = {1: 'GRP', 2: 'GRF', 3: 'SOL', 4: 'SOL', 5: 'K', 6: 'CL', 7: 'MSF', 8: 'MSF'}
    elif material_name == "Ti2C MXene":
        TYPE_ELEMENT  = {1: 'C',  2: 'C',  3: 'O',  4: 'H',  5: 'K',  6: 'CL', 7: 'Ti', 8: 'C'}
        TYPE_ATOMNAME = {1: 'CPT', 2: 'CFT', 3: 'OW', 4: 'HW', 5: 'K',  6: 'CL', 7: 'TI', 8: 'CM'}
        TYPE_RESNAME  = {1: 'GRP', 2: 'GRF', 3: 'SOL', 4: 'SOL', 5: 'K', 6: 'CL', 7: 'MXF', 8: 'MXF'}
    else:
        TYPE_ELEMENT  = {1: 'C',  2: 'C',  3: 'O',  4: 'H',  5: 'K',  6: 'CL', 7: 'B',  8: 'N'}
        TYPE_ATOMNAME = {1: 'CPT', 2: 'CFT', 3: 'OW', 4: 'HW', 5: 'K',  6: 'CL', 7: 'B',  8: 'N'}
        TYPE_RESNAME  = {1: 'GRP', 2: 'GRF', 3: 'SOL', 4: 'SOL', 5: 'K', 6: 'CL', 7: 'BNF', 8: 'BNF'}

    atoms = data['atoms']

    # ── Build residue list ──────────────────────────────────────────
    # residue key → (index, resname, resid)
    residue_map   = {}
    atom_resindex = []

    for a in atoms:
        if a['mol'] == 0:
            # Membrane atom: group by type
            key = ('mem', a['type'])
            resname = TYPE_RESNAME[a['type']]
            resid   = a['type']          # unique per membrane type
        else:
            # Fluid atom: group by molecule ID
            key = ('fluid', a['mol'])
            resname = TYPE_RESNAME.get(a['type'], 'UNK')
            resid   = a['mol']

        if key not in residue_map:
            residue_map[key] = (len(residue_map), resname, resid)
        atom_resindex.append(residue_map[key][0])

    # Sort residue entries by their assigned index
    sorted_residues = sorted(residue_map.values(), key=lambda x: x[0])
    resnames_list   = [r[1] for r in sorted_residues]
    resids_list     = [r[2] for r in sorted_residues]
    n_residues      = len(sorted_residues)

    # ── Build Universe ──────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        u = mda.Universe.empty(
            len(atoms),
            n_residues=n_residues,
            trajectory=True,
            atom_resindex=atom_resindex,
        )

        u.add_TopologyAttr('name',    [TYPE_ATOMNAME.get(a['type'], 'X') for a in atoms])
        u.add_TopologyAttr('resname', resnames_list)
        u.add_TopologyAttr('resid',   resids_list)
        u.add_TopologyAttr('element', [TYPE_ELEMENT.get(a['type'], 'X') for a in atoms])
        u.add_TopologyAttr('tempfactor', [0.0] * len(atoms))
        u.add_TopologyAttr('occupancy',  [1.0] * len(atoms))

        # Box dimensions (Å, orthorhombic)
        box   = data['box']
        lx    = box['xlo xhi'][1] - box['xlo xhi'][0]
        ly    = box['ylo yhi'][1] - box['ylo yhi'][0]
        lz    = box['zlo zhi'][1] - box['zlo zhi'][0]
        u.dimensions = [lx, ly, lz, 90., 90., 90.]

        u.atoms.positions = np.array([[a['x'], a['y'], a['z']] for a in atoms])

        u.atoms.write(output_path)

    return len(atoms)


# ── Input script generation ───────────────────────────────────────────

def generate_input_script(data_filename, data, material_name, pressure_mpa=100, run_id=1):
    mat = MATERIALS[material_name]

    type1_ids  = [a['id'] for a in data['atoms'] if a['type'] == 1]
    # Filter IDs: all types that belong to the membrane
    pore_ids   = [a['id'] for a in data['atoms'] if a['type'] in mat['filter_types']]

    gra1_lo, gra1_hi = min(type1_ids), max(type1_ids)
    gra2_lo, gra2_hi = min(pore_ids),  max(pore_ids)

    kick_force = 0.03386 * pressure_mpa / 100.0
    base       = os.path.splitext(data_filename)[0]
    trj_name   = f"{base}_{pressure_mpa}_{run_id}.lammpstrj"

    # Build pair_coeff block
    base_pairs = """\
pair_coeff      1 1  0.06919 3.390                  #gra (piston)
pair_coeff      3 3  0.15539 3.1656                 #O
pair_coeff      4 4  0.0000  0.0000                 #H
pair_coeff      5 5  0.4297  2.8384                 #K+
pair_coeff      6 6  0.01279 4.8305                 #Cl-"""

    # For non-Graphene materials the filter uses types 7/8; type 2 is a
    # declared-but-empty slot in the data file.  LAMMPS requires a coeff
    # for every declared type before mixing rules can be applied, so we
    # always emit a zero-interaction entry for it.
    _type2_placeholder = "pair_coeff      2 2  0.0000  0.0000               #unused type (graphene placeholder, no atoms present)"

    if material_name == "Graphene":
        filter_pairs = "pair_coeff      2 2  0.06919 3.390                  #PORE graphene"
    elif material_name == "h-BN":
        eps_B, sig_B = mat['extra_pairs'][7]
        eps_N, sig_N = mat['extra_pairs'][8]
        filter_pairs = (
            f"{_type2_placeholder}\n"
            f"pair_coeff      7 7  {eps_B:.4f} {sig_B:.3f}                 #B (h-BN filter)\n"
            f"pair_coeff      8 8  {eps_N:.4f} {sig_N:.3f}                 #N (h-BN filter)"
        )
    elif material_name == "MoS2":
        eps_Mo, sig_Mo = mat['extra_pairs'][7]
        eps_S,  sig_S  = mat['extra_pairs'][8]
        filter_pairs = (
            f"{_type2_placeholder}\n"
            f"pair_coeff      7 7  {eps_Mo:.4f} {sig_Mo:.3f}                #Mo (MoS2 filter)\n"
            f"pair_coeff      8 8  {eps_S:.4f} {sig_S:.3f}                 #S  (MoS2 filter)"
        )
    else:  # Ti2C MXene
        eps_Ti, sig_Ti = mat['extra_pairs'][7]
        eps_C,  sig_C  = mat['extra_pairs'][8]
        filter_pairs = (
            f"{_type2_placeholder}\n"
            f"pair_coeff      7 7  {eps_Ti:.4f} {sig_Ti:.4f}              #Ti (Ti2C filter, UFF)\n"
            f"pair_coeff      8 8  {eps_C:.4f} {sig_C:.3f}               #C  (Ti2C filter, Steele)"
        )

    # PORE group definition
    pore_group_line = f"group           PORE type {mat['pore_group_types']}    # filter membrane"

    script = f"""echo screen
units         real
dimension     3
boundary      p p p
atom_style    full
# neigh_modify exclude is incompatible with GPU neighbor builds;
# build neighbor lists on CPU, offload only pair forces to GPU.
package gpu 1 neigh no
neigh_modify delay 0 every 1 check yes {mat['neigh_exclude']}
processors    * * *

variable timecoef equal 1.0
variable T equal 300.00
variable thermo_itv equal 1.0
variable dump_itv equal 1.0
variable dumpvel_itv equal 1.0
variable equi_steps equal 1000
variable flow_stepsA equal 1000000
variable seed equal 2021
variable c_low equal 93.0
variable c_high equal 96.5

read_data     {data_filename}

# group definitions
group           gra type 1                # piston at z=12.5
{pore_group_line}
group           fluidmols type 3 4
group           Potassium type 5
group           Chloride type 6
group           mixfluids type 3 4 5 6
group           oxygen type 3
group           gra1 id {gra1_lo}:{gra1_hi}
group           gra2 id {gra2_lo}:{gra2_hi}

# force fields
variable        rcF equal 12.0
pair_style      lj/cut/coul/long  ${{rcF}} ${{rcF}}
{filter_pairs}
{base_pairs}
pair_modify     mix arithmetic

kspace_style    pppm 1e-05
dielectric      1.0
bond_style      harmonic
bond_coeff      1 450.0 1.0
angle_style     harmonic
angle_coeff     1 100.0 109.47

fix 20 PORE setforce 0.0 0.0 0.0
fix 9 gra setforce 0.0 0.0 0.0
min_style       sd
minimize        1.0e-4 1.0e-4 1000 1000

reset_timestep 0
timestep 1
compute fluidtemp mixfluids temp
velocity  mixfluids create ${{T}} ${{seed}} dist gaussian temp fluidtemp mom yes rot yes units box
velocity  PORE set 0.0 0.0 0.0 units box
fix 3 fluidmols shake 0.0001 20 0 b 1 a 1
velocity gra set 0.0 0.0 NULL units box
fix 18 gra rigid single force * off off on torque * off off off
fix 12 fluidmols npt temp 300 300 100.0 iso 1.0 1.0 100.0
run 5000
unfix 12
unfix 9

reset_timestep 0
timestep 1
fix 22 gra setforce 0.0 0.0 NULL
fix 2 fluidmols nvt temp 300 300 100
fix 15 Potassium nvt temp 300 300 100
fix 16 Chloride nvt temp 300 300 100

fix kick gra addforce 0.0 0.0 {kick_force:.5f}           #{pressure_mpa} MPa

dump 1 all custom 5000 {trj_name} id mol type x y z

thermo 500
thermo_style custom step lx ly lz press c_fluidtemp ke evdwl ecoul elong pe
run 10000000 #10ns
"""
    return script


# ── Streamlit App ─────────────────────────────────────────────────────

st.set_page_config(page_title="2D Material Pore Editor", layout="wide")
st.title("2D Material Pore Editor")

pore_dir     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "custom_lammps")
pore_dir     = os.path.abspath(pore_dir)
os.makedirs(pore_dir, exist_ok=True)

uploaded = st.sidebar.file_uploader("Upload .lammps file", type=["lammps"])
if uploaded is not None:
    upload_path = os.path.join(pore_dir, os.path.basename(uploaded.name))
    with open(upload_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success(f"Uploaded `{uploaded.name}`")

lammps_files = sorted([f for f in os.listdir(pore_dir) if f.endswith('.lammps')])

if not lammps_files:
    st.error(f"No .lammps files found in `{pore_dir}`")
    st.stop()

# ── Sidebar controls ──────────────────────────────────────────────────
st.sidebar.markdown("### Source file")
selected_file = st.sidebar.selectbox("Select data file", lammps_files)
filepath = os.path.join(pore_dir, selected_file)

st.sidebar.markdown("### Material")
material = st.sidebar.radio("Filter membrane material", list(MATERIALS.keys()))
st.sidebar.caption(MATERIALS[material]["description"])

if material == "h-BN":
    st.sidebar.markdown("### h-BN Charge Model")
    bn_charge_label = st.sidebar.radio(
        "Partial charge on B/N",
        list(BN_CHARGE_MODELS.keys()),
        index=0,
        help=(
            "Sets ±q on B and N atoms in the exported data file.\n\n"
            "The LAMMPS input already uses lj/cut/coul/long + PPPM, so "
            "charges are picked up automatically at run time."
        ),
    )
    bn_charge = BN_CHARGE_MODELS[bn_charge_label]
    if bn_charge > 0:
        st.sidebar.caption(
            f"B → +{bn_charge:.2f} e   N → −{bn_charge:.2f} e"
        )
    else:
        st.sidebar.caption("All charges = 0.0 e (LJ-only model)")
else:
    bn_charge = 0.0
    mos2_charges = (0.0, 0.0)

if material == "MoS2":
    st.sidebar.markdown("### MoS2 Charge Model")
    mos2_charge_label = st.sidebar.radio(
        "Partial charge on Mo/S",
        list(MOS2_CHARGE_MODELS.keys()),
        index=0,
        help=(
            "Sets partial charges on Mo and S atoms in the exported data file.\n\n"
            "The LAMMPS input already uses lj/cut/coul/long + PPPM, so "
            "charges are picked up automatically at run time."
        ),
    )
    mos2_charges = MOS2_CHARGE_MODELS[mos2_charge_label]
    if mos2_charges != (0.0, 0.0):
        st.sidebar.caption(
            f"Mo → {mos2_charges[0]:+.2f} e   S → {mos2_charges[1]:+.2f} e"
        )
    else:
        st.sidebar.caption("All charges = 0.0 e (LJ-only model)")
else:
    mos2_charges = (0.0, 0.0)

if material == "Ti2C MXene":
    st.sidebar.markdown("### Ti₂C Charge Model")
    ti2c_charge_label = st.sidebar.radio(
        "Partial charge on Ti/C",
        list(TI2C_CHARGE_MODELS.keys()),
        index=0,
        help=(
            "Sets partial charges on Ti and C atoms in the exported data file.\n\n"
            "Charge balance per formula unit: 2×q(Ti) + q(C) = 0.\n"
            "The LAMMPS input uses lj/cut/coul/long + PPPM so charges are "
            "applied automatically at run time."
        ),
    )
    ti2c_charges = TI2C_CHARGE_MODELS[ti2c_charge_label]
    if ti2c_charges != (0.0, 0.0):
        st.sidebar.caption(
            f"Ti → {ti2c_charges[0]:+.2f} e   C → {ti2c_charges[1]:+.2f} e"
        )
    else:
        st.sidebar.caption("All charges = 0.0 e (LJ-only model)")
else:
    ti2c_charges = (0.0, 0.0)

# Compute the z-shift needed to place the top atomic layer at TRILAYER_TOP_Z.
# For MoS2  the top S  sits at filter_z + ss/2;
# for Ti2C  the top Ti sits at filter_z + ti_z.
# Rearranging: sheet_z_shift = TRILAYER_TOP_Z − half_thickness − orig_filter_z.
# For monolayer materials (Graphene, h-BN) no shift is applied.
def _compute_sheet_z_shift(raw_atoms, mat):
    existing_t2 = [at for at in raw_atoms if at['type'] == 2]
    orig_fz = existing_t2[0]['z'] if existing_t2 else 96.5
    if mat == "MoS2":
        return TRILAYER_TOP_Z - (MOS2_DEFAULT_SS / 2.0) - orig_fz
    if mat == "Ti2C MXene":
        return TRILAYER_TOP_Z - TI2C_DEFAULT_TI_Z - orig_fz
    return 0.0

if material == "MoS2":
    _default_a = MOS2_DEFAULT_A
elif material == "Ti2C MXene":
    _default_a = TI2C_DEFAULT_A
elif material == "h-BN":
    _default_a = 2.50
else:
    _default_a = 2.46   # Graphene
lattice_a = _default_a
commensurate_box = True

st.sidebar.markdown("### Pore Design Tools")
tool_mode = st.sidebar.radio(
    "Selection mode",
    ["Lasso / box select", "Circle brush", "Rectangle brush", "Clear all"],
)

# ── Load / reconstruct ────────────────────────────────────────────────
# Sheet is rebuilt when the source file, lattice constant, or sheet kind changes.
# Material type conversion happens at export time, but MoS2 uses a different
# sheet geometry, so it triggers a rebuild.
if ('parsed' not in st.session_state
        or st.session_state.get('_loaded_file') != selected_file
        or st.session_state.get('_loaded_lattice_a') != lattice_a
        or st.session_state.get('_loaded_commensurate_box') != commensurate_box
        or st.session_state.get('_loaded_sheet_kind') != material):
    raw_data = parse_lammps_data(filepath)
    if material == "MoS2":
        sheet_kind = "mos2"
    elif material == "Ti2C MXene":
        sheet_kind = "ti2c"
    else:
        sheet_kind = "graphene"
    sheet_z_shift = _compute_sheet_z_shift(raw_data['atoms'], material)
    work = reconstruct_filter_with_ase(raw_data, a=lattice_a, commensurate_box=commensurate_box,
                                       sheet_kind=sheet_kind, sheet_z_shift=sheet_z_shift)
    if commensurate_box:
        # Rebuild piston sheet to match the commensurate box
        work = replace_piston_sheet(work, a=2.46)
    st.session_state.parsed = work
    st.session_state._loaded_file    = selected_file
    st.session_state._loaded_lattice_a = lattice_a
    st.session_state._loaded_commensurate_box = commensurate_box
    st.session_state._loaded_sheet_kind = material
    st.session_state._loaded_sheet_z_shift = sheet_z_shift
    st.session_state.deleted_ids     = set()
    st.session_state._sublattice     = None   # invalidate cached BFS

data         = st.session_state.parsed
filter_atoms = [a for a in data['atoms'] if a['type'] == 2]
piston_atoms = [a for a in data['atoms'] if a['type'] == 1]
n_full       = len(filter_atoms)   # ASE sheet size (may differ from piston)

# ── Box info ──────────────────────────────────────────────────────────
_xlo, _xhi = data['box']['xlo xhi']
_ylo, _yhi = data['box']['ylo yhi']
_lx = _xhi - _xlo
_ly = _yhi - _ylo
_ay = np.sqrt(3) * lattice_a
_nx = max(1, int(np.ceil(_lx / lattice_a)))
_ny = max(1, int(np.ceil(_ly / _ay)))
_target_lx = _nx * lattice_a
_target_ly = _ny * _ay

# ── BFS sublattice (cached, recomputed only when filter changes) ──────
# Key: frozenset of current filter atom IDs (changes as deletions are toggled)
if material == "h-BN":
    current_filter_ids = frozenset(a['id'] for a in filter_atoms)
    if (st.session_state.get('_sublattice_key') != current_filter_ids
            or st.session_state.get('_sublattice') is None):
        st.session_state._sublattice     = assign_bn_sublattices(filter_atoms, data['box'])
        st.session_state._sublattice_key = current_filter_ids
    sublattice = st.session_state._sublattice   # {atom_id: 0(B) or 1(N)}
else:
    sublattice = {}

# ── Clear all ─────────────────────────────────────────────────────────
if tool_mode == "Clear all":
    st.session_state.deleted_ids = set()
    st.sidebar.success("Selection cleared.")

# ── Brush tools ───────────────────────────────────────────────────────
if tool_mode == "Circle brush":
    brush_r = st.sidebar.slider("Radius (Å)", 1.0, 15.0, 5.0, 0.5)
    cx = st.sidebar.number_input("Center X (Å)", value=20.26, step=0.5)
    cy = st.sidebar.number_input("Center Y (Å)", value=20.56, step=0.5)
    if st.sidebar.button("Apply"):
        for a in filter_atoms:
            if (a['x']-cx)**2 + (a['y']-cy)**2 <= brush_r**2:
                st.session_state.deleted_ids.add(a['id'])
        st.rerun()

if tool_mode == "Rectangle brush":
    rx_lo = st.sidebar.number_input("X min (Å)", value=15.0, step=0.5)
    rx_hi = st.sidebar.number_input("X max (Å)", value=25.0, step=0.5)
    ry_lo = st.sidebar.number_input("Y min (Å)", value=15.0, step=0.5)
    ry_hi = st.sidebar.number_input("Y max (Å)", value=25.0, step=0.5)
    if st.sidebar.button("Apply"):
        for a in filter_atoms:
            if rx_lo <= a['x'] <= rx_hi and ry_lo <= a['y'] <= ry_hi:
                st.session_state.deleted_ids.add(a['id'])
        st.rerun()

# ── Stats ─────────────────────────────────────────────────────────────
n_deleted  = len(st.session_state.deleted_ids)
n_kept     = n_full - n_deleted
c1, c2, c3, c4 = st.columns(4)
c1.metric("Full sheet",        n_full)
c2.metric("Marked for deletion", n_deleted)
c3.metric("Remaining",         n_kept)
c4.metric("Total system atoms", data['counts']['atoms'] - n_deleted)
st.markdown("---")

# ── Plot ──────────────────────────────────────────────────────────────
# Color scheme:
#   Graphene: green (kept) / red (deleted)
#   h-BN:     #6fa8dc blue = B (kept) / #e69138 orange = N (kept) / red = deleted

colors, texts = [], []
for a in filter_atoms:
    aid = a['id']
    if aid in st.session_state.deleted_ids:
        colors.append('red')
    elif material == "h-BN":
        colors.append('#4a90d9' if sublattice.get(aid, 0) == 0 else '#e8a838')
    elif material == "Ti2C MXene":
        # Color by z-layer: C mid-plane (teal) vs Ti planes (steel blue).
        # We compare against the stored filter_z which is the first filter atom's z.
        # Use the same C_PLANE_TOL = 0.3 Å as apply_material.
        ref_z = filter_atoms[0]['z'] if filter_atoms else 0.0
        fzs = [at['z'] for at in filter_atoms]
        mid_z = float(np.median(fzs)) if fzs else ref_z
        colors.append('#20b2aa' if abs(a['z'] - mid_z) < 0.3 else '#4682b4')
    else:
        colors.append('#2ecc71')
    sl_label = ""
    if material == "h-BN":
        sl_label = "  B" if sublattice.get(aid, 0) == 0 else "  N"
    texts.append(f"ID: {aid}{sl_label}<br>x: {a['x']:.2f} Å<br>y: {a['y']:.2f} Å")

df_filter = pd.DataFrame(filter_atoms)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_filter['x'], y=df_filter['y'],
    mode='markers',
    marker=dict(size=8, color=colors, line=dict(width=0.5, color='#333')),
    text=texts,
    hoverinfo='text',
    customdata=df_filter['id'].values,
))

if material == "h-BN":
    title_mat = "h-BN  (blue=B, orange=N)"
elif material == "MoS2":
    title_mat = "MoS₂  (green=Mo/S)"
elif material == "Ti2C MXene":
    title_mat = "Ti₂C MXene  (steel=Ti, teal=C)"
else:
    title_mat = "Graphene"
fig.update_layout(
    title=f"Filter membrane — {selected_file}  [{title_mat}]",
    xaxis_title="x (Å)", yaxis_title="y (Å)",
    xaxis=dict(scaleanchor="y", scaleratio=1, range=[-1, 43]),
    yaxis=dict(range=[-1, 44]),
    height=650,
    dragmode='lasso',
    template='plotly_white',
)

event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="pore_plot")

# Handle lasso / box selection
if event and event.selection and event.selection.points:
    newly = set()
    for pt in event.selection.points:
        cd = pt.get('customdata')
        if cd is None:
            continue
        newly.add(int(cd[0]) if isinstance(cd, (list, tuple)) else int(cd))
    if newly:
        already = newly & st.session_state.deleted_ids
        if already == newly:
            st.session_state.deleted_ids -= newly
        else:
            st.session_state.deleted_ids |= newly
        st.rerun()

# ── Export ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Export")

col_a, col_b = st.columns(2)
output_name = col_a.text_input("Output filename (.lammps)", value="custom_pore.lammps")
pressure    = col_b.number_input("Pressure (MPa)", value=100, step=10)

if st.button("Generate data file & input script", type="primary",
             disabled=n_deleted == 0):

    # 1. Delete selected atoms from working copy
    work, id_map = delete_atoms_and_rewrite(copy.deepcopy(data), st.session_state.deleted_ids)

    # 2. Convert to target material (remap sublattice IDs after deletion/reindexing)
    remapped_sublattice = {id_map[old_id]: sl for old_id, sl in sublattice.items() if old_id in id_map}
    work = apply_material(
        work,
        material,
        sublattice_map=remapped_sublattice,
        bn_charge=bn_charge,
        mos2_charges=mos2_charges,
        ti2c_charges=ti2c_charges,
    )

    # 3. Build header
    n_filter_out = len([a for a in work['atoms'] if a['type'] in MATERIALS[material]['filter_types']])
    if material == "h-BN":
        charge_note = (f", q(B/N)=±{bn_charge:.2f}e" if bn_charge > 0 else ", uncharged")
    elif material == "MoS2":
        mo_q, s_q = mos2_charges
        charge_note = (f", q(Mo/S)={mo_q:+.2f}/{s_q:+.2f}e" if mos2_charges != (0.0, 0.0)
                       else ", uncharged")
    elif material == "Ti2C MXene":
        ti_q, c_q = ti2c_charges
        charge_note = (f", q(Ti/C)={ti_q:+.2f}/{c_q:+.2f}e" if ti2c_charges != (0.0, 0.0)
                       else ", uncharged")
    else:
        charge_note = ""
    header = (f"{material} pore — removed {n_deleted} filter atoms, "
              f"{n_filter_out} remaining{charge_note}")

    # 4. Write data file
    data_str = write_lammps_data(work, header_comment=header)
    with open(os.path.join(pore_dir, output_name), 'w') as f:
        f.write(data_str)

    # 5. Write input script
    script_str = generate_input_script(output_name, work, material,
                                       pressure_mpa=pressure, run_id=1)
    input_name = os.path.splitext(output_name)[0] + f"_{pressure}_1.input"
    with open(os.path.join(pore_dir, input_name), 'w') as f:
        f.write(script_str)

    st.success(f"Wrote **{output_name}** ({work['counts']['atoms']} atoms) and **{input_name}**")
    st.balloons()

    pore_ids = [a['id'] for a in work['atoms'] if a['type'] in MATERIALS[material]['filter_types']]
    if material == "h-BN":
        charge_summary = f"  |  charges: ±{bn_charge:.2f} e"
    elif material == "MoS2":
        mo_q, s_q = mos2_charges
        charge_summary = f"  |  charges: {mo_q:+.2f}/{s_q:+.2f} e"
    else:
        charge_summary = ""
    st.code(
        f"Material: {material}{charge_summary}  |  "
        f"Filter atoms: {n_filter_out}  |  "
        f"gra2 range: {min(pore_ids)}:{max(pore_ids)}  |  "
        f"Total: {work['counts']['atoms']}",
        language=None,
    )

    # Store the export-ready data for PDB export below
    st.session_state._last_export = work
    st.session_state._last_export_material = material

# ── PDB Export ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("PDB Export")
st.caption(
    "Exports the last generated data file to PDB format for use in VMD, OVITO, or PyMOL. "
    "Generates the .lammps file first if you haven't already."
)

pdb_col_a, pdb_col_b = st.columns(2)
pdb_output_name = pdb_col_a.text_input("PDB filename", value="custom_pore.pdb")
export_all = pdb_col_b.checkbox("Full system (water + ions)", value=False,
                                 help="Unchecked = membrane only; checked = all atoms")

if st.button("Export to PDB", disabled='_last_export' not in st.session_state):
    export_data = st.session_state._last_export
    export_mat  = st.session_state._last_export_material

    if not export_all:
        # Membrane-only: keep only types belonging to the filter + piston
        keep_types = {1} | set(MATERIALS[export_mat]['filter_types'])
        filtered_atoms = [a for a in export_data['atoms'] if a['type'] in keep_types]
        # Build a minimal data dict for PDB writer
        pdb_data = copy.deepcopy(export_data)
        pdb_data['atoms'] = filtered_atoms
    else:
        pdb_data = export_data

    pdb_path = os.path.join(pore_dir, pdb_output_name)
    n_written = export_to_pdb(pdb_data, export_mat, pdb_path)
    st.success(f"Wrote **{pdb_output_name}** ({n_written} atoms)")
    st.code(
        f"Residues: GRP=piston  "
        + ("GRF=filter" if export_mat == "Graphene"
           else ("BNF=filter (B+N)" if export_mat == "h-BN" else "MSF=filter (Mo+S)"))
        + ("  SOL=water  K/CL=ions" if export_all else ""),
        language=None,
    )
elif '_last_export' not in st.session_state:
    st.info("Generate a data file first, then export to PDB.")
