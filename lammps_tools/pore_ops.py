"""Operations for pore editing and filter reconstruction."""

from __future__ import annotations

import copy
from typing import Dict, Any, Iterable, Tuple


def reconstruct_full_filter(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rebuild a complete filter membrane (type 2) from the piston (type 1).

    The piston sheet at z=12.5 is always a full 680-atom graphene lattice.
    The filter sheet at z=96.5 may already have atoms missing (a pre-cut pore).
    This function mirrors every type-1 atom to create a full type-2 sheet,
    replacing whatever partial set of type-2 atoms was in the file.
    """
    piston_atoms = [a for a in data["atoms"] if a["type"] == 1]
    non_filter_atoms = [a for a in data["atoms"] if a["type"] != 2]

    # Determine filter z from existing type-2 atoms (fallback 96.5)
    existing_filter = [a for a in data["atoms"] if a["type"] == 2]
    filter_z = existing_filter[0]["z"] if existing_filter else 96.5

    # Create full filter by cloning piston positions at filter_z
    next_id = max(a["id"] for a in non_filter_atoms) + 1 if non_filter_atoms else 1
    new_filter = []
    for pa in piston_atoms:
        new_filter.append(
            {
                "id": next_id,
                "mol": 0,
                "type": 2,
                "charge": 0.0,
                "x": pa["x"],
                "y": pa["y"],
                "z": filter_z,
            }
        )
        next_id += 1

    # Reassemble: non-filter atoms + full filter, then renumber
    all_atoms = non_filter_atoms + new_filter
    for i, a in enumerate(all_atoms, start=1):
        a["id"] = i

    new_data = copy.deepcopy(data)
    new_data["atoms"] = all_atoms
    new_data["counts"]["atoms"] = len(all_atoms)

    # Bonds and angles don't reference graphene atoms (they're rigid),
    # but we still need to remap IDs for the non-filter atoms that
    # may have shifted due to renumbering.
    # Build old->new map for atoms that existed before
    id_map = {}
    old_non_filter = [a for a in data["atoms"] if a["type"] != 2]
    for new_a, old_a in zip(all_atoms[: len(non_filter_atoms)], old_non_filter):
        id_map[old_a["id"]] = new_a["id"]

    # Remap bonds
    kept_bonds = []
    for b in data["bonds"]:
        if b["a1"] in id_map and b["a2"] in id_map:
            kept_bonds.append(
                {
                    "id": len(kept_bonds) + 1,
                    "type": b["type"],
                    "a1": id_map[b["a1"]],
                    "a2": id_map[b["a2"]],
                }
            )
    new_data["bonds"] = kept_bonds
    new_data["counts"]["bonds"] = len(kept_bonds)

    # Remap angles
    kept_angles = []
    for ang in data["angles"]:
        if ang["a1"] in id_map and ang["a2"] in id_map and ang["a3"] in id_map:
            kept_angles.append(
                {
                    "id": len(kept_angles) + 1,
                    "type": ang["type"],
                    "a1": id_map[ang["a1"]],
                    "a2": id_map[ang["a2"]],
                    "a3": id_map[ang["a3"]],
                }
            )
    new_data["angles"] = kept_angles
    new_data["counts"]["angles"] = len(kept_angles)

    return new_data


def delete_atoms_and_rewrite(
    data: Dict[str, Any], ids_to_delete: Iterable[int]
) -> Tuple[Dict[str, Any], Dict[int, int]]:
    """
    Remove atoms by ID, prune bonds/angles that reference them,
    renumber everything contiguously.
    """
    remove_set = set(ids_to_delete)
    # Filter atoms
    kept_atoms = [a for a in data["atoms"] if a["id"] not in remove_set]

    # Build old→new ID map
    id_map: Dict[int, int] = {}
    for i, a in enumerate(kept_atoms, start=1):
        id_map[a["id"]] = i
        a["id"] = i

    # Filter bonds: keep only if both atoms survive
    kept_bonds = []
    for b in data["bonds"]:
        if b["a1"] in id_map and b["a2"] in id_map:
            kept_bonds.append(
                {
                    "id": len(kept_bonds) + 1,
                    "type": b["type"],
                    "a1": id_map[b["a1"]],
                    "a2": id_map[b["a2"]],
                }
            )

    # Filter angles
    kept_angles = []
    for ang in data["angles"]:
        if ang["a1"] in id_map and ang["a2"] in id_map and ang["a3"] in id_map:
            kept_angles.append(
                {
                    "id": len(kept_angles) + 1,
                    "type": ang["type"],
                    "a1": id_map[ang["a1"]],
                    "a2": id_map[ang["a2"]],
                    "a3": id_map[ang["a3"]],
                }
            )

    new_data = copy.deepcopy(data)
    new_data["atoms"] = kept_atoms
    new_data["bonds"] = kept_bonds
    new_data["angles"] = kept_angles
    new_data["counts"]["atoms"] = len(kept_atoms)
    new_data["counts"]["bonds"] = len(kept_bonds)
    new_data["counts"]["angles"] = len(kept_angles)

    return new_data, id_map
