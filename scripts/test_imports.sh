#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

PYTHONPATH=. python -c "from mcp_implement.lammps_tools import parse_lammps_data, reconstruct_full_filter, delete_atoms_and_rewrite, write_lammps_data, generate_input_script; print('ok')"
