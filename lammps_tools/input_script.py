"""LAMMPS input script generation."""

from __future__ import annotations

import os
from typing import Dict, Any


def generate_input_script(
    data_filename: str,
    data: Dict[str, Any],
    pressure_mpa: float = 100,
    run_id: int = 1,
) -> str:
    """Generate a matching LAMMPS input script for the new data file."""
    # Figure out group ranges
    type1_ids = [a["id"] for a in data["atoms"] if a["type"] == 1]
    type2_ids = [a["id"] for a in data["atoms"] if a["type"] == 2]

    gra1_lo, gra1_hi = min(type1_ids), max(type1_ids)
    gra2_lo, gra2_hi = min(type2_ids), max(type2_ids)

    # Pressure → force mapping (from your input files: 0.03386 = 100 MPa)
    force_per_100mpa = 0.03386
    kick_force = force_per_100mpa * pressure_mpa / 100.0

    base = os.path.splitext(data_filename)[0]
    trj_name = f"{base}_{pressure_mpa}_{run_id}.lammpstrj"

    script = f"""echo screen
units         real
dimension     3
boundary      p p p
atom_style    full
neigh_modify delay 0 every 1 check yes exclude type 1 1 exclude type 2 2
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
group           PORE type 2               # filter membrane at z=96.5
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
pair_coeff      1 1  0.06919 3.390                  #gra (piston)
pair_coeff      2 2  0.06919 3.390                  #PORE (filter)
pair_coeff      3 3  0.15539 3.1656                 #O
pair_coeff      4 4  0.0000  0.0000                 #H
pair_coeff      5 5  0.4297  2.8384                 #K+
pair_coeff      6 6  0.01279 4.8305                 #Cl-
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
