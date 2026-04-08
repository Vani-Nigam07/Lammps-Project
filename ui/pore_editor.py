"""
Stable Graphene Pore Editor — Streamlit app with brush + manual ID selection.

No lasso/selection events are used to avoid fragile Streamlit event handling.
"""

import copy
import json
import os
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


from mcp_implement.lammps_tools import (
    parse_lammps_data,
    reconstruct_full_filter,
    delete_atoms_and_rewrite,
    write_lammps_data,
    generate_input_script,
)


def _parse_id_input(text: str) -> set:
    """Parse IDs like '1,5,7-10' into a set of ints."""
    ids = set()
    if not text:
        return ids
    parts = re.split(r"[,\s]+", text.strip())
    for part in parts:
        if not part:
            continue
        if "-" in part:
            try:
                lo_s, hi_s = part.split("-", 1)
                lo = int(lo_s.strip())
                hi = int(hi_s.strip())
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            ids.update(range(lo, hi + 1))
        else:
            try:
                ids.add(int(part))
            except ValueError:
                continue
    return ids


def _export_outputs(
    output_name: str,
    pressure: float,
    run_id: int,
    data: dict,
    deleted_ids: set,
) -> tuple:
    """Write modified LAMMPS data + input script to custom_lammps."""
    new_data, _id_map = delete_atoms_and_rewrite(copy.deepcopy(data), deleted_ids)
    n_removed = len(deleted_ids)
    n_filter_new = len([a for a in new_data["atoms"] if a["type"] == 2])
    new_header = (
        f"Custom pore — removed {n_removed} filter atoms, "
        f"{n_filter_new} remaining (type2)"
    )

    repo_root = Path(__file__).resolve().parents[2]
    export_dir = repo_root / "mcp_implement" / "custom_lammps"
    os.makedirs(export_dir, exist_ok=True)

    data_str = write_lammps_data(new_data, header_comment=new_header)
    out_data_path = export_dir / output_name
    out_data_path.write_text(data_str)

    input_script = generate_input_script(
        output_name, new_data,
        pressure_mpa=pressure, run_id=run_id
    )
    input_name = os.path.splitext(output_name)[0] + f"_{pressure}_run{run_id}_export.input"
    out_input_path = export_dir / input_name
    out_input_path.write_text(input_script)

    meta_payload = {
        "lammps_path": str(out_data_path),
        "input_path": str(out_input_path),
        "pressure": pressure,
        "run_id": run_id,
    }
    meta_path = export_dir / f"{pressure}_run{run_id}.json"
    meta_path.write_text(json.dumps(meta_payload, indent=2))

    last_meta_path = export_dir / "last_export.json"
    last_meta_path.write_text(json.dumps(meta_payload, indent=2))

    return new_data, str(out_data_path), str(out_input_path), input_name


st.set_page_config(page_title="Graphene Pore Editor (Stable)", layout="wide")
st.title("Graphene Pore Editor — Stable")

repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "mcp_implement" / "script_generation"
os.makedirs(data_dir, exist_ok=True)

st.sidebar.header("Data File")
uploaded = st.sidebar.file_uploader("Upload .lammps file", type=["lammps"])
if uploaded is not None:
    upload_path = data_dir / uploaded.name
    upload_path.write_bytes(uploaded.getbuffer())
    st.sidebar.success(f"Uploaded `{uploaded.name}`")

lammps_files = sorted([p.name for p in data_dir.iterdir() if p.suffix == ".lammps"])
if not lammps_files:
    st.info("No .lammps files found. Please upload one using the sidebar.")
    st.stop()

selected_file = st.sidebar.selectbox("Select data file", lammps_files)
filepath = data_dir / selected_file

if (
    "parsed" not in st.session_state
    or st.session_state.get("_loaded_file") != selected_file
):
    raw_data = parse_lammps_data(str(filepath))
    st.session_state.parsed = reconstruct_full_filter(raw_data)
    st.session_state._loaded_file = selected_file
    st.session_state.deleted_ids = set()
    st.session_state.run_id = 1

data = st.session_state.parsed
filter_atoms = [a for a in data["atoms"] if a["type"] == 2]
piston_atoms = [a for a in data["atoms"] if a["type"] == 1]
n_full_sheet = len(piston_atoms)

df_filter = pd.DataFrame(filter_atoms)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Full sheet atoms", n_full_sheet)
col2.metric("Marked for deletion", len(st.session_state.deleted_ids))
col3.metric("Remaining after cut", n_full_sheet - len(st.session_state.deleted_ids))
col4.metric("Total atoms (system)", data["counts"]["atoms"] - len(st.session_state.deleted_ids))

st.markdown("---")

st.sidebar.header("Selection Tools")
tool_mode = st.sidebar.radio(
    "Selection mode",
    ["Circle brush", "Rectangle brush", "Manual ID input", "Clear all"],
)

if tool_mode == "Clear all":
    st.session_state.deleted_ids = set()
    st.sidebar.success("Selection cleared.")

fig = go.Figure()
colors = []
texts = []
for _, row in df_filter.iterrows():
    aid = int(row["id"])
    if aid in st.session_state.deleted_ids:
        colors.append("red")
    else:
        colors.append("#2ecc71")
    texts.append(f"ID: {aid}<br>x: {row['x']:.2f}<br>y: {row['y']:.2f}")

fig.add_trace(go.Scatter(
    x=df_filter["x"],
    y=df_filter["y"],
    mode="markers",
    marker=dict(size=8, color=colors, line=dict(width=0.5, color="#333")),
    text=texts,
    hoverinfo="text",
))
fig.update_layout(
    title=f"Filter membrane — {selected_file}",
    xaxis_title="x (Å)",
    yaxis_title="y (Å)",
    xaxis=dict(scaleanchor="y", scaleratio=1),
    yaxis=dict(),
    height=650,
    dragmode="pan",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

if tool_mode == "Circle brush":
    st.sidebar.markdown("#### Brush center")
    cx = st.sidebar.number_input("Center X (Å)", value=20.26, step=0.5)
    cy = st.sidebar.number_input("Center Y (Å)", value=20.56, step=0.5)
    radius = st.sidebar.slider("Brush radius (Å)", 1.0, 15.0, 5.0, 0.5)
    if st.sidebar.button("Apply circle brush"):
        for a in filter_atoms:
            dist = np.sqrt((a["x"] - cx) ** 2 + (a["y"] - cy) ** 2)
            if dist <= radius:
                st.session_state.deleted_ids.add(a["id"])
        st.rerun()

if tool_mode == "Rectangle brush":
    st.sidebar.markdown("#### Rectangle bounds")
    rx_lo = st.sidebar.number_input("X min (Å)", value=15.0, step=0.5)
    rx_hi = st.sidebar.number_input("X max (Å)", value=25.0, step=0.5)
    ry_lo = st.sidebar.number_input("Y min (Å)", value=15.0, step=0.5)
    ry_hi = st.sidebar.number_input("Y max (Å)", value=25.0, step=0.5)
    if st.sidebar.button("Apply rectangle brush"):
        for a in filter_atoms:
            if rx_lo <= a["x"] <= rx_hi and ry_lo <= a["y"] <= ry_hi:
                st.session_state.deleted_ids.add(a["id"])
        st.rerun()

if tool_mode == "Manual ID input":
    st.sidebar.markdown("#### Manual IDs")
    id_text = st.sidebar.text_input("IDs (e.g., 1,5,7-10)")
    if st.sidebar.button("Add IDs"):
        ids = _parse_id_input(id_text)
        st.session_state.deleted_ids |= ids
        st.rerun()

st.markdown("---")
st.subheader("Export")

col_a, col_b = st.columns(2)
output_name = col_a.text_input("Output filename", value="custom_pore.lammps")
pressure = col_b.number_input("Pressure (MPa)", value=100, step=10)

if st.button(
    "Generate data file & input script",
    type="primary",
    disabled=len(st.session_state.deleted_ids) == 0,
):
    new_data, out_data_path, out_input_path, input_name = _export_outputs(
        output_name,
        pressure,
        st.session_state.run_id,
        data,
        st.session_state.deleted_ids,
    )
    st.session_state.run_id += 1

    st.success(
        f"Wrote **{output_name}** ({new_data['counts']['atoms']} atoms) and **{input_name}**"
    )
    st.caption(f"Data file: {out_data_path}")
    st.caption(f"Input script: {out_input_path}")
    st.balloons()

    type2_new = [a for a in new_data["atoms"] if a["type"] == 2]
    gra2_lo = min(a["id"] for a in type2_new)
    gra2_hi = max(a["id"] for a in type2_new)
    st.code(
        f"Filter atoms: {len(type2_new)}  |  "
        f"gra2 range: {gra2_lo}:{gra2_hi}  |  "
        f"Total atoms: {new_data['counts']['atoms']}",
        language=None,
    )
