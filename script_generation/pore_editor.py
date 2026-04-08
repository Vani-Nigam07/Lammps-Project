"""
Graphene Pore Editor — Streamlit app for interactive pore design.

Reads a LAMMPS data file, displays the filter membrane (type 2) atoms
as a 2D scatter plot, lets you lasso/box-select atoms for deletion,
then rewrites the data file with those atoms removed and IDs renumbered.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import copy
import os
import sys
from pathlib import Path
import json
from streamlit_plotly_events import plotly_events

# Allow running this file directly (e.g., `python pore_editor.py` or `streamlit run`)
# by ensuring the repo root is on sys.path so `mcp_implement` imports resolve.
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

# ── Streamlit App ────────────────────────────────────────────────────


def _extract_selected_ids(event) -> set:
    """Extract selected atom IDs from a Streamlit plotly selection event."""
    if event is None:
        print("DEBUG: No event data received from plotly selection.")
        return set()
    if isinstance(event, list):
        points = event
        if not points:
            print("DEBUG: No points in selection.")
            return set()
        print(f"DEBUG: Extracting IDs from {len(points)} selected points.")
        selected_ids = set()
        for pt in points:
            cd = pt.get("customdata") if isinstance(pt, dict) else getattr(pt, "customdata", None)
            if cd is None:
                continue
            if isinstance(cd, (list, tuple)):
                if len(cd) > 0:
                    selected_ids.add(int(cd[0]))
            else:
                selected_ids.add(int(cd))
        return selected_ids
    selection = None
    if isinstance(event, dict):
        selection = event.get("selection")
    elif hasattr(event, "selection"):
        selection = getattr(event, "selection")
    if not selection:
        print("DEBUG: No  selection.")
        return set()
    points = selection.get("points") if isinstance(selection, dict) else getattr(selection, "points", None)
    if not points:
        print("DEBUG: No points in selection.")
        return set()
    print(f"DEBUG: Extracting IDs from {len(points)} selected points.")
    selected_ids = set()
    for pt in points:
        cd = pt.get("customdata") if isinstance(pt, dict) else getattr(pt, "customdata", None)
        if cd is None:
            continue
        if isinstance(cd, (list, tuple)):
            if len(cd) > 0:
                selected_ids.add(int(cd[0]))
        else:
            selected_ids.add(int(cd))
    return selected_ids


def _write_export_outputs(
    pore_dir: str,
    output_name: str,
    pressure: float,
    run_id: int,
    data: dict,
    deleted_ids: set,
) -> tuple:
    """Write modified LAMMPS data + input script to disk."""
    new_data, _id_map = delete_atoms_and_rewrite(copy.deepcopy(data), deleted_ids)
    n_removed = len(deleted_ids)
    n_filter_new = len([a for a in new_data["atoms"] if a["type"] == 2])
    new_header = (
        f"Custom pore — removed {n_removed} filter atoms, "
        f"{n_filter_new} remaining (type2)"
    )

    export_dir = _repo_root / "mcp_implement" / "custom_lammps"
    os.makedirs(export_dir, exist_ok=True)

    data_str = write_lammps_data(new_data, header_comment=new_header)
    out_data_path = os.path.join(str(export_dir), output_name)
    with open(out_data_path, "w") as f:
        f.write(data_str)

    input_script = generate_input_script(
        output_name, new_data,
        pressure_mpa=pressure, run_id=run_id
    )
    input_name = os.path.splitext(output_name)[0] + f"_{pressure}_run{run_id}_export.input"
    out_input_path = os.path.join(str(export_dir), input_name)
    with open(out_input_path, "w") as f:
        f.write(input_script)

    meta_payload = {
        "lammps_path": out_data_path,
        "input_path": out_input_path,
        "pressure": pressure,
        "run_id": run_id,
    }
    meta_path = export_dir / f"{pressure}_run{run_id}.json"
    with open(meta_path, "w") as f:
        json.dump(meta_payload, f, indent=2)

    last_meta_path = export_dir / "last_export.json"
    with open(last_meta_path, "w") as f:
        json.dump(meta_payload, f, indent=2)

    return new_data, out_data_path, out_input_path, input_name

st.set_page_config(page_title="Graphene Pore Editor", layout="wide")
st.title("Graphene Pore Editor")

# Sidebar: file selection
pore_dir = os.path.dirname(os.path.abspath(__file__))
lammps_files = sorted([f for f in os.listdir(pore_dir) if f.endswith('.lammps')])

uploaded = st.sidebar.file_uploader("Upload .lammps file", type=["lammps"])
if uploaded is not None:
    upload_path = os.path.join(pore_dir, uploaded.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded.getbuffer())
    if uploaded.name not in lammps_files:
        lammps_files.append(uploaded.name)
        lammps_files = sorted(lammps_files)
    st.sidebar.success(f"Uploaded `{uploaded.name}`")

if not lammps_files:
    st.info(" No .lammps files found. Please upload one using the sidebar.")
    # ← Remove st.stop() entirely, let sidebar finish rendering
else:
    default_index = 0
    if uploaded is not None and uploaded.name in lammps_files:
        default_index = lammps_files.index(uploaded.name)

    selected_file = st.sidebar.selectbox("Select data file", lammps_files, index=default_index)
    filepath = os.path.join(pore_dir, selected_file)

    # Parse
    if 'parsed' not in st.session_state or st.session_state.get('_loaded_file') != selected_file:
        raw_data = parse_lammps_data(filepath)
        st.session_state.parsed = reconstruct_full_filter(raw_data)
        st.session_state._loaded_file = selected_file
        st.session_state.deleted_ids = set()
        st.session_state.run_id = 1
    
    data = st.session_state.parsed

    # Extract filter atoms (type 2) — now guaranteed to be a full 680-atom sheet
    filter_atoms = [a for a in data['atoms'] if a['type'] == 2]
    print(f"DEBUG: Loaded {len(filter_atoms)} filter atoms (type 2) from {selected_file}")
    piston_atoms = [a for a in data['atoms'] if a['type'] == 1]
    print(f"DEBUG: Loaded {len(piston_atoms)} piston atoms (type 1) from {selected_file}")
    n_full_sheet = len(piston_atoms)  # piston is always the complete lattice

    df_filter = pd.DataFrame(filter_atoms)
    st.write("x range:", float(df_filter['x'].min()), "→", float(df_filter['x'].max()))
    st.write("y range:", float(df_filter['y'].min()), "→", float(df_filter['y'].max()))
    st.write("Sample atom:", df_filter.iloc[0].to_dict())

    # MINIMAL TEST — ---------------REMOVE ----------------------------
    st.write("df_filter shape:", df_filter.shape)
    st.write("df_filter columns:", list(df_filter.columns))
    st.write("First 3 rows:", df_filter[['x','y','id']].head(3))

    # Bare minimum plot — no colors, no customdata, nothing fancy
   
    fig_test = px.scatter(df_filter, x='x', y='y')
    st.plotly_chart(fig_test, use_container_width=True)

    # ── Stats ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Full sheet atoms", n_full_sheet)
    col2.metric("Marked for deletion", len(st.session_state.deleted_ids))
    col3.metric("Remaining after cut", n_full_sheet - len(st.session_state.deleted_ids))
    col4.metric("Total atoms (system)", data['counts']['atoms'] - len(st.session_state.deleted_ids))

    st.markdown("---")

    # ── Selection tools ──
    st.sidebar.markdown("### Pore Design Tools")

    tool_mode = st.sidebar.radio(
        "Selection mode",
        ["Click to toggle", "Circle brush", "Rectangle brush", "Clear all"],
    )

    if tool_mode == "Clear all":
        st.session_state.deleted_ids = set()
        st.sidebar.success("Selection cleared.")

    brush_radius = None
    if tool_mode == "Circle brush":
        brush_radius = st.sidebar.slider("Brush radius (Å)", 1.0, 15.0, 5.0, 0.5)

    if tool_mode == "Rectangle brush":
        st.sidebar.markdown("Use the inputs below to define the rectangle bounds.")
        x_min = st.sidebar.number_input("X Min (Å)", value=df_filter['x'].min())
        x_max = st.sidebar.number_input("X Max (Å)", value=df_filter['x'].max())
        y_min = st.sidebar.number_input("Y Min (Å)", value=df_filter['y'].min())
        y_max = st.sidebar.number_input("Y Max (Å)", value=df_filter['y'].max())

    if tool_mode == 'click to toggle':
        st.sidebar.markdown("Use the lasso or box select tool on the plot to mark atoms for deletion. "
                            "Clicking already marked atoms will unmark them.")
        

    # ── Build the scatter plot ──
    fig = go.Figure()

    # Existing pore (atoms that would be in a full sheet but aren't here)
    # We only show what IS in the file as type 2

    # Color: green = kept, red = marked for deletion
    colors = []
    texts = []
    for _, row in df_filter.iterrows():
        aid = int(row['id'])
        if aid in st.session_state.deleted_ids:
            colors.append('red')
        else:
            colors.append('#2ecc71')
        texts.append(f"ID: {aid}<br>x: {row['x']:.2f}<br>y: {row['y']:.2f}")

    fig.add_trace(go.Scatter(
        x=df_filter['x'],
        y=df_filter['y'],
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            line=dict(width=0.5, color='#333'),
        ),
        text=texts,
        hoverinfo='text',
        customdata=df_filter['id'].values,
    ))

    fig.update_layout(
        title=f"Filter membrane — {selected_file}",
        xaxis_title="x (Å)",
        yaxis_title="y (Å)",
        # xaxis=dict(scaleanchor="y", scaleratio=1, range=[-1, 43]),
        # yaxis=dict(range=[-1, 44]),
        xaxis=dict(scaleanchor="y", scaleratio=1),   # ← remove hardcoded range
        yaxis=dict(),
        height=650,
        dragmode='lasso',
        clickmode="event+select",
        template='plotly_white',
    )

    # Show plot
    # event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="pore_plot")

    # # Handle lasso/box selection from plotly
    # if event and event.selection and event.selection.points:
    #     selected_points = event.selection.points
    #     newly_selected = set()
    #     for pt in selected_points:
    #         cd = pt.get('customdata')
    #         if cd is None:
    #             continue
    #         # customdata can be a bare int/float or a list depending on Plotly version
    #         if isinstance(cd, (list, tuple)):
    #             if len(cd) > 0:
    #                 newly_selected.add(int(cd[0]))
    #         else:
    #             newly_selected.add(int(cd))

    #     if newly_selected:
    #         # Toggle: if all selected are already marked, unmark them; otherwise mark them
    #         already_marked = newly_selected & st.session_state.deleted_ids
    #         if already_marked == newly_selected:
    #             st.session_state.deleted_ids -= newly_selected
    #         else:
    #             st.session_state.deleted_ids |= newly_selected
    #         st.rerun()
    selected_points = plotly_events(
        fig,
        click_event=False,
        select_event=True,
        hover_event=False,
        override_height=650,
        key="pore_plot",
    )
    newly_selected = _extract_selected_ids(selected_points)
    if newly_selected:
        already_marked = newly_selected & st.session_state.deleted_ids
        if already_marked == newly_selected:
            st.session_state.deleted_ids -= newly_selected
        else:
            st.session_state.deleted_ids |= newly_selected
        st.rerun()

    # ── Circle brush via coordinate input ──
    if tool_mode == "Circle brush":
        st.sidebar.markdown("#### Brush center")
        cx = st.sidebar.number_input("Center X (Å)", value=20.26, step=0.5)
        cy = st.sidebar.number_input("Center Y (Å)", value=20.56, step=0.5)
        if st.sidebar.button("Apply circle brush"):
            for a in filter_atoms:
                dist = np.sqrt((a['x'] - cx)**2 + (a['y'] - cy)**2)
                if dist <= brush_radius:
                    st.session_state.deleted_ids.add(a['id'])
            st.write(f"DEBUG: deleted_ids = {len(st.session_state.get('deleted_ids', set()))}")
            st.rerun()

    if tool_mode == "Rectangle brush":
        st.sidebar.markdown("#### Rectangle bounds")
        rx_lo = st.sidebar.number_input("X min (Å)", value=15.0, step=0.5)
        rx_hi = st.sidebar.number_input("X max (Å)", value=25.0, step=0.5)
        ry_lo = st.sidebar.number_input("Y min (Å)", value=15.0, step=0.5)
        ry_hi = st.sidebar.number_input("Y max (Å)", value=25.0, step=0.5)
        if st.sidebar.button("Apply rectangle brush"):
            for a in filter_atoms:
                if rx_lo <= a['x'] <= rx_hi and ry_lo <= a['y'] <= ry_hi:
                    st.session_state.deleted_ids.add(a['id'])
            st.write(f"DEBUG: deleted_ids = {len(st.session_state.get('deleted_ids', set()))}")
            st.rerun()

    
    # ── Export ──
    st.markdown("---")
    st.subheader("Export")

    col_a, col_b = st.columns(2)
    output_name = col_a.text_input("Output filename", value="custom_pore.lammps")
    pressure = col_b.number_input("Pressure (MPa)", value=100, step=10)

    if st.button("Generate data file & input script", type="primary",
                disabled=len(st.session_state.deleted_ids) == 0):

        new_data, out_data_path, out_input_path, input_name = _write_export_outputs(
            pore_dir,
            output_name,
            pressure,
            st.session_state.run_id,
            data,
            st.session_state.deleted_ids,
        )
        st.session_state.run_id += 1
        n_filter_new = len([a for a in new_data['atoms'] if a['type'] == 2])

        st.success(
            f"Wrote **{output_name}** ({new_data['counts']['atoms']} atoms) and **{input_name}**"
        )
        st.caption(f"Data file: {out_data_path}")
        st.caption(f"Input script: {out_input_path}")
        st.balloons()

        # Summary
        type2_new = [a for a in new_data['atoms'] if a['type'] == 2]
        gra2_lo = min(a['id'] for a in type2_new)
        gra2_hi = max(a['id'] for a in type2_new)
        st.code(
            f"Filter atoms: {n_filter_new}  |  "
            f"gra2 range: {gra2_lo}:{gra2_hi}  |  "
            f"Total atoms: {new_data['counts']['atoms']}",
            language=None,
        )
