import types


def test_extract_selected_ids_from_dict():
    from mcp_implement.script_generation import pore_editor as pe

    event = {
        "selection": {
            "points": [
                {"customdata": 5},
                {"customdata": [7]},
                {"customdata": (9,)},
            ]
        }
    }
    assert pe._extract_selected_ids(event) == {5, 7, 9}


def test_extract_selected_ids_from_object():
    from mcp_implement.script_generation import pore_editor as pe

    point_a = types.SimpleNamespace(customdata=3)
    point_b = types.SimpleNamespace(customdata=[4])
    selection = types.SimpleNamespace(points=[point_a, point_b])
    event = types.SimpleNamespace(selection=selection)

    assert pe._extract_selected_ids(event) == {3, 4}


def test_extract_selected_ids_from_list():
    from mcp_implement.script_generation import pore_editor as pe

    event = [{"customdata": 11}, {"customdata": [12]}]
    assert pe._extract_selected_ids(event) == {11, 12}


def test_write_export_outputs(tmp_path, monkeypatch):
    from mcp_implement.script_generation import pore_editor as pe

    def fake_delete_atoms_and_rewrite(data, deleted_ids):
        data["deleted_ids"] = sorted(list(deleted_ids))
        return data, {}

    def fake_write_lammps_data(data, header_comment=None):
        return f"DATA:{header_comment}"

    def fake_generate_input_script(data_filename, data, pressure_mpa=100, run_id=1):
        return f"INPUT:{data_filename}:{pressure_mpa}:{run_id}"

    monkeypatch.setattr(pe, "delete_atoms_and_rewrite", fake_delete_atoms_and_rewrite)
    monkeypatch.setattr(pe, "write_lammps_data", fake_write_lammps_data)
    monkeypatch.setattr(pe, "generate_input_script", fake_generate_input_script)

    data = {"atoms": [{"id": 1, "type": 2}, {"id": 2, "type": 1}], "counts": {"atoms": 2}}
    deleted_ids = {1}
    monkeypatch.setattr(pe, "_repo_root", tmp_path)

    new_data, out_data_path, out_input_path, input_name = pe._write_export_outputs(
        str(tmp_path), "custom_pore.lammps", 120, 3, data, deleted_ids
    )

    assert new_data["deleted_ids"] == [1]
    assert out_data_path.endswith("custom_pore.lammps")
    assert out_input_path.endswith("custom_pore_120_run3_export.input")
    export_dir = tmp_path / "mcp_implement" / "custom_lammps"
    assert (export_dir / "custom_pore.lammps").read_text().startswith("DATA:")
    assert (export_dir / "custom_pore_120_run3_export.input").read_text().startswith("INPUT:")
    meta = (export_dir / "last_export.json").read_text()
    assert "custom_pore.lammps" in meta
    assert "custom_pore_120_run3_export.input" in meta
    assert "\"run_id\": 3" in meta
