# agent/tools.py
import sys
from ..runner.bash_runner import run_lammps_subprocess
from ..runner.workdir import get_workdir, safe_join, validate_filename
from ..parsers.thermo_parser import parse_thermo
from ..lammps_tools import write_lammps_data, generate_input_script

TOOL_DEFINITIONS = [
    {
        "name": "write_lammps_files",
        "description": "Write the LAMMPS data file and input script to the working directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_filename": {"type": "string"},
                "input_filename": {"type": "string"},
            },
            "required": ["data_filename", "input_filename"],
        },
    },
    {
        "name": "run_lammps",
        "description": "Execute LAMMPS in the bash sandbox. Returns exit code and last 50 log lines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "input_filename": {"type": "string"},
                "n_cores": {"type": "integer", "default": 4},
            },
            "required": ["input_filename"],
        },
    },
    {
        "name": "read_log",
        "description": "Read the last N lines from log.lammps in the working directory.",
        "input_schema": {
            "type": "object",
            "properties": {"n_lines": {"type": "integer", "default": 100}},
        },
    },
    {
        "name": "parse_thermo",
        "description": "Parse thermodynamic data columns from log.lammps. Returns JSON with step, temp, press, pe arrays.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

def dispatch_tool(name, inputs, data):
    workdir = get_workdir()
    if name == "write_lammps_files":
        try:
            data_filename = validate_filename(inputs["data_filename"])
            input_filename = validate_filename(inputs["input_filename"])
            data_path = safe_join(workdir, data_filename)
            input_path = safe_join(workdir, input_filename)

            with open(data_path, "w") as f:
                f.write(write_lammps_data(data))

            script = generate_input_script(data_filename, data)
            with open(input_path, "w") as f:
                f.write(script)

            return {
                "data_path": data_path,
                "input_path": input_path,
            }
        except Exception as exc:
            print(f"[write_lammps_files] {exc}", file=sys.stderr)
            return {"error": str(exc)}
    elif name == "run_lammps":
        try:
            inputs["input_filename"] = validate_filename(inputs["input_filename"])
        except Exception as exc:
            print(f"[run_lammps] {exc}", file=sys.stderr)
            return {"error": str(exc)}
        return run_lammps_subprocess(
            workdir, inputs["input_filename"],
            n_cores=inputs.get("n_cores", 4)
        )
    elif name == "read_log":
        log_path = safe_join(workdir, "log.lammps")
        with open(log_path) as f:
            return "\n".join(f.readlines()[-inputs.get("n_lines", 100):])
    elif name == "parse_thermo":
        return parse_thermo(safe_join(workdir, "log.lammps"))
