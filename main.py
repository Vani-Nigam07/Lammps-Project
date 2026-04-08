"""FastMCP server for LAMMPS pore tools."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import socket
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import FastMCP

try:
    from mcp_implement.lammps_tools import (
        parse_lammps_data,
        reconstruct_full_filter,
        delete_atoms_and_rewrite,
        write_lammps_data,
        generate_input_script,
    )
    from mcp_implement.parsers.thermo_parser import parse_thermo
    # from mcp_implement.runner.bash_runner import run_lammps_subprocess
    from mcp_implement.runner.workdir import get_workdir, safe_join, validate_filename
except ModuleNotFoundError:
    # Allow running from inside the mcp_implement/ directory
    from lammps_tools import (
        parse_lammps_data,
        reconstruct_full_filter,
        delete_atoms_and_rewrite,
        write_lammps_data,
        generate_input_script,
    )
    from parsers.thermo_parser import parse_thermo
    # from runner.bash_runner import run_lammps_subprocess
    from runner.workdir import get_workdir, safe_join, validate_filename

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("autoMD")

server = FastMCP(
    name="autoMD",
    log_level="INFO",
)

_streamlit_proc: Optional[subprocess.Popen] = None
_streamlit_stderr_tail: deque[str] = deque(maxlen=20)
_streamlit_stderr_thread: Optional[threading.Thread] = None


def _read_streamlit_stderr(pipe: Any) -> None:
    try:
        for line in pipe:
            _streamlit_stderr_tail.append(line.rstrip("\n"))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _wait_for_port(host: str, port: int, timeout_s: float = 8.0) -> bool:
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except Exception as exc:
            last_err = exc
            time.sleep(0.2)
    if last_err:
        _streamlit_stderr_tail.append(f"[healthcheck] {last_err}")
    return False


def _ensure_workdir() -> str:
    return get_workdir()


@server.tool(name="parse_lammps_data")
def parse_lammps_data_tool(filename: str) -> Dict[str, Any]:
    """Parse a LAMMPS data file from the fixed workdir."""
    try:
        workdir = _ensure_workdir()
        fname = validate_filename(filename)
        path = safe_join(workdir, fname)
        return parse_lammps_data(path)
    except Exception as exc:
        print(f"[parse_lammps_data] {exc}", file=sys.stderr)
        return {"error": str(exc)}


@server.tool(name="reconstruct_full_filter")
def reconstruct_full_filter_tool(data: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild a complete filter membrane (type 2) from the piston (type 1)."""
    try:
        return reconstruct_full_filter(data)
    except Exception as exc:
        print(f"[reconstruct_full_filter] {exc}", file=sys.stderr)
        return {"error": str(exc)}


@server.tool(name="delete_atoms_and_rewrite")
def delete_atoms_and_rewrite_tool(data: Dict[str, Any], ids_to_delete: List[int]) -> Dict[str, Any]:
    """Remove atoms by ID, prune bonds/angles, and renumber everything."""
    try:
        new_data, id_map = delete_atoms_and_rewrite(data, ids_to_delete)
        return {"data": new_data, "id_map": id_map}
    except Exception as exc:
        print(f"[delete_atoms_and_rewrite] {exc}", file=sys.stderr)
        return {"error": str(exc)}


@server.tool(name="write_lammps_data")
def write_lammps_data_tool(data: Dict[str, Any], header_comment: Optional[str] = None) -> str:
    """Serialize parsed data back to LAMMPS data file format."""
    try:
        return write_lammps_data(data, header_comment=header_comment)
    except Exception as exc:
        print(f"[write_lammps_data] {exc}", file=sys.stderr)
        return {"error": str(exc)}


@server.tool(name="generate_input_script")
def generate_input_script_tool(
    data_filename: str,
    data: Dict[str, Any],
    pressure_mpa: float = 100,
    run_id: int = 1,
) -> str:
    """Generate a matching LAMMPS input script for the new data file."""
    try:
        return generate_input_script(data_filename, data, pressure_mpa=pressure_mpa, run_id=run_id)
    except Exception as exc:
        print(f"[generate_input_script] {exc}", file=sys.stderr)
        return {"ERROR": str(exc)}


@server.tool(name="write_lammps_files")
def write_lammps_files(
    data_filename: str,
    input_filename: str,
    data: Dict[str, Any],
    header_comment: Optional[str] = None,
    pressure_mpa: float = 100,
    run_id: int = 1,
) -> Dict[str, Any]:
    """Write LAMMPS data and input script to the fixed workdir."""
    try:
        workdir = _ensure_workdir()
        data_filename = validate_filename(data_filename)
        input_filename = validate_filename(input_filename)
        data_path = safe_join(workdir, data_filename)
        input_path = safe_join(workdir, input_filename)

        with open(data_path, "w") as f:
            f.write(write_lammps_data(data, header_comment=header_comment))

        script = generate_input_script(
            data_filename, data, pressure_mpa=pressure_mpa, run_id=run_id
        )
        with open(input_path, "w") as f:
            f.write(script)

        return {"data_path": data_path, "input_path": input_path}
    except Exception as exc:
        print(f"[write_lammps_files] {exc}", file=sys.stderr)
        return {"error": str(exc)}


# @server.tool(name="run_lammps")
# def run_lammps(input_filename: str, n_cores: int = 16, timeout_sec: Optional[int] = None) -> Dict[str, Any]:
#     """Execute LAMMPS in the bash sandbox. Returns exit code and last 50 log lines."""
#     try:
#         workdir = _ensure_workdir()
#         input_filename = validate_filename(input_filename)
#         return run_lammps_subprocess(
#             workdir, input_filename, n_cores=n_cores, timeout=timeout_sec
#         )
#     except Exception as exc:
#         print(f"[run_lammps] {exc}", file=sys.stderr)
#         return {"error": str(exc)}


@server.tool(name="read_log")
def read_log(n_lines: int = 100) -> str:
    """Read the last N lines from log.lammps in the fixed workdir."""
    try:
        workdir = _ensure_workdir()
        log_path = safe_join(workdir, "log.lammps")
        with open(log_path) as f:
            return "".join(f.readlines()[-n_lines:])
    except Exception as exc:
        print(f"[read_log] {exc}", file=sys.stderr)
        return ""


@server.tool(name="parse_thermo")
def parse_thermo_tool() -> Dict[str, Any]:
    """Parse thermodynamic data columns from log.lammps."""
    try:
        workdir = _ensure_workdir()
        log_path = safe_join(workdir, "log.lammps")
        return parse_thermo(log_path)
    except Exception as exc:
        print(f"[parse_thermo] {exc}", file=sys.stderr)
        return {"error": str(exc)}


@server.tool(name="launch_streamlit_app")
def launch_streamlit_app() -> Dict[str, Any]:
    """Launch the Streamlit pore editor app via a fixed command and path."""
    global _streamlit_proc
    try:
        port = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
        address = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
        display_host = "localhost" if address in {"0.0.0.0", "::"} else address
        url = f"http://{display_host}:{port}"

        if _streamlit_proc and _streamlit_proc.poll() is None:
            return {"status": "already_running", "pid": _streamlit_proc.pid, "url": url}

        repo_root = Path(__file__).resolve().parents[1]
        app_path = repo_root / "mcp_implement" / "script_generation" / "pore_editor.py"
        cmd = [
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(port),
            "--server.address",
            address,
        ]
        _streamlit_stderr_tail.clear()
        _streamlit_proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if _streamlit_proc.stderr is not None:
            global _streamlit_stderr_thread
            _streamlit_stderr_thread = threading.Thread(
                target=_read_streamlit_stderr,
                args=(_streamlit_proc.stderr,),
                daemon=True,
            )
            _streamlit_stderr_thread.start()

        # Health check: connect to the bound port.
        check_host = "127.0.0.1" if address == "0.0.0.0" else ("::1" if address == "::" else address)
        if _wait_for_port(check_host, port):
            return {"status": "started", "pid": _streamlit_proc.pid, "url": url}

        # Health check failed; see if process exited.
        if _streamlit_proc.poll() is not None:
            return {
                "error": "streamlit exited during startup",
                "stderr_tail": list(_streamlit_stderr_tail),
            }
        return {
            "error": "streamlit did not become reachable before timeout",
            "stderr_tail": list(_streamlit_stderr_tail),
        }
    except Exception as exc:
        print(f"[launch_streamlit_app] {exc}", file=sys.stderr)
        return {"error": str(exc)}


@server.tool(name="get_last_export_paths")
def get_last_export_paths() -> Dict[str, Any]:
    """Read last export paths written by the Streamlit app."""
    try:
        repo_root = Path(__file__).resolve().parents[1]
        meta_path = repo_root / "mcp_implement" / "custom_lammps" / "last_export.json"
        if not meta_path.exists():
            return {"error": "no export metadata found"}
        with open(meta_path) as f:
            return json.load(f)
    except Exception as exc:
        print(f"[get_last_export_paths] {exc}", file=sys.stderr)
        return {"error": str(exc)}


def main() -> None:
    try:
        # Default to stdio for Gemini CLI usage.
        transport = os.getenv("MCP_TRANSPORT", "").strip().lower() or "stdio"
        server.run(transport=transport)
    except Exception as exc:
        print(f"[mcp-server] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
