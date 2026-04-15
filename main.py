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
import base64
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
_streamlit_port: Optional[int] = None
_streamlit_started_at: Optional[float] = None

_streamlit_v2_proc: Optional[subprocess.Popen] = None
_streamlit_v2_stderr_tail: deque[str] = deque(maxlen=20)
_streamlit_v2_stderr_thread: Optional[threading.Thread] = None
_streamlit_v2_port: Optional[int] = None
_streamlit_v2_started_at: Optional[float] = None


def _read_streamlit_stderr(pipe: Any, tail: deque[str]) -> None:
    try:
        for line in pipe:
            tail.append(line.rstrip("\n"))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _wait_for_port(
    host: str,
    port: int,
    timeout_s: float = 8.0,
    stderr_tail: Optional[deque[str]] = None,
) -> bool:
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
        tail = stderr_tail if stderr_tail is not None else _streamlit_stderr_tail
        tail.append(f"[healthcheck] {last_err}")
    return False


def _port_available(address: str, port: int) -> bool:
    bind_host = "127.0.0.1" if address == "localhost" else address
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((bind_host, port))
        return True
    except Exception:
        return False


def _pick_port(address: str, preferred: int) -> int:
    if _port_available(address, preferred):
        return preferred
    for offset in range(1, 20):
        port = preferred + offset
        if _port_available(address, port):
            return port
    raise RuntimeError(f"no free port near {preferred}")


def _display_host(address: str) -> str:
    external_host = os.getenv("STREAMLIT_EXTERNAL_HOST")
    if external_host:
        return external_host
    return "localhost" if address in {"0.0.0.0", "::"} else address


def _healthcheck_host(address: str) -> str:
    if address == "0.0.0.0":
        return "127.0.0.1"
    if address == "::":
        return "::1"
    if address == "localhost":
        return "127.0.0.1"
    return address


def _launch_streamlit_process(
    app_path: Path,
    address: str,
    preferred_port: int,
    stderr_tail: deque[str],
) -> tuple[subprocess.Popen, int, Optional[threading.Thread]]:
    port = _pick_port(address, preferred_port)
    cmd = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        address,
    ]
    stderr_tail.clear()
    proc = subprocess.Popen(
        cmd,
        cwd=str(app_path.parents[2]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    thread = None
    if proc.stderr is not None:
        thread = threading.Thread(
            target=_read_streamlit_stderr,
            args=(proc.stderr, stderr_tail),
            daemon=True,
        )
        thread.start()
    return proc, port, thread


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
    global _streamlit_proc, _streamlit_port, _streamlit_stderr_thread, _streamlit_started_at
    try:
        preferred_port = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
        address = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
        display_host = _display_host(address)

        if _streamlit_proc and _streamlit_proc.poll() is None:
            port = _streamlit_port or preferred_port
            url = f"http://{display_host}:{port}"
            return {"status": "already_running", "pid": _streamlit_proc.pid, "url": url, "port": port}

        repo_root = Path(__file__).resolve().parents[1]
        app_path = repo_root / "mcp_implement" / "app" / "pore_editor.py"
        _streamlit_proc, port, _streamlit_stderr_thread = _launch_streamlit_process(
            app_path, address, preferred_port, _streamlit_stderr_tail
        )
        _streamlit_port = port
        _streamlit_started_at = time.time()

        # Health check: connect to the bound port.
        check_host = _healthcheck_host(address)
        if _wait_for_port(check_host, port, stderr_tail=_streamlit_stderr_tail):
            url = f"http://{display_host}:{port}"
            return {"status": "started", "pid": _streamlit_proc.pid, "url": url, "port": port}

        # Health check failed; see if process exited.
        if _streamlit_proc.poll() is not None:
            return {
                "error": "streamlit exited during startup",
                "stderr_tail": list(_streamlit_stderr_tail),
                "port": port,
            }
        return {
            "error": "streamlit did not become reachable before timeout",
            "stderr_tail": list(_streamlit_stderr_tail),
            "port": port,
        }
    except Exception as exc:
        print(f"[launch_streamlit_app] {exc}", file=sys.stderr)
        return {"error": str(exc)}


@server.tool(name="upload_lammps_file")
def upload_lammps_file(
    filename: str,
    content: Optional[str] = None,
    content_b64: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload a .lammps file into the custom_lammps directory used by pore_editor_v2.py."""
    try:
        fname = validate_filename(filename)
        if not fname.lower().endswith(".lammps"):
            raise ValueError("Only .lammps files are accepted")
        repo_root = Path(__file__).resolve().parents[1]
        pore_dir = repo_root / "mcp_implement" / "custom_lammps"
        pore_dir.mkdir(parents=True, exist_ok=True)
        if content_b64 is not None:
            data = base64.b64decode(content_b64)
        elif content is not None:
            data = content.encode("utf-8")
        else:
            raise ValueError("Provide content or content_b64")
        path = safe_join(str(pore_dir), fname)
        with open(path, "wb") as f:
            f.write(data)
        return {"status": "ok", "path": path, "filename": fname, "bytes": len(data)}
    except Exception as exc:
        print(f"[upload_lammps_file] {exc}", file=sys.stderr)
        return {"error": str(exc)}


@server.tool(name="launch_streamlit_v2_app")
def launch_streamlit_v2_app() -> Dict[str, Any]:
    """Launch the Streamlit pore editor v2 app."""
    global _streamlit_v2_proc, _streamlit_v2_port, _streamlit_v2_stderr_thread, _streamlit_v2_started_at
    try:
        preferred_port = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
        address = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
        display_host = _display_host(address)

        if _streamlit_v2_proc and _streamlit_v2_proc.poll() is None:
            port = _streamlit_v2_port or preferred_port
            url = f"http://{display_host}:{port}"
            return {"status": "already_running", "pid": _streamlit_v2_proc.pid, "url": url, "port": port}

        repo_root = Path(__file__).resolve().parents[1]
        app_path = repo_root / "mcp_implement" / "ui" / "pore_editor_v2.py"
        _streamlit_v2_proc, port, _streamlit_v2_stderr_thread = _launch_streamlit_process(
            app_path, address, preferred_port, _streamlit_v2_stderr_tail
        )
        _streamlit_v2_port = port
        _streamlit_v2_started_at = time.time()

        check_host = _healthcheck_host(address)
        if _wait_for_port(check_host, port, stderr_tail=_streamlit_v2_stderr_tail):
            url = f"http://{display_host}:{port}"
            return {"status": "started", "pid": _streamlit_v2_proc.pid, "url": url, "port": port}

        if _streamlit_v2_proc.poll() is not None:
            return {
                "error": "streamlit exited during startup",
                "stderr_tail": list(_streamlit_v2_stderr_tail),
                "port": port,
            }
        return {
            "error": "streamlit did not become reachable before timeout",
            "stderr_tail": list(_streamlit_v2_stderr_tail),
            "port": port,
        }
    except Exception as exc:
        print(f"[launch_streamlit_v2_app] {exc}", file=sys.stderr)
        return {"error": str(exc)}


@server.tool(name="get_streamlit_v2_status")
def get_streamlit_v2_status() -> Dict[str, Any]:
    """Return current status for the v2 Streamlit app."""
    address = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    display_host = _display_host(address)
    alive = _streamlit_v2_proc is not None and _streamlit_v2_proc.poll() is None
    port = _streamlit_v2_port
    url = f"http://{display_host}:{port}" if port else None
    started_at = _streamlit_v2_started_at
    uptime_s = (time.time() - started_at) if started_at else None
    return {
        "status": "running" if alive else "stopped",
        "pid": _streamlit_v2_proc.pid if alive else None,
        "port": port,
        "url": url,
        "started_at": started_at,
        "uptime_s": uptime_s,
        "stderr_tail": list(_streamlit_v2_stderr_tail),
    }


@server.tool(name="restart_streamlit_v2_app")
def restart_streamlit_v2_app() -> Dict[str, Any]:
    """Restart the v2 Streamlit app."""
    global _streamlit_v2_proc
    if _streamlit_v2_proc and _streamlit_v2_proc.poll() is None:
        _streamlit_v2_proc.terminate()
        try:
            _streamlit_v2_proc.wait(timeout=5)
        except Exception:
            _streamlit_v2_proc.kill()
            _streamlit_v2_proc.wait(timeout=2)
    _streamlit_v2_proc = None
    return launch_streamlit_v2_app()


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
