import json
import os
import random
import signal
import string
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

# ── paths for YOUR installation ──────────────────────────────────────────────
LAMMPS_BUILD = "/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps_build/lammps-22Jul2025/build"
LAMMPS_BIN = f"{LAMMPS_BUILD}/lmp"  # the executable
LAMMPS_LIB = LAMMPS_BUILD  # where liblammps.so lives
LAMMPS_DIR = Path("/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps")
RUNS_DIR = LAMMPS_DIR / "runs"
# ─────────────────────────────────────────────────────────────────────────────


def detect_hardware() -> dict:
    info = {"cores": 1, "gpu": False, "gpu_count": 0}
    try:
        info["cores"] = int(subprocess.check_output(["nproc"]).strip())
    except Exception:
        pass
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        info["gpu_count"] = out.strip().count("GPU ")
        info["gpu"] = info["gpu_count"] > 0
    except Exception:
        pass
    return info



def build_lammps_command(input_file: str, hw: dict, mode: str = "auto") -> List[str]:
    lmp = LAMMPS_BIN
    if mode == "gpu" or (mode == "auto" and hw["gpu"]):
        return [
            "mpirun",
            "-np",
            str(hw["gpu_count"]),
            lmp,
            "-k",
            "on",
            "g",
            str(hw["gpu_count"]),
            "-sf",
            "kk",
            "-in",
            input_file,
        ]
    if mode == "mpi" or (mode == "auto" and hw["cores"] > 2):
        n = max(1, hw["cores"] // 2)
        return ["mpirun", "--bind-to", "core", "-np", str(n), lmp, "-in", input_file]
    raise Exception("Serial mode is not supported. Please use 'mpi' or 'gpu' mode.")


def _env_with_extras() -> Dict[str, str]:
    env_extras = {
        "OMP_NUM_THREADS": "1",
        "LD_LIBRARY_PATH": (
            f"{LAMMPS_LIB}"
            ":/usr/lib/x86_64-linux-gnu"
            ":/usr/lib/x86_64-linux-gnu/openmpi/lib"
            + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        ),
        "CUDA_VISIBLE_DEVICES": "0",
    }
    return {**os.environ, **env_extras}


def _job_id() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "".join(random.choice(string.hexdigits.lower()) for _ in range(4))
    return f"{stamp}_{suffix}"


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _tail_file(path: Path, n_lines: int) -> str:
    if not path.exists():
        return ""
    buf: deque[str] = deque(maxlen=max(1, n_lines))
    with open(path, "r") as f:
        for line in f:
            buf.append(line.rstrip("\n"))
    return "\n".join(buf)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _signal_from_name(name: str) -> int:
    name = (name or "").upper()
    if name == "TERM":
        return signal.SIGTERM
    if name == "KILL":
        return signal.SIGKILL
    if name == "INT":
        return signal.SIGINT
    if name == "HUP":
        return signal.SIGHUP
    raise ValueError(f"Unsupported signal: {name}")


def _write_run_script(
    run_sh: Path,
    cmd: List[str],
    stdout_log: Path,
    stderr_log: Path,
    done_json: Path,
    job_id: str,
    run_dir: Path,
) -> None:
    cmd_str = " ".join(f'"{c}"' for c in cmd)
    script = f"""#!/usr/bin/env bash
set +e
cd "{LAMMPS_DIR}"

{cmd_str} > "{stdout_log}" 2> "{stderr_log}"
rc=$?
export LAMMPS_RC=$rc

python3 - <<PY
import json, time
import os
done = {{
  "job_id": "{job_id}",
  "returncode": int(os.environ.get("LAMMPS_RC", "-1")),
  "success": int(os.environ.get("LAMMPS_RC", "-1")) == 0,
  "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
  "command": {json.dumps(" ".join(cmd))},
  "run_dir": "{run_dir}"
}}
with open("{done_json}", "w") as f:
    json.dump(done, f, indent=2)
PY

python3 - <<'PY'
import os
import smtplib
from email.message import EmailMessage

try:
    host = os.environ.get("SMTP_HOST")
    port = os.environ.get("SMTP_PORT")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    sender = os.environ.get("SMTP_FROM")
    to_addr = os.environ.get("SMTP_TO")
    rc = os.environ.get("LAMMPS_RC", "-1")
    if not all([host, port, user, password, sender, to_addr]):
        raise SystemExit(0)

    msg = EmailMessage()
    msg["Subject"] = f"LAMMPS job {job_id} finished"
    msg["From"] = sender
    msg["To"] = to_addr
    msg.set_content(
        f"Job ID: {job_id}\\n"
        f"Return code: {{rc}}\\n"
        f"Run dir: {run_dir}\\n"
        f"stdout: {stdout_log}\\n"
        f"stderr: {stderr_log}\\n"
        f"done: {done_json}\\n"
    )

    with smtplib.SMTP(host, int(port)) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
except Exception:
    pass
PY

exit $rc
"""
    with open(run_sh, "w") as f:
        f.write(script)
    os.chmod(run_sh, 0o755)


def start_lammps_detached(lammps_file: str, input_file: str, mode: str = "auto") -> dict:
    lf = LAMMPS_DIR / lammps_file
    inf = LAMMPS_DIR / input_file
    if not lf.exists():
        return {"status": "error", "message": f"File not found: {lf}"}
    if not inf.exists():
        return {"status": "error", "message": f"File not found: {inf}"}

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = _job_id()
    run_dir = RUNS_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    meta_json = run_dir / "meta.json"
    done_json = run_dir / "done.json"
    run_sh = run_dir / "run.sh"

    try:
        hw = detect_hardware()
        cmd = build_lammps_command(str(inf.name), hw, mode)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
    _write_run_script(run_sh, cmd, stdout_log, stderr_log, done_json, job_id, run_dir)

    meta = {
        "job_id": job_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(cmd),
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "done_json": str(done_json),
        "pid": None,
    }
    _write_json(meta_json, meta)

    env = _env_with_extras()
    proc = subprocess.Popen(
        ["bash", "-lc", str(run_sh)],
        cwd=str(LAMMPS_DIR),
        env=env,
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    meta["pid"] = proc.pid
    _write_json(meta_json, meta)

    return {
        "status": "started",
        "job_id": job_id,
        "pid": proc.pid,
        "run_dir": str(run_dir),
        "log_paths": {"stdout": str(stdout_log), "stderr": str(stderr_log)},
        "command": " ".join(cmd),
    }


def get_lammps_status(job_id: str) -> dict:
    run_dir = RUNS_DIR / job_id
    if not run_dir.exists():
        return {"state": "missing", "message": f"Job not found: {job_id}"}

    meta_json = run_dir / "meta.json"
    done_json = run_dir / "done.json"
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"

    meta = {}
    if meta_json.exists():
        with open(meta_json, "r") as f:
            meta = json.load(f)

    if done_json.exists():
        with open(done_json, "r") as f:
            done = json.load(f)
        return {
            "state": "finished",
            "pid": meta.get("pid"),
            "returncode": done.get("returncode"),
            "done_json": str(done_json),
            "last_stdout_tail": _tail_file(stdout_log, 200),
            "last_stderr_tail": _tail_file(stderr_log, 200),
        }

    pid = meta.get("pid")
    if isinstance(pid, int) and _pid_alive(pid):
        return {
            "state": "running",
            "pid": pid,
            "last_stdout_tail": _tail_file(stdout_log, 200),
            "last_stderr_tail": _tail_file(stderr_log, 200),
        }

    return {"state": "missing", "pid": pid, "message": "Process not running and no done.json."}


def tail_lammps_log(job_id: str, stream: str = "stdout", n_lines: int = 200) -> str:
    run_dir = RUNS_DIR / job_id
    if not run_dir.exists():
        return ""
    if stream not in ("stdout", "stderr"):
        stream = "stdout"
    log_path = run_dir / ("stdout.log" if stream == "stdout" else "stderr.log")
    return _tail_file(log_path, n_lines)


def stop_lammps_job(job_id: str, signal_name: str = "TERM") -> dict:
    run_dir = RUNS_DIR / job_id
    if not run_dir.exists():
        return {"stopped": False, "message": f"Job not found: {job_id}"}
    meta_json = run_dir / "meta.json"
    if not meta_json.exists():
        return {"stopped": False, "message": "Missing meta.json"}
    with open(meta_json, "r") as f:
        meta = json.load(f)
    pid = meta.get("pid")
    if not isinstance(pid, int):
        return {"stopped": False, "message": "Missing PID"}
    try:
        sig = _signal_from_name(signal_name)
        os.killpg(pid, sig)
        return {"stopped": True, "message": f"Sent {signal_name} to PGID {pid}"}
    except Exception as exc:
        return {"stopped": False, "message": str(exc)}
