# runner/bash_runner.py
import os
import shutil
import subprocess
import sys
import threading
from typing import Optional


def _clamp_cores(n_cores: int) -> int:
    # if not isinstance(n_cores, int):
    #     return 1
    return max(1, min(n_cores, 32))


def run_lammps_subprocess(workdir, input_filename, n_cores=16, line_queue=None, timeout=None):
    safe_cores = _clamp_cores(n_cores)
    use_mpirun = shutil.which("mpirun") is not None
    if use_mpirun:
        cmd = ["mpirun", "-np", str(safe_cores), "lmp", "-in", input_filename]
    else:
        cmd = ["lmp", "-in", input_filename]

    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        text=True,
        bufsize=1,  # line-buffered
    )

    output_lines = []

    def reader():
        for line in proc.stdout:
            output_lines.append(line.rstrip())
            if line_queue:
                line_queue.put(line.rstrip())  # push to Streamlit live panel

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("[run_lammps] Timeout expired; process killed", file=sys.stderr)
    t.join()

    result = {
        "returncode": proc.returncode,
        "output": "\n".join(output_lines[-50:]),  # last 50 lines for agent context
        "error": proc.returncode != 0,
        "used_mpirun": use_mpirun,
        "n_cores": safe_cores,
    }
    if result["error"]:
        print(
            f"[run_lammps] Failed with code {proc.returncode}. Last lines:\n{result['output']}",
            file=sys.stderr,
        )
    return result
