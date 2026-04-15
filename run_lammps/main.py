import asyncio
import subprocess
import os
from mcp.server import FastMCP

try:
    from .tools import (
        LAMMPS_DIR,
        LAMMPS_LIB,
        build_lammps_command,
        detect_hardware,
        get_lammps_status,
        start_lammps_detached,
        stop_lammps_job,
        tail_lammps_log,
    )
except Exception:
    from tools import (  # type: ignore
        LAMMPS_DIR,
        LAMMPS_LIB,
        build_lammps_command,
        detect_hardware,
        
        get_lammps_status,
        start_lammps_detached,
        stop_lammps_job,
        tail_lammps_log,
    )

mcp = FastMCP("LAMMPS Simulation Server")


@mcp.tool(name="run_lammps")
async def run_lammps(
    lammps_file: str,
    input_file: str,
    mode: str = "auto",
    timeout: int = 3600
) -> dict:
    """
    Run a LAMMPS simulation.
    lammps_file: .lammps data file name (in custom_lammps dir)
    input_file:  .input script file name (in custom_lammps dir)
    mode:        execution mode — auto, serial, mpi, gpu
    timeout:     max seconds to run (default 1 hour)
    """
    lf  = LAMMPS_DIR / lammps_file
    inf = LAMMPS_DIR / input_file

    if not lf.exists():
        return {"status": "error", "message": f"File not found: {lf}"}
    if not inf.exists():
        return {"status": "error", "message": f"File not found: {inf}"}

    hw  = detect_hardware()
    cmd = build_lammps_command(str(inf.name), hw, mode)

    # ── CHANGE 2: correct LD_LIBRARY_PATH for your installation ──────────────
    env_extras = {
        "OMP_NUM_THREADS": "1",
        "LD_LIBRARY_PATH": (
            f"{LAMMPS_LIB}"                          # liblammps.so
            ":/usr/lib/x86_64-linux-gnu"             # libcudart.so
            ":/usr/lib/x86_64-linux-gnu/openmpi/lib" # libmpi.so
            + ":" + os.environ.get("LD_LIBRARY_PATH", "")  # keep existing
        ),
        # ── CHANGE 3: tell lmp where to find CUDA at runtime ─────────────────
        "CUDA_VISIBLE_DEVICES": "0",                 # use GPU 0 (your 2080 Ti)
    }
    env = {**os.environ, **env_extras}

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(LAMMPS_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return {
            "status":     "success" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "command":    " ".join(cmd),
            "hardware":   hw,
            "stdout_tail": stdout.decode()[-3000:],
            "stderr_tail": stderr.decode()[-1000:],
        }
    except asyncio.TimeoutError:
        proc.kill()
        return {"status": "timeout", "message": f"Killed after {timeout}s"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool(name="start_lammps_detached")
def start_lammps_detached_tool(
    lammps_file: str,
    input_file: str,
    mode: str = "auto",
) -> dict:
    return start_lammps_detached(lammps_file, input_file, mode)


@mcp.tool(name="get_lammps_status")
def get_lammps_status_tool(job_id: str) -> dict:
    return get_lammps_status(job_id)


@mcp.tool(name="tail_lammps_log")
def tail_lammps_log_tool(job_id: str, stream: str = "stdout", n_lines: int = 200) -> str:
    return tail_lammps_log(job_id, stream=stream, n_lines=n_lines)


@mcp.tool(name="stop_lammps_job")
def stop_lammps_job_tool(job_id: str, signal: str = "TERM") -> dict:
    return stop_lammps_job(job_id, signal_name=signal)


if __name__ == "__main__":
    mcp.run()

# LAMMPS_DIR = Path("/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps")

# def detect_hardware() -> dict:
#     """Detect available CPU cores and GPU."""
#     info = {"cores": 1, "gpu": False, "gpu_count": 0}
#     try:
#         info["cores"] = int(subprocess.check_output(["nproc"]).strip())
#     except Exception:
#         pass
#     try:
#         out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
#         info["gpu_count"] = out.strip().count("GPU ")
#         info["gpu"] = info["gpu_count"] > 0
#     except Exception:
#         pass
#     return info

# def build_lammps_command(input_file: str, hw: dict, mode: str = "auto") -> list[str]:
#     """Build the optimal mpirun/lmp command."""
#     lmp = shutil.which("lmp") or shutil.which("lmp_mpi") or "lmp"
    
#     if mode == "gpu" or (mode == "auto" and hw["gpu"]):
#         # KOKKOS GPU mode — 1 MPI rank per GPU
#         return ["mpirun", "-np", str(hw["gpu_count"]),
#                 lmp, "-k", "on", "g", str(hw["gpu_count"]),
#                 "-sf", "kk", "-in", input_file]
#     elif mode == "mpi" or (mode == "auto" and hw["cores"] > 2):
#         n = max(1, hw["cores"])   # conservative: half the cores
#         return ["mpirun", "--bind-to", "core",
#                 "-np", str(n), lmp, "-in", input_file]
#     # else:
#     #     return [lmp, "-in", input_file]

# @mcp.tool(name="run_lammps")
# async def run_lammps(
#     lammps_file: str,
#     input_file: str,
#     mode: str = "auto",   # "auto" | "serial" | "mpi" | "gpu"
#     timeout: int = 3600
# ) -> dict:
#     """
#     Run a LAMMPS simulation.
#     lammps_file: .lammps data file name (in custom_lammps dir)
#     input_file:  .input script file name (in custom_lammps dir)
#     mode:        execution mode — auto, serial, mpi, gpu
#     timeout:     max seconds to run (default 1 hour)
#     """
#     lf = LAMMPS_DIR / lammps_file
#     inf = LAMMPS_DIR / input_file

#     if not lf.exists():
#         return {"status": "error", "message": f"File not found: {lf}"}
#     if not inf.exists():
#         return {"status": "error", "message": f"File not found: {inf}"}

#     hw = detect_hardware()
#     cmd = build_lammps_command(str(inf.name), hw, mode)

#     env_extras = {
#         "OMP_NUM_THREADS": "1",                        # avoid oversubscription
#         "LD_LIBRARY_PATH": "/usr/local/lib:/usr/lib",  # adjust to your install
#     }
#     import os
#     env = {**os.environ, **env_extras}

#     try:
#         proc = await asyncio.create_subprocess_exec(
#             *cmd,
#             cwd=str(LAMMPS_DIR),
#             stdout=asyncio.subprocess.PIPE,
#             stderr=asyncio.subprocess.PIPE,
#             env=env
#         )
#         stdout, stderr = await asyncio.wait_for(
#             proc.communicate(), timeout=timeout
#         )
#         return {
#             "status": "success" if proc.returncode == 0 else "error",
#             "returncode": proc.returncode,
#             "command": " ".join(cmd),
#             "hardware": hw,
#             "stdout_tail": stdout.decode()[-3000:],   # last 3k chars
#             "stderr_tail": stderr.decode()[-1000:],
#         }
#     except asyncio.TimeoutError:
#         proc.kill()
#         return {"status": "timeout", "message": f"Killed after {timeout}s"}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# if __name__ == "__main__":
#     mcp.run()
