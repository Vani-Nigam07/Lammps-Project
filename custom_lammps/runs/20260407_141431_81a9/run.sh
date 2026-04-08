#!/usr/bin/env bash
set +e
cd "/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps"

"mpirun" "--bind-to" "core" "-np" "8" "/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps_build/lammps-22Jul2025/build/lmp" "-in" "code_2_100_run1_export.input" > "/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260407_141431_81a9/stdout.log" 2> "/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260407_141431_81a9/stderr.log"
rc=$?
export LAMMPS_RC=$rc

python3 - <<PY
import json, time
import os
done = {
  "job_id": "20260407_141431_81a9",
  "returncode": int(os.environ.get("LAMMPS_RC", "-1")),
  "success": int(os.environ.get("LAMMPS_RC", "-1")) == 0,
  "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
  "command": "mpirun --bind-to core -np 8 /media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps_build/lammps-22Jul2025/build/lmp -in code_2_100_run1_export.input",
  "run_dir": "/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260407_141431_81a9"
}
with open("/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260407_141431_81a9/done.json", "w") as f:
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
    msg["Subject"] = f"LAMMPS job 20260407_141431_81a9 finished"
    msg["From"] = sender
    msg["To"] = to_addr
    msg.set_content(
        f"Job ID: 20260407_141431_81a9\n"
        f"Return code: {rc}\n"
        f"Run dir: /media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260407_141431_81a9\n"
        f"stdout: /media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260407_141431_81a9/stdout.log\n"
        f"stderr: /media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260407_141431_81a9/stderr.log\n"
        f"done: /media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260407_141431_81a9/done.json\n"
    )

    with smtplib.SMTP(host, int(port)) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
except Exception:
    pass
PY

exit $rc
