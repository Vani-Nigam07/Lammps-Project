#!/usr/bin/env bash
set +e
cd "/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps"

"mpirun" "-np" "1" "/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps_build/lammps-22Jul2025/build/lmp" "-k" "on" "g" "1" "-sf" "kk" "-in" "trial_2_MoS2_120_1.input" > "/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260411_123747_8138/stdout.log" 2> "/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260411_123747_8138/stderr.log"
rc=$?
export LAMMPS_RC=$rc

python3 - <<PY
import json, time
import os
done = {
  "job_id": "20260411_123747_8138",
  "returncode": int(os.environ.get("LAMMPS_RC", "-1")),
  "success": int(os.environ.get("LAMMPS_RC", "-1")) == 0,
  "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
  "command": "mpirun -np 1 /media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps_build/lammps-22Jul2025/build/lmp -k on g 1 -sf kk -in trial_2_MoS2_120_1.input",
  "run_dir": "/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260411_123747_8138"
}
with open("/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260411_123747_8138/done.json", "w") as f:
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
    msg["Subject"] = f"LAMMPS job 20260411_123747_8138 finished"
    msg["From"] = sender
    msg["To"] = to_addr
    msg.set_content(
        f"Job ID: 20260411_123747_8138\n"
        f"Return code: {rc}\n"
        f"Run dir: /media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260411_123747_8138\n"
        f"stdout: /media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260411_123747_8138/stdout.log\n"
        f"stderr: /media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260411_123747_8138/stderr.log\n"
        f"done: /media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/lammps/mcp_implement/custom_lammps/runs/20260411_123747_8138/done.json\n"
    )

    with smtplib.SMTP(host, int(port)) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
except Exception:
    pass
PY

exit $rc
