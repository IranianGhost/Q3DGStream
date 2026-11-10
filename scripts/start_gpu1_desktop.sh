#!/usr/bin/env bash
# Prepared user-local TurboVNC/VirtualGL installation; no sudo required.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu1_env.sh
export Q3DG_DESKTOP_ROOT="${Q3DGSTREAM_GPU_ROOT:-/local/users/$(id -un)}/q3dg-desktop"
export DISPLAY=:9
export XAUTHORITY="$Q3DG_DESKTOP_ROOT/session/Xauthority"
umask 077
mkdir -p "$Q3DG_DESKTOP_ROOT/session"
python - <<'AUTH'
from pathlib import Path
import os, secrets, subprocess
root=Path(os.environ['Q3DG_DESKTOP_ROOT'])
s=root/'session'
password=s/'password'
if not password.exists():
    password.write_text(secrets.token_urlsafe(6))
    password.chmod(0o600)
if not (s/'passwd').exists():
    result=subprocess.run([str(root/'root/opt/TurboVNC/bin/vncpasswd'),'-f'],input=password.read_bytes()+b'\n',stdout=subprocess.PIPE,check=True)
    (s/'passwd').write_bytes(result.stdout)
    (s/'passwd').chmod(0o600)
if not (s/'Xauthority').exists():
    subprocess.run(['xauth','-f',str(s/'Xauthority'),'add',':9','.',secrets.token_hex(16)],check=True)
AUTH
start_once() {
    local name="$1"; shift
    local pid_file="$Q3DG_DESKTOP_ROOT/session/$name.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "$name already running"; return
    fi
    nohup "$@" > "$Q3DG_DESKTOP_ROOT/session/$name.log" 2>&1 < /dev/null &
    echo $! > "$pid_file"
}
start_once desktop "$Q3DG_DESKTOP_ROOT/root/opt/TurboVNC/bin/Xvnc" :9 \
    -auth "$XAUTHORITY" -nolisten tcp -localhost -rfbport 5909 \
    -rfbauth "$Q3DG_DESKTOP_ROOT/session/passwd" -securitytypes VNC \
    -dridir /usr/lib/x86_64-linux-gnu/dri -geometry 1440x900 -depth 24 \
    -desktop 'Q3DGStream GPU'
# Wait for an authenticated display, not an arbitrary startup delay.
for attempt in {1..100}; do
    if xdpyinfo >/dev/null 2>&1; then break; fi
    sleep 0.1
done
xdpyinfo >/dev/null
start_once openbox env \
    LD_LIBRARY_PATH="$Q3DG_DESKTOP_ROOT/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
    XDG_DATA_DIRS="$Q3DG_DESKTOP_ROOT/root/usr/share:/usr/share" \
    "$Q3DG_DESKTOP_ROOT/root/usr/bin/openbox" \
    --config-file "$Q3DG_DESKTOP_ROOT/root/etc/xdg/openbox/rc.xml"
start_once websockify websockify --web="$Q3DG_DESKTOP_ROOT/novnc" \
    127.0.0.1:6081 127.0.0.1:5909
echo 'Desktop :9; VNC 127.0.0.1:5909; browser proxy 127.0.0.1:6081'
