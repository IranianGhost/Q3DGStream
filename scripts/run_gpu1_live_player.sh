#!/usr/bin/env bash
# Foreground GUI player; keep alive using nohup or your normal session manager.
set -euo pipefail
cd "$(dirname "$0")/.."
root="${Q3DGSTREAM_GPU_ROOT:-/local/users/$(id -un)}/q3dg-desktop"
if [[ -f "$root/player.pid" ]] && kill -0 "$(cat "$root/player.pid")" 2>/dev/null; then
    echo 'Player already running; reconnect to the current desktop.'
    exit 0
fi
# Check before replacing the certificate/cache pointer of an active receiver.
python3 - <<'PORT'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener:
    try:
        listener.bind(('127.0.0.1', 5003))
    except OSError:
        raise SystemExit('QUIC port 5003 is in use; reconnect or stop its owner first.')
PORT
source scripts/gpu1_env.sh
export DISPLAY=:9 XAUTHORITY="$root/session/Xauthority"
export LD_LIBRARY_PATH="$root/root/usr/lib:${LD_LIBRARY_PATH:-}"
export Q3DG_GLFW_CONTEXT_API=egl
export PYOPENGL_PLATFORM=egl
export NTC_DEBUG=0 NTC_DEBUG_DXYZ=0 LOOP_PLAYBACK=1
export PYTHONUNBUFFERED=1
umask 077
run_dir=$(mktemp -d "$root/live.XXXXXX")
printf '%s\n' "$run_dir" > "$root/latest-live"
printf '%s\n' "$$" > "$root/player.pid"
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$run_dir/key.pem" \
    -out "$run_dir/cert.pem" -days 7 -subj /CN=localhost \
    -addext subjectAltName=DNS:localhost >/dev/null 2>&1
python - "$run_dir/camera.json" <<'CAMERA'
import json, sys
from pathlib import Path
report=json.loads(Path('research/quic/priority_transport/evidence/gpu-server/REPORT.json').read_text())
camera=report['camera']
camera.update(up=[0,-1,0],yaw=-1.5707963267948966,pitch=0,zfar=250)
Path(sys.argv[1]).write_text(json.dumps(camera))
CAMERA
export FIXED_CAMERA_JSON="$run_dir/camera.json" LOCK_CAMERA=0
# Initial scene is NOT autoloaded: received QUIC files drive LivePriorityState.
exec "$root/root/opt/VirtualGL/bin/vglrun" -d "${Q3DG_VGL_DEVICE:-egl0}" \
    python main.py --priority_listen 5003 --priority_cache "$run_dir/cache" \
    --priority_cert "$run_dir/cert.pem" --priority_key "$run_dir/key.pem" \
    --frames "${Q3DG_DEMO_FRAMES:-31}" --video_fps 30 --autoplay
