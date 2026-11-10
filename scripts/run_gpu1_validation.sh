#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu1_env.sh
q3dg_run_root=${Q3DGSTREAM_GPU_ROOT:-/local/users/$(id -un)}
run_dir=$(mktemp -d "$q3dg_run_root/q3dg-gpu-run.XXXXXX")
printf '%s\n' "$run_dir" > "$q3dg_run_root/latest-gpu-run"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv > "$run_dir/gpu-inventory.csv"
python -m pip freeze > "$run_dir/environment.txt"
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$run_dir/key.pem" -out "$run_dir/cert.pem" -days 7 -subj /CN=localhost -addext subjectAltName=DNS:localhost >/dev/null 2>&1
python -m research.quic.priority_transport.gpu_smoke --scene data/scenes/flame_steak --output "$run_dir/output" --cert "$run_dir/cert.pem" --key "$run_dir/key.pem" 2>&1 | tee "$run_dir/validation.log"
echo "GPU artifacts: $run_dir/output"
