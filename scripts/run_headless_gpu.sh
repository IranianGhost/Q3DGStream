#!/usr/bin/env bash
# Run after activating a CUDA-capable environment with tinycudann and rasterizer.
set -euo pipefail
cd "$(dirname "$0")/.."
scene=${1:-data/scenes/flame_steak}
run_dir=$(mktemp -d "${TMPDIR:-/tmp}/q3dg-gpu.XXXXXX")
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$run_dir/key.pem" -out "$run_dir/cert.pem" -days 7 -subj /CN=localhost -addext subjectAltName=DNS:localhost >/dev/null 2>&1
python -m research.quic.priority_transport.gpu_smoke --scene "$scene" --output "$run_dir/output" --cert "$run_dir/cert.pem" --key "$run_dir/key.pem"
echo "GPU report and frames: $run_dir/output"
