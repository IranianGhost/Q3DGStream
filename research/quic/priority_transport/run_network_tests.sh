#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
docker build -t q3dg-priority -f research/quic/priority_transport/Dockerfile research/quic/priority_transport
container=$(docker create --cap-add NET_ADMIN q3dg-priority sh -c 'openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 7 -subj /CN=localhost -addext subjectAltName=DNS:localhost >/dev/null 2>&1 && python -m research.quic.priority_transport.matrix && python -m research.quic.priority_transport.analyze')
output="data/outputs/priority-run-$(date +%Y%m%d-%H%M%S)-$$"
finish() {
  status=$1
  trap - EXIT INT TERM
  mkdir -p "$output"
  docker cp "$container:/project/data/outputs/priority-matrix/." "$output/" || status=1
  docker logs "$container" > "$output/container.log" 2>&1 || status=1
  docker rm -f "$container" >/dev/null || status=1
  echo "Results: $output"
  exit "$status"
}
trap 'finish $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for file in research q3dgstream.py q3dg_paths.py; do docker cp "$file" "$container:/project/$file"; done
docker start -a "$container"
status=$(docker inspect --format '{{.State.ExitCode}}' "$container")
exit "$status"
