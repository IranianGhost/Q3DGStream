#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
DOCKER=${DOCKER:-docker}
# Reuse existing pinned image, build if absent.
if ! "$DOCKER" image inspect q3dg-priority >/dev/null 2>&1; then
  "$DOCKER" build -t q3dg-priority -f research/quic/priority_transport/Dockerfile research/quic/priority_transport
fi
container=$("$DOCKER" create --cap-add NET_ADMIN q3dg-priority sh -c 'mkdir -p data/real-ntc; mv NTC_*.pth data/real-ntc/; openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 7 -subj /CN=localhost -addext subjectAltName=DNS:localhost >/dev/null 2>&1; python -m research.quic.priority_transport.adaptive_matrix')
output="data/outputs/adaptive-run-$(date +%Y%m%d-%H%M%S)-$$"
finish() {
  status=$1
  trap - EXIT INT TERM
  mkdir -p "$output"
  "$DOCKER" cp "$container:/project/data/outputs/adaptive-matrix/." "$output/" || status=1
  "$DOCKER" logs "$container" > "$output/container.log" 2>&1 || status=1
  "$DOCKER" rm -f "$container" >/dev/null || status=1
  echo "Results: $output"
  exit "$status"
}
trap 'finish $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for file in research q3dgstream.py q3dg_paths.py; do "$DOCKER" cp "$file" "$container:/project/$file"; done
for index in 000000 000001; do
  "$DOCKER" cp "data/scenes/flame_steak/NTCs/NTC_$index.pth" "$container:/project/NTC_$index.pth"
done
"$DOCKER" start -a "$container"
status=$("$DOCKER" inspect --format '{{.State.ExitCode}}' "$container")
exit "$status"
