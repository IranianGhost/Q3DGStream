#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
container=$(docker create --cap-add NET_ADMIN q3dg-priority sh -c 'mkdir -p data/real-ntc; mv NTC_*.pth data/real-ntc/; openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 2 -subj /CN=localhost -addext subjectAltName=DNS:localhost >/dev/null 2>&1; python -m research.quic.priority_transport.playback_matrix && python -m research.quic.priority_transport.playback_followup')
output="data/outputs/playback-run-$(date +%Y%m%d-%H%M%S)-$$"
finish() {
 status=$1
 trap - EXIT INT TERM
 mkdir -p "$output/main" "$output/followup"
 docker cp "$container:/project/data/outputs/playback-matrix/." "$output/main/" || status=1
 docker cp "$container:/project/data/outputs/playback-followup/." "$output/followup/" || status=1
 docker logs "$container" > "$output/container.log" 2>&1 || status=1
 docker rm -f "$container" >/dev/null || status=1
 echo "Results: $output"
 exit "$status"
}
trap 'finish $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
docker cp research "$container:/project/"
for n in 000000 000001; do docker cp "data/scenes/flame_steak/NTCs/NTC_$n.pth" "$container:/project/NTC_$n.pth"; done
docker start -a "$container"
status=$(docker inspect --format '{{.State.ExitCode}}' "$container")
exit "$status"
