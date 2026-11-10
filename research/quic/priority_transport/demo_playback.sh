#!/bin/sh
set -eu
# Run from repository root with the optional QUIC environment activated.
work=$(mktemp -d /tmp/q3dg-playback.XXXXXX)
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$work/key.pem" -out "$work/cert.pem" -days 1 -subj /CN=localhost -addext subjectAltName=DNS:localhost >/dev/null 2>&1
python -m research.quic.priority_transport.experiment --cert "$work/cert.pem" --key "$work/key.pem" --output "$work/run" --duration 8 --fps 2 --policy buffer --prebuffer 2 --max-inflight 128
python -m json.tool "$work/run/summary.json"
echo "Raw demo evidence: $work/run"
