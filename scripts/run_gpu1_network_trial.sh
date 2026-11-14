#!/usr/bin/env bash
# Run on the prepared university GPU server. All shaping stays in a private netns.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/gpu1_env.sh
export Q3DG_PARENT_NETNS
Q3DG_PARENT_NETNS=$(readlink /proc/self/ns/net)
exec unshare -Urn python -m research.quic.priority_transport.native_playback_run "$@"
