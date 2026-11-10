#!/usr/bin/env bash
# Run on the Mac. Browser traffic is encrypted inside SSH; no public VNC port.
set -euo pipefail
socket="${Q3DG_SSH_SOCKET:-/tmp/q3dg-desktop-ssh.sock}"
if ! ssh -S "$socket" -O check abeigi@gpu1.itec.aau.at >/dev/null 2>&1; then
ssh -i "$HOME/.ssh/q3dg_aau_itec" -o BatchMode=yes -M -S "$socket" -fnNT \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L 127.0.0.1:16081:127.0.0.1:6081 abeigi@gpu1.itec.aau.at
fi
printf '%s\n' 'Open http://127.0.0.1:16081/vnc.html?autoconnect=true&resize=scale'
