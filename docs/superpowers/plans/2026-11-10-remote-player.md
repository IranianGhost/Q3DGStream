# Remote live-QUIC player implementation plan

Goal: run the original GUI player on the RTX 8000 and allow interactive access from the Mac.
Architecture: user-local TurboVNC display + VirtualGL EGL GPU rendering; browser noVNC over SSH localhost tunnel. Original main.py and LivePriorityState receive scene files over QUIC.
Authorization: user selected this remote desktop approach and requested execution; continue existing review branch. No system drivers, display-manager or other users' sessions are modified.

- [x] Audit GPU EGL permissions and user-local runtime dependencies.
- [x] Download official VirtualGL/TurboVNC release packages, verify release SHA256, extract under private scratch.
- [x] Start authenticated localhost-only desktop and verify NVIDIA OpenGL plus CUDA/GL texture registration.
- [x] Install GUI Python dependencies in existing isolated environment; run native main.py QUIC listener.
- [x] Send original scene/NTCs over QUIC and verify live player status and visible rendering.
- [x] Provide localhost browser viewer through SSH; verify camera/playback controls on Mac.
- [x] Save startup/stop instructions, evidence and reproducible scripts; tests/review/push.

Limits: admin access is unavailable; if EGL is denied, report the exact required GPU access change rather than changing global permissions. Native CUDA/GL interoperability must pass, not only glxinfo. GUI reception and render are assessed separately from desktop transport quality. Remote desktop compression/latency is not a QUIC benchmark.

Execution: actual GUI/CUDA interop passed; 30-update live transfer passed. Optional GLFW EGL hint preserves default context behavior. Reconnect finding fixed with two RED-to-GREEN shell integration regressions. Final full-screen capture awaits manual Mac unlock; earlier working GUI screenshot retained.
