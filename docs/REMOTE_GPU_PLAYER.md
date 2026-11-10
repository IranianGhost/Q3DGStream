# Interactive native player over a private remote desktop

The original main.py GUI runs on gpu1's Quadro RTX 8000. TurboVNC exports a
1440x900 desktop, VirtualGL redirects OpenGL to NVIDIA through EGL, and noVNC
provides a browser viewer. The Mac receives the desktop through an SSH tunnel;
NTC inference, CUDA rasterization and CUDA/OpenGL texture transfer remain on gpu1.

## Verified configuration

- User-local VirtualGL 3.1.5 and TurboVNC 3.3.1, extracted from official release
  Debian packages without sudo or changes to the shared display manager.
- VirtualGL package SHA256: df3f7788ce41b182a47c0d298e5cd6d2d63579522cb41825970b7726e825485e.
- TurboVNC package SHA256: 5d99050312360a07c28aca484b944816aa32e9fe07912e3ed50e6bfda45f80ad.
- Openbox 3.6.1-10 and dependencies were downloaded through Ubuntu apt and
  extracted into the same user-local root.
- Python GUI modules glfw, imgui, imageio and websockify installed in q3dg-venv312;
  noVNC comes from its official GitHub repository (exact revision recorded in evidence).
- NVIDIA OpenGL 4.3, Quadro RTX 8000; actual native CUDA texture registration passed.

## Recorded functional result

33,148 QUIC chunks delivered 63 committed files (271,325,109 bytes): manifest,
initial scene, config, 30 original checkpoints and 30 additions. All 62 original
asset SHA256 values also matched the Mac sources. The GUI reported init/config
ready, ready_ntc=30, ready_add=30 and max_play=30. Native CUDA backend rendering,
camera rotation, Reset to frame 0 and Step to frame 1 were observed from the Mac;
Play resumed and Pause stopped the segment at frame 7. Functional rendering was
also observed before all 30 updates had arrived.

The required-asset scene sender assigns priority 0 to these assets. This run
validates live receiver/player integration; it is not an independent priority-order
comparison. Earlier network priority experiments remain separate evidence.

52 scoped tests passed on the Mac. On gpu1, 51 passed and one occupied-port test
was skipped because the requested live receiver owns that port; that case passed
on the Mac. Restart guards preserve active session metadata instead of replacing
its cache/certificate. Re-running the launch script against the running server
confirmed the active session pointer stayed unchanged.

## Compatibility findings

The TurboVNC binary searches /usr/lib64/dri by default, but Ubuntu's Mesa driver
is at /usr/lib/x86_64-linux-gnu/dri. Passing -dridir explicitly restores the GLX
extension required by VirtualGL's GLX emulation; actual application rendering is NVIDIA EGL.

An emulated GLX context rendered OpenGL correctly but CUDA device discovery and
texture registration returned cudaErrorUnknown (999). A native GLFW EGL context
on the same GPU passed registration, mapping and unmapping. The player now accepts
Q3DG_GLFW_CONTEXT_API=egl; without that opt-in, the original context choice is unchanged.
PYOPENGL_PLATFORM=egl is also necessary so PyOpenGL/ImGui discover the current EGL
context. No CPU image-copy fallback or alternate renderer is used.

## Start or reconnect

Prepared server:

```bash
cd /local/users/abeigi/Q3DGStream
bash scripts/start_gpu1_desktop.sh
nohup bash scripts/run_gpu1_live_player.sh > /local/users/abeigi/q3dg-desktop/player.log 2>&1 < /dev/null &
```

The player listens for QUIC on 127.0.0.1:5003 with a fresh private cache and TLS
certificate. Its run directory is in /local/users/abeigi/q3dg-desktop/latest-live.
The initial scene is not loaded from disk through --autoload_fvv: it is received
through LivePriorityState, just like the NTC checkpoints and additions.

Send 30 original updates on the server:

```bash
source scripts/gpu1_env.sh
live_dir=$(cat /local/users/abeigi/q3dg-desktop/latest-live)
python -m research.quic.priority_transport.send_scene \
  --scene data/scenes/flame_steak --cert "$live_dir/cert.pem" \
  --port 5003 --updates 30 --cc cubic --datagram-size 1400 \
  --max-inflight 128 --log "$live_dir/sender.json"
```

On the Mac, from the local Q3DGStream checkout:

```bash
bash scripts/connect_gpu1_desktop.sh
```

Open http://127.0.0.1:16081/vnc.html?autoconnect=true&resize=scale in Chrome.
The launch script reuses an active player; to create a fresh receiving session, close
the previous player first. Its port check runs before replacing active metadata.

The generated VNC password is in the private remote file
/local/users/abeigi/q3dg-desktop/session/password. It is never committed.
The browser session is already authenticated after setup; reconnecting may ask again.
Clipboard sharing is unnecessary and can remain disabled.

To verify the window and real CUDA/OpenGL registration independently, run the
following inside the prepared server environment:

```bash
source scripts/gpu1_env.sh
export DISPLAY=:9 XAUTHORITY=/local/users/abeigi/q3dg-desktop/session/Xauthority
export LD_LIBRARY_PATH=/local/users/abeigi/q3dg-desktop/root/usr/lib:$LD_LIBRARY_PATH
/local/users/abeigi/q3dg-desktop/root/opt/VirtualGL/bin/vglrun -d egl0 \
  python scripts/check_gpu1_window.py
```

## Controls and scope

Left-drag rotates the camera; right-drag translates it; scroll moves it along the
view direction. Window > Show Control exposes Pause, Play, Reset and Step. Hide
Help and the optional QNTC demo panel through the Window menu if they overlap.
The optional QNTC demo manifest is not required by the live receiver.

The demo contains the initial scene plus the first 30 real FP32 NTC updates and
30 additions: 31 states, roughly one source second at the configured 30 fps.
LOOP_PLAYBACK=1 repeats the received segment. This is a functional GUI/live-receiver
test, not an FPS, WAN transport or image-quality benchmark. QUIC sender and receiver
run on the server; the remote desktop stream to the Mac is a separate connection.
The player's displayed INT4/INT8 bandwidth targets are inherited estimates, not
measurements of the FP32 checkpoints used in this demo.

## Private listeners and stopping

VNC binds only 127.0.0.1:5909, websockify only 127.0.0.1:6081, QUIC only
127.0.0.1:5003; the Mac tunnel binds only 127.0.0.1:16081. VNC password and Xauthority
live in a private session directory, and TLS keys remain in private live-run directories.
No public listener or global permission change is needed.

Close the player window when finished to release its CUDA resources. The desktop
may stay running for reconnecting. To stop only the processes from this setup,
check the command associated with each PID in q3dg-desktop/session/*.pid, then
send TERM to those PIDs. Do not kill unrelated X servers or shared GPU jobs.
Stop the Mac tunnel with:

```bash
ssh -S /tmp/q3dg-desktop-ssh.sock -O exit abeigi@gpu1.itec.aau.at
```

Official references: [VirtualGL EGL setup](https://github.com/VirtualGL/virtualgl/blob/main/doc/unixconfig.txt), [TurboVNC/VirtualGL](https://github.com/TurboVNC/turbovnc/blob/main/doc/vglusage.txt).
