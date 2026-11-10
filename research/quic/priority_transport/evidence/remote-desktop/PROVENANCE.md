Native GUI validation on gpu1, 10 November 2026.
User-local TurboVNC3.3.1, VirtualGL3.1.5, noVNC commit in REPORT.json.
Original main.py/CUDARenderer; EGL GLFW context and PYOPENGL_PLATFORM=egl.
Observed pre-fix failure: emulated GLX CUDA discovery/registration error999.
Observed post-fix success: native EGL texture register/map/unmap and GUI rendering.
Live run uQJjl1: 33,148 chunks,63 files,30NTC updates,30additions.
All62original file hashes match sources on Mac and server; manifest is sender-generated.
Mac interactions: camera rotate,Reset0,Step1,Play thenPause7,Play resumed.
52 scoped tests passed locally;51passed+1skip on server (occupied live port).
Receiver and VNC/proxy listeners are localhost-only; Mac connection uses SSH.
Password,TLSprivatekeys and Xauthority files excluded from repository.
GUI screenshot is retained locally under data/outputs/remote-desktop-20261110.
