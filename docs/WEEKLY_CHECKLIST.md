# Weekly streaming experiments — 10 November 2026

- [x] Check upstream PR and continue existing branch; follow-up sent to Publio.
- [x] Audit repository sizes: original 7,968,015 bytes; keep05 packed 66,741; stages 62,296.
- [x] Verify compressed assets are external and unavailable locally; do not claim their bytes were tested.
- [x] Make experimental admission window configurable, default 128 chunks; compare 16 and 128.
- [x] Diagnose random-loss slowdown with original checkpoint, window and congestion-controller ablation.
- [x] Validate chosen transport configuration and burst recovery budget with before/after evidence.
- [x] Implement contiguous media-time prebuffer, playback progression and rebuffer accounting.
- [x] Replay deadlines using 1/2/4 seconds of initial playable media.
- [x] Implement buffer/rate-based future-size selection and finite-video prefetch.
- [x] Finish fixed/adaptive netem comparisons at repository-documented byte scales.
- [x] Select metrics and quantify startup/stall/volume trade-offs; preserve failures.
- [x] Update English report, reproducible demo, charts and presentation.
- [x] Verify tests, review and push branch.

Exact compressed NTC transfer and decoded-quality comparisons remain dependent on
external assets and same-frame alternative representations from Publio. Size-matched
payload tests exercise the network only. They are not interchangeable NTC models.

Execution plan: audit → window/CC ablation → playback and feedback evaluation →
evidence/report/demo → tests/review/push. Defaults: 128 chunks, 120-second per-chunk
recovery budget, cubic as a candidate pending ablation, prebuffer 2 s with 1/2/4 s
sensitivity analysis. No host network changes; isolated Docker netem only.

## External dependency / remaining limits

- [ ] Transfer Publio’s actual compressed variants and measure rendered quality: assets are external and have not been supplied.
- [ ] Establish sustained throughput recovery under random loss across broader paths: current tuning improves observed runtime, but random-loss congestion response remains.

18 retained experiments (19 physical including a superseded baseline), 17 completed
transfers and one deliberate timeout reproduction. All results/failures retained.
The 18 playback experiments are media-time simulations with initial scene assumed available. Separate native GPU validation is recorded below.

## GPU server validation

- [x] Verify SSH key authentication to gpu1.itec.aau.at.
- [x] Inventory available GPUs: RTX 8000 (48 GiB), GV100 (32 GiB); driver 535.183.01.
- [x] Create isolated scratch environments; leave system packages unchanged.
- [x] Transfer initial Gaussian scene, config, first three NTCs and matching additions.
- [x] Finish Python 3.12 / PyTorch CUDA 12.1 and native CUDA extension setup.
- [x] Run real QUIC reception through original CUDA NTC/player providers.
- [x] Produce nonblank headless images with native Gaussian rasterizer and verify motion.
- [x] Retrieve artifacts, record reproducible environment and update GPU evidence.

The old Linux setup hardcodes compute capability 8.9 and CUDA 12.6; gpu1 requires
hardware/version detection instead (RTX 8000 capability 7.5; driver 535).

GPU result: 6,518 real QUIC chunks, 3 original NTC updates (1,991,680 parameters),
4 nonblack distinct CUDA-rendered frames, exact weights/bounds, nonzero motion;
peak allocated CUDA memory 149.92 MiB. 50 scoped tests passed on the server.
Interactive OpenGL, concurrent streaming/playback and compressed-model quality
remain outside the GPU smoke validation. See docs/GPU_SERVER_REPORT.md.

## Interactive remote player

- [x] Install user-local TurboVNC, VirtualGL, Openbox and noVNC without sudo.
- [x] Verify RTX 8000 OpenGL and native CUDA texture register/map/unmap through EGL.
- [x] Run the original main.py GUI with a live QUIC listener and fresh cache.
- [x] Transfer and verify 30 original NTCs, 30 additions, initial scene and config.
- [x] Connect Chrome on the Mac through a localhost-only SSH tunnel.
- [x] Verify camera rotation, Reset, Step, Play and Pause through the remote GUI.
- [x] Add restart guards, regression tests, operational guide and compact evidence.

The working Chrome session remains available. The Mac locked during final screenshot
capture; its screen must be unlocked manually to continue interacting. The server
session and SSH tunnel do not require an unlocked Mac display to remain active.
