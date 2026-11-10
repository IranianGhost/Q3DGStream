# Native GPU validation on the ITEC server

The server has an NVIDIA Quadro RTX 8000 (48 GiB) and Quadro GV100 (32 GiB), with
NVIDIA driver 535.183.01. Validation uses the RTX 8000, compute capability 7.5.
The native CUDA model and Gaussian rasterization path is exercised without an
OpenGL window. Interactive GUI presentation is not part of this validation.

## Recorded result

The final run passed on the Quadro RTX 8000: 6,518 verified QUIC chunks, three
original FP32 NTC updates, 1,991,680 float32 parameters with exact source weight
and bounds equality, and four finite nonblack 640×480 PNGs. Maximum native Gaussian
movement per update was 0.02424 / 0.01681 / 0.02242 scene units. Peak allocated
PyTorch CUDA memory was 149.92 MiB. Frame images have distinct SHA-256 hashes;
mean absolute pixel difference from the initial image increased to 0.07301 on
an 8-bit RGB scale after three updates. This is evidence of changed pixels, not
an image-quality score.

All eight source asset hashes matched the original files on the Mac after
retrieval. Compact evidence and images are in
research/quic/priority_transport/evidence/gpu-server. Full received caches and
run logs remain in the dedicated remote run directories. The initial unsupported
antialiasing failure was retained and fixed; two subsequent camera-overview runs
passed. Camera parameters are recorded in REPORT.json.

50 scoped tests passed on the Linux server in CPU mode, separately from the real
CUDA integration smoke. The older full-repository progressive receiver collection
error remains outside this change.

## Isolated environment

Project checkout: `/local/users/abeigi/Q3DGStream` on gpu1. The active environment
is `/local/users/abeigi/q3dg-venv312`, using Python 3.12.16, PyTorch 2.5.1+cu121,
CUDA toolkit/runtime 12.1 and aioquic 1.7.2. The CUDA compiler prefix is
`/local/users/abeigi/q3dg-cuda`. No shared driver or system Python was modified.
The older Python 3.10 setup attempt is unused: aioquic 1.7.2 requires Python 3.11+.

The previous Linux helper hardcoded CUDA 12.6 and compute 8.9; those assumptions
are unsuitable here. CUDA device order differs from nvidia-smi order, so the RTX
is selected by its UUID, resolved at runtime, rather than by CUDA index alone.

Native source revisions:
- tiny-cuda-nn: `3daa6e5b8b8e0fa17971a1a20acfbb3754494b38` (tinycudann 2.0)
- diff-gaussian-rasterization: `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d`

Compiler jobs were capped at two. Toolkit components include cuda-nvcc,
cuda-cudart-dev, cuda-cccl, cuda-nvrtc-dev, cuda-nvrtc and cuda-driver-dev, all at
12.1 where versioned. NVRTC and cuSPARSE header dependencies had to be supplied:
CUDA_HOME points to the toolkit, and CPATH includes installed NVIDIA wheel header
directories. LIBRARY_PATH can include toolkit link stubs at build time, but the
runtime LD_LIBRARY_PATH must never include driver stubs.

Native extensions were installed with --no-build-isolation, MAX_JOBS=2,
TCNN_CUDA_ARCHITECTURES=75 and TORCH_CUDA_ARCH_LIST=7.5. GPU 1 (GV100, SM70) has
not been validated with these SM75-specific builds. Check resource availability
with nvidia-smi before use. /local scratch is not backed up; code and compact
validation evidence are retained in the review branch.

## What the headless adapter does

`gpu_smoke.py` starts the real priority QUIC receiver, transfers the initial scene,
config, first three original FP32 checkpoints and matching additions, then processes
the committed files through LivePriorityState and its original lazy providers.
No CPU model holder or fake rasterizer is used.

The headless renderer bypasses presentation-only OpenGL initialization and uses
the native Gaussian upload, NTC wrapper, motion application, additions, camera
matrices and CUDA rasterizer. Native motion defaults are preserved: translation
scale 0.42, rotation mode none. Raw predicted translation and rotation tensors
are checked for finiteness before native sanitization; motion must be nonzero.
Loaded CUDA parameter tensors and learned bounds must equal source checkpoint
values, with no dtype-casting tolerance in the weight comparison.

The initial state and three updates produce four finite, nonblack PNGs. Images
must differ and checkpoint sources are hashed. Camera parameters are recorded;
the camera is an automatic overview, not the original capture pose. Diagnostic
step wall times include loading/parity copies, motion checks, rendering and PNG
writing; they are not inference latency or an FPS benchmark.

## Renderer compatibility fix

The installed official rasterizer has no antialiasing keyword, while the player
passed antialiasing=False. `raster_compat.compatible_settings` omits only that
unsupported disabled flag. Enabling unsupported antialiasing or supplying another
unknown option still raises an error. Regression tests cover these cases. This
compatibility fix also benefits the regular native renderer.

## Reproduction

On the prepared server:

```bash
cd /local/users/abeigi/Q3DGStream
source scripts/gpu1_env.sh
bash scripts/run_gpu1_validation.sh
```

The wrapper generates a self-signed localhost TLS certificate trusted explicitly by the test client, uses a fresh dedicated
cache/output directory and saves environment metadata. The latest run directory
is recorded in `/local/users/abeigi/latest-gpu-run`. Retrieve only reports/images
and public metadata; TLS keys stay in the private remote run directory.

For another already-configured CUDA workstation, activate its environment and run
`scripts/run_headless_gpu.sh`. Its extension build must match that workstation's
architecture and CUDA version. GPU1's environment recipe is hardware-specific.

## Limits

This is a post-transfer network-to-GPU integration smoke test: data is fully
received before the sequential inference/render loop. Concurrent network/playback,
playback stalls under GPU load, full-sequence stability, GUI/OpenGL presentation,
capture-camera equivalence and objective image-quality metrics remain unverified.
The exact compressed NTC assets from Publio are still external and unavailable.
The earlier synthetic media-time adaptation results remain separate evidence.
