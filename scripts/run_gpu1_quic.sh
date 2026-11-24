#!/usr/bin/env bash
# Prepared university GPU1 runtime. Network shaping stays in private namespaces.
set -euo pipefail
[[ $(uname -s) == Linux ]] || { echo 'Run experiments on the GPU server' >&2; exit 64; }
(( $# )) || { echo 'Usage: run_gpu1_quic.sh COMMAND [ARGS]' >&2; exit 64; }
repo=$(cd "$(dirname "$0")/.." && pwd)
export Q3DGSTREAM_GPU_INDEX=${Q3DGSTREAM_GPU_INDEX:-1}
source "$repo/scripts/gpu1_env.sh"
overlay=${Q3DG_GPU_OVERLAY:-/local/users/abeigi/q3dg-gpu1-overlay}
[[ -d $overlay ]] || { echo 'Set Q3DG_GPU_OVERLAY to the prepared architecture-compatible Python extensions' >&2; exit 66; }
export PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}"
runtime=${Q3DG_RUNTIME_DIR:-/local/users/abeigi/q3dg-runtime}
mkdir -p "$runtime/tmp" "$runtime/cuda-cache" "$runtime/torch-cache"
export TMPDIR="$runtime/tmp" CUDA_CACHE_PATH="$runtime/cuda-cache" TORCH_EXTENSIONS_DIR="$runtime/torch-cache"
export DISPLAY=${Q3DG_DISPLAY:-:9}
export XAUTHORITY=${Q3DG_XAUTHORITY:-/local/users/abeigi/q3dg-desktop/session/Xauthority}
export Q3DG_GLFW_CONTEXT_API=egl PYOPENGL_PLATFORM=egl
site_lib=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH="$site_lib/nvidia/cudnn/lib:/local/users/abeigi/q3dg-desktop/root/usr/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=${TORCH_HOME:-/local/users/abeigi/q3dg-quinn-eval-20261122/torch-cache}
export PATH="/local/users/abeigi/q3dg-quality-tools/bin:$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
vgl=${Q3DG_VGLRUN:-/local/users/abeigi/q3dg-desktop/root/opt/VirtualGL/bin/vglrun}
exec "$vgl" -d "${Q3DG_VGL_DEVICE:-egl${Q3DGSTREAM_GPU_INDEX}}" "$repo/scripts/rootless_exec.sh" "$@"
