# Source this file on gpu1 after the isolated environment has been installed.
# No passwords or SSH keys are stored in this file.
q3dg_gpu_root=${Q3DGSTREAM_GPU_ROOT:-/local/users/$(id -un)}
source "$q3dg_gpu_root/q3dg-venv312/bin/activate"
export CUDA_HOME="$q3dg_gpu_root/q3dg-cuda"
export PATH="$q3dg_gpu_root/q3dg-venv312/bin:$CUDA_HOME/bin:$PATH"
# UUID avoids the mismatch between CUDA and nvidia-smi numeric ordering.
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "${Q3DGSTREAM_GPU_INDEX:-0}" --query-gpu=uuid --format=csv,noheader)"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MAX_JOBS=2
export TCNN_CUDA_ARCHITECTURES=75
export TORCH_CUDA_ARCH_LIST=7.5
unset q3dg_gpu_root

export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
