"""Verify the original GLFW window and native CUDA/OpenGL texture registration."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("Q3DG_GLFW_CONTEXT_API", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glfw
import torch
from OpenGL import GL as gl
import main
from renderer_cuda import CUDARenderer

window = main.impl_glfw_init()
try:
    renderer = CUDARenderer(640, 480)
    assert renderer.cuda_image is not None
    print(json.dumps(dict(
        gl_vendor=gl.glGetString(gl.GL_VENDOR).decode(),
        gl_renderer=gl.glGetString(gl.GL_RENDERER).decode(),
        gl_version=gl.glGetString(gl.GL_VERSION).decode(),
        cuda_device=torch.cuda.get_device_name(),
        texture_registered=True,
        context_api=os.environ["Q3DG_GLFW_CONTEXT_API"],
    )))
finally:
    glfw.destroy_window(window)
    glfw.terminate()
