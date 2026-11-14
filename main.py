# Dynamic Gaussian viewer, playback controls, and stream reception.

import argparse
import json
import math
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog

import glfw
import imageio
import imgui
import numpy as np
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer

import util
import util_3dgstream
import util_gau
from live_tcp import LiveTCPState, infer_total_frames
from renderer_ogl import OpenGLRenderer

try:
    import torch
except Exception:
    torch = None


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

g_camera = util.Camera(720, 1280)

BACKEND_OGL = 0
BACKEND_CUDA = 1

g_renderer_list = [None]
g_renderer_idx = BACKEND_OGL
g_renderer = g_renderer_list[g_renderer_idx]

g_scale_modifier = 1.0
g_auto_sort = False

g_show_control_win = True
g_show_help_win = True
g_show_camera_win = False
g_show_qntc_demo_win = True

g_qntc_demo_manifest = None
g_qntc_demo_manifest_path = os.environ.get(
    "QNTC_DEMO_MANIFEST",
    r"qntc_demo_manifest.json",
)

g_render_mode_tables = [
    "Gaussian Ball",
    "Flat Ball",
    "Billboard",
    "Depth",
    "SH:0",
    "SH:0~1",
    "SH:0~2",
    "SH:0~3 (default)",
]
g_render_mode = 7

g_FVV_path = ""

VIDEO_FPS = 30.0
VIDEO_INTERVAL = 1.0 / VIDEO_FPS
g_last_frame_time = 0.0
g_timestep = 0
g_paused = True
g_reset = False
g_total_frame = 300

# -----------------------------
# Live camera print / fixed camera
# -----------------------------
_CAMERA_LAST_PRINT_TIME = 0.0

# -----------------------------
# Foveated overlay controls
# -----------------------------
g_show_fovea_overlay = False
g_show_fovea_fill = False
g_fovea_use_center = True
g_fovea_center_x = 0.5
g_fovea_center_y = 0.5
g_fovea_inner_deg = 30.0
g_fovea_outer_deg = 45.0
g_fovea_segments = 128
g_fovea_thickness = 2.0
g_fovea_dark_alpha = 0.45

# -----------------------------
# Black-mask state
# -----------------------------
g_black_mask_dir = ""
g_black_mask_loaded = False
g_black_mask_status = "disabled"
g_black_mask_error = ""
g_black_mask_warned = set()
g_base_global_mask = None
g_current_base_mask_kind = "none"
g_current_base_mask_count = 0
g_current_add_mask_count = 0
g_base_orig_opacity = None
g_add_orig_opacity = {}


def _camera_value_to_list(x):
    if x is None:
        return None

    try:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        return [float(v) for v in arr.tolist()]
    except Exception:
        try:
            return float(x)
        except Exception:
            return str(x)


def get_camera_debug_dict(camera):
    return {
        "position": _camera_value_to_list(getattr(camera, "position", None)),
        "target": _camera_value_to_list(getattr(camera, "target", None)),
        "up": _camera_value_to_list(getattr(camera, "up", None)),
        "yaw": float(getattr(camera, "yaw", 0.0)),
        "pitch": float(getattr(camera, "pitch", 0.0)),
        "target_dist": float(getattr(camera, "target_dist", 0.0)),
        "fovy": float(getattr(camera, "fovy", 0.0)),
        "fov_deg": float(np.rad2deg(getattr(camera, "fovy", 0.0))),
        "znear": float(getattr(camera, "znear", 0.0)),
        "zfar": float(getattr(camera, "zfar", 0.0)),
        "w": int(getattr(camera, "w", 0)),
        "h": int(getattr(camera, "h", 0)),
    }


def maybe_print_camera_info(camera):
    """
    Live camera logger.

    Enable with:
      set PRINT_CAMERA=1

    Optional:
      set CAMERA_PRINT_INTERVAL=1.0
      set CAMERA_JSON_OUT=C:\\Users\\Publi\\3DGStream\\viewer_fvv\\current_camera_live.json
    """
    global _CAMERA_LAST_PRINT_TIME

    if os.environ.get("PRINT_CAMERA", "").strip() != "1":
        return

    interval = float(os.environ.get("CAMERA_PRINT_INTERVAL", "1.0"))
    now = time.time()

    if now - _CAMERA_LAST_PRINT_TIME < interval:
        return

    _CAMERA_LAST_PRINT_TIME = now

    data = get_camera_debug_dict(camera)
    text = json.dumps(data, indent=2)

    print()
    print("[CAMERA LIVE]")
    print(text, flush=True)

    out_path = os.environ.get("CAMERA_JSON_OUT", "").strip()
    if out_path:
        try:
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print("[CAMERA LIVE] wrote:", out_path, flush=True)
        except Exception as e:
            print("[CAMERA LIVE] could not write camera JSON:", repr(e), flush=True)


def apply_fixed_camera_from_env(camera):
    """
    Optional fixed-camera mode for metric rendering.

    Environment variables:
      LOCK_CAMERA=1
      FIXED_CAMERA_JSON=C:\\path\\camera.json

    JSON fields accepted:
      position, target, up, yaw, pitch, target_dist, fovy, fov_deg, znear, zfar

    If LOCK_CAMERA=1, mouse wheel, mouse movement, and Q/E roll are disabled.
    """

    fixed_camera_json = os.environ.get("FIXED_CAMERA_JSON", "").strip()
    lock_camera = os.environ.get("LOCK_CAMERA", "").strip() == "1"

    if fixed_camera_json:
        if not os.path.isfile(fixed_camera_json):
            print("[FIXED CAMERA] file not found:", fixed_camera_json)
        else:
            with open(fixed_camera_json, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            if "position" in cfg:
                camera.position = np.array(cfg["position"], dtype=np.float32)

            if "target" in cfg:
                camera.target = np.array(cfg["target"], dtype=np.float32)

            if "up" in cfg:
                camera.up = np.array(cfg["up"], dtype=np.float32)

            if "yaw" in cfg:
                camera.yaw = float(cfg["yaw"])

            if "pitch" in cfg:
                camera.pitch = float(cfg["pitch"])

            if "target_dist" in cfg:
                camera.target_dist = float(cfg["target_dist"])

            if "fovy" in cfg:
                camera.fovy = float(cfg["fovy"])

            if "fov_deg" in cfg:
                camera.fovy = float(np.deg2rad(float(cfg["fov_deg"])))

            if "znear" in cfg:
                camera.znear = float(cfg["znear"])

            if "zfar" in cfg:
                camera.zfar = float(cfg["zfar"])

            camera.is_pose_dirty = True
            camera.is_intrin_dirty = True

            print("[FIXED CAMERA] loaded:", fixed_camera_json)
            print(json.dumps(get_camera_debug_dict(camera), indent=2), flush=True)

    if lock_camera:
        camera.process_mouse = lambda *args, **kwargs: None
        camera.process_wheel = lambda *args, **kwargs: None
        camera.process_roll_key = lambda *args, **kwargs: None
        print("[FIXED CAMERA] camera input locked")


def impl_glfw_init():
    window_name = "Tiny 3DGStream Viewer"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # VirtualGL's headless EGL backend supports CUDA texture interop when GLFW
    # creates an EGL context rather than an emulated GLX context.
    if os.environ.get("Q3DG_GLFW_CONTEXT_API", "native").lower() == "egl":
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)

    global window
    window = glfw.create_window(g_camera.w, g_camera.h, window_name, None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False

    g_camera.process_mouse(xpos, ypos)


def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return

    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)


def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)


def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)


def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:
        g_renderer.update_camera_pose(g_camera)
        g_camera.is_pose_dirty = False


def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera)
        g_camera.is_intrin_dirty = False


def update_activated_renderer_state(gaus: util_gau.GaussianData):
    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)


def window_resize_callback(window, width, height):
    if os.environ.get("LOCK_RENDER_RES", "").strip() == "1":
        return

    if width <= 0 or height <= 0:
        return

    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)

    if g_renderer is not None:
        g_renderer.set_render_reso(width, height)


def autoload_session_if_requested():
    # Load a scene when playback requests it.
    global g_FVV_path, g_total_frame, g_paused, g_timestep, g_last_frame_time
    global g_renderer

    if not args.autoload_fvv:
        return

    fvv = args.autoload_fvv
    ply_path = os.path.join(fvv, "init_3dgs.ply")
    if not os.path.exists(ply_path):
        print(f"[AUTOLOAD] init_3dgs.ply not found at: {ply_path}")
        return

    gaussians = util_gau.load_ply(ply_path)
    g_renderer.update_gaussian_data(gaussians)
    g_renderer.sort_and_update(g_camera)

    if args.frames > 0:
        g_total_frame = args.frames
    else:
        g_total_frame = infer_total_frames(fvv)

    g_FVV_path = fvv
    g_renderer.NTCs = util_3dgstream.load_NTCs(
        g_FVV_path, g_renderer.gaussians, g_total_frame
    )
    g_renderer.additional_3dgs = util_3dgstream.load_Additions(
        g_FVV_path, g_total_frame
    )

    g_timestep = 0
    g_last_frame_time = time.time()
    g_paused = not args.autoplay
    reset_mask_runtime_state()


def angular_radius_to_pixels(theta_deg: float, fovy_rad: float, height_px: int) -> float:
    theta_deg = max(0.001, float(theta_deg))
    theta_deg = min(theta_deg, 89.0)
    theta = math.radians(theta_deg)
    denom = math.tan(max(1e-6, fovy_rad * 0.5))
    r_px = (math.tan(theta) / denom) * (height_px * 0.5)
    return max(0.0, float(r_px))


def get_fovea_center_pixels():
    if g_fovea_use_center:
        return g_camera.w * 0.5, g_camera.h * 0.5

    cx = np.clip(g_fovea_center_x, 0.0, 1.0) * g_camera.w
    cy = np.clip(g_fovea_center_y, 0.0, 1.0) * g_camera.h
    return float(cx), float(cy)


def draw_foveated_overlay():
    """
    QNTC demo recording build: foveated overlay disabled.

    This intentionally draws nothing, so the red/yellow circles and the dark
    rectangular mask are never rendered even if UI variables are toggled.
    """
    return


def fmt_vec3(v):
    return f"{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}"


def print_camera_state_to_console():
    pos = fmt_vec3(g_camera.position)
    target = fmt_vec3(g_camera.target)
    up = fmt_vec3(g_camera.up)
    fov_deg = np.rad2deg(g_camera.fovy)

    print("\n[CAMERA STATE]")
    print(f"position = ({pos})")
    print(f"target = ({target})")
    print(f"up = ({up})")
    print(f"yaw = {g_camera.yaw:.6f}")
    print(f"pitch = {g_camera.pitch:.6f}")
    print(f"fov_deg = {fov_deg:.6f}")
    print(f"target_dist = {g_camera.target_dist:.6f}")
    print("[renderer.py args]")
    print(
        f"--cam_pos={pos} --cam_target={target} --cam_up={up} --cam_fov_deg={fov_deg:.6f}\n"
    )

    out_path = os.environ.get("CAMERA_JSON_OUT", "").strip()
    if out_path:
        text = json.dumps(get_camera_debug_dict(g_camera), indent=2)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("[CAMERA STATE] wrote:", out_path)


# ------------------------------------------------------------
# Mask helpers
# ------------------------------------------------------------
def reset_mask_runtime_state():
    global g_base_orig_opacity, g_add_orig_opacity
    global g_current_base_mask_kind, g_current_base_mask_count, g_current_add_mask_count

    g_base_orig_opacity = None
    g_add_orig_opacity = {}
    g_current_base_mask_kind = "none"
    g_current_base_mask_count = 0
    g_current_add_mask_count = 0


def _warn_once(key, msg):
    if key in g_black_mask_warned:
        return
    g_black_mask_warned.add(key)
    print(msg)


def _is_torch_tensor(x):
    return (torch is not None) and isinstance(x, torch.Tensor)


def _clone_array(x):
    if _is_torch_tensor(x):
        return x.clone()
    return np.array(x, copy=True)


def _restore_array(dst, src):
    if _is_torch_tensor(dst):
        dst.copy_(src)
    else:
        dst[...] = src


def _set_zero_by_mask(arr, mask_bool):
    if _is_torch_tensor(arr):
        mask_t = torch.as_tensor(mask_bool, device=arr.device, dtype=torch.bool)
        arr[mask_t] = 0
    else:
        arr[mask_bool] = 0


def _load_npy_if_exists(path):
    if path is None:
        return None
    if not os.path.isfile(path):
        return None
    try:
        return np.load(path)
    except Exception as e:
        print(f"[MASK] Failed to load {path}: {repr(e)}")
        return None


def _base_global_mask_candidates(mask_dir):
    return [
        os.path.join(mask_dir, "base_black_global.npy"),
        os.path.join(mask_dir, "base_black.npy"),
    ]


def _base_counts_mask_path(mask_dir):
    return os.path.join(mask_dir, "base_black_counts.npy")


def _find_framewise_base_mask_path(mask_dir, idx):
    base_dir = os.path.join(mask_dir, "base_black_framewise")
    if not os.path.isdir(base_dir):
        return None

    candidates = [
        os.path.join(base_dir, f"{idx:06d}.npy"),
        os.path.join(base_dir, f"base_black_{idx:06d}.npy"),
        os.path.join(base_dir, f"{idx:04d}.npy"),
        os.path.join(base_dir, f"base_black_{idx:04d}.npy"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _find_add_mask_path(mask_dir, idx):
    add_dir = os.path.join(mask_dir, "add_black")
    if not os.path.isdir(add_dir):
        return None

    candidates = [
        os.path.join(add_dir, f"{idx:06d}.npy"),
        os.path.join(add_dir, f"add_black_{idx:06d}.npy"),
        os.path.join(add_dir, f"{idx:04d}.npy"),
        os.path.join(add_dir, f"add_black_{idx:04d}.npy"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def load_mask_dir_once():
    global g_black_mask_dir, g_black_mask_loaded, g_black_mask_status, g_black_mask_error
    global g_base_global_mask

    if not args.black_mask_dir:
        g_black_mask_status = "disabled"
        return

    if g_black_mask_loaded:
        return

    g_black_mask_dir = os.path.normpath(
        os.path.abspath(os.path.expanduser(args.black_mask_dir))
    )

    try:
        counts = _load_npy_if_exists(_base_counts_mask_path(g_black_mask_dir))
        if counts is not None and args.base_black_min_count > 1:
            g_base_global_mask = counts >= int(args.base_black_min_count)
            g_black_mask_status = f"global from counts >= {args.base_black_min_count}"
        else:
            g_base_global_mask = None
            for c in _base_global_mask_candidates(g_black_mask_dir):
                arr = _load_npy_if_exists(c)
                if arr is not None:
                    g_base_global_mask = arr.astype(bool)
                    g_black_mask_status = f"global from {os.path.basename(c)}"
                    break

        if g_base_global_mask is None:
            if os.path.isdir(os.path.join(g_black_mask_dir, "base_black_framewise")):
                g_black_mask_status = "framewise base available"
            elif os.path.isdir(os.path.join(g_black_mask_dir, "add_black")):
                g_black_mask_status = "additions only"
            else:
                g_black_mask_status = "no masks found"

        g_black_mask_loaded = True
        g_black_mask_error = ""

        print(f"[MASK] dir = {g_black_mask_dir}")
        print(f"[MASK] status = {g_black_mask_status}")
        if g_base_global_mask is not None:
            print(
                f"[MASK] global base loaded: {int(g_base_global_mask.sum())}/{len(g_base_global_mask)}"
            )

    except Exception as e:
        g_black_mask_error = repr(e)
        g_black_mask_status = "error"
        g_black_mask_loaded = True
        print(f"[MASK] ERROR: {g_black_mask_error}")


def _ensure_base_opacity_snapshot():
    global g_base_orig_opacity

    gaus = getattr(g_renderer, "gaussians", None)
    if gaus is None or not hasattr(gaus, "opacity"):
        return False

    if g_base_orig_opacity is None:
        g_base_orig_opacity = _clone_array(gaus.opacity)
        return True

    try:
        if gaus.opacity.shape != g_base_orig_opacity.shape:
            g_base_orig_opacity = _clone_array(gaus.opacity)
    except Exception:
        g_base_orig_opacity = _clone_array(gaus.opacity)

    return True


def _ensure_add_opacity_snapshot(idx):
    adds = getattr(g_renderer, "additional_3dgs", None)
    if adds is None:
        return None

    try:
        add_obj = adds[idx]
    except Exception:
        return None

    if add_obj is None or not hasattr(add_obj, "opacity"):
        return None

    snap = g_add_orig_opacity.get(idx, None)
    if snap is None:
        g_add_orig_opacity[idx] = _clone_array(add_obj.opacity)
    else:
        try:
            if add_obj.opacity.shape != snap.shape:
                g_add_orig_opacity[idx] = _clone_array(add_obj.opacity)
        except Exception:
            g_add_orig_opacity[idx] = _clone_array(add_obj.opacity)

    return add_obj


def restore_original_opacities_for_current_frame():
    if _ensure_base_opacity_snapshot():
        gaus = getattr(g_renderer, "gaussians", None)
        _restore_array(gaus.opacity, g_base_orig_opacity)

    add_obj = _ensure_add_opacity_snapshot(g_timestep)
    if add_obj is not None:
        _restore_array(add_obj.opacity, g_add_orig_opacity[g_timestep])


def _get_base_mask_for_frame(idx, n_gaussians):
    framewise_path = _find_framewise_base_mask_path(g_black_mask_dir, idx)
    if framewise_path is not None:
        m = _load_npy_if_exists(framewise_path)
        if m is not None:
            m = m.astype(bool)
            if len(m) == n_gaussians:
                return m, "framewise"
            _warn_once(
                ("frame_base_len", idx, n_gaussians),
                f"[MASK] framewise base mask length mismatch at frame {idx}: mask={len(m)} gaussians={n_gaussians}",
            )

    if g_base_global_mask is not None:
        if len(g_base_global_mask) == n_gaussians:
            return g_base_global_mask, "global"
        _warn_once(
            ("global_base_len", n_gaussians),
            f"[MASK] global base mask length mismatch: mask={len(g_base_global_mask)} gaussians={n_gaussians}",
        )

    return None, "none"


def apply_opacity_masks_for_current_frame():
    global g_current_base_mask_kind, g_current_base_mask_count, g_current_add_mask_count
    global g_black_mask_error

    g_current_base_mask_kind = "none"
    g_current_base_mask_count = 0
    g_current_add_mask_count = 0

    if not args.black_mask_dir:
        return

    load_mask_dir_once()

    gaus = getattr(g_renderer, "gaussians", None)
    if gaus is None or not hasattr(gaus, "opacity"):
        return

    restore_original_opacities_for_current_frame()

    try:
        n_base = int(gaus.opacity.shape[0])
    except Exception:
        return

    base_mask, kind = _get_base_mask_for_frame(g_timestep, n_base)
    g_current_base_mask_kind = kind

    if base_mask is not None:
        try:
            _set_zero_by_mask(gaus.opacity, base_mask)
            g_current_base_mask_count = int(base_mask.sum())
        except Exception as e:
            g_black_mask_error = repr(e)
            _warn_once(
                ("base_apply", g_timestep, repr(e)),
                f"[MASK] base apply error at frame {g_timestep}: {repr(e)}",
            )

    add_obj = _ensure_add_opacity_snapshot(g_timestep)
    add_path = _find_add_mask_path(g_black_mask_dir, g_timestep)

    if add_obj is not None and add_path is not None:
        add_mask = _load_npy_if_exists(add_path)
        if add_mask is not None:
            add_mask = add_mask.astype(bool)
            try:
                n_add = int(add_obj.opacity.shape[0])
            except Exception:
                n_add = -1

            if n_add >= 0 and len(add_mask) == n_add:
                try:
                    _set_zero_by_mask(add_obj.opacity, add_mask)
                    g_current_add_mask_count = int(add_mask.sum())
                except Exception as e:
                    g_black_mask_error = repr(e)
                    _warn_once(
                        ("add_apply", g_timestep, repr(e)),
                        f"[MASK] additions apply error at frame {g_timestep}: {repr(e)}",
                    )
            else:
                _warn_once(
                    ("add_len", g_timestep, n_add),
                    f"[MASK] additions mask length mismatch at frame {g_timestep}: mask={len(add_mask)} additions={n_add}",
                )



def _qntc_format_float(value, digits=2, suffix=""):
    try:
        if value is None:
            return "N/A"
        if isinstance(value, str):
            return value
        if np.isnan(float(value)):
            return "∞" if "psnr" in suffix.lower() else "N/A"
        if np.isinf(float(value)):
            return "∞"
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return str(value)


def _qntc_load_demo_manifest():
    global g_qntc_demo_manifest

    if g_qntc_demo_manifest is not None:
        return g_qntc_demo_manifest

    path = g_qntc_demo_manifest_path

    try:
        with open(path, "r", encoding="utf-8") as f:
            # Python json accepts NaN by default.
            g_qntc_demo_manifest = json.load(f)
        print("[QNTC DEMO] loaded manifest:", path)
    except Exception as e:
        g_qntc_demo_manifest = {
            "error": repr(e),
            "path": path,
            "modes": [],
        }
        print("[QNTC DEMO] failed to load manifest:", repr(e))

    return g_qntc_demo_manifest


def _qntc_metric_text(metrics, key, digits=4, inf_for_nan=False):
    if not metrics:
        return "N/A"

    value = metrics.get(key, None)

    try:
        if value is None:
            return "N/A"
        v = float(value)
        if np.isnan(v):
            return "∞" if inf_for_nan else "N/A"
        if np.isinf(v):
            return "∞"
        return f"{v:.{digits}f}"
    except Exception:
        return str(value)


def draw_qntc_demo_panel():
    """
    Live demo panel for the Quantized NTC compression demo.
    Reads qntc_demo_manifest.json and displays size, bandwidth, and quality metrics.
    """
    global g_show_qntc_demo_win

    if not g_show_qntc_demo_win:
        return

    manifest = _qntc_load_demo_manifest()

    opened, g_show_qntc_demo_win = imgui.begin("QNTC Dynamic 3DGS Demo", True)

    if not opened:
        imgui.end()
        return

    if "error" in manifest:
        imgui.text("Could not load demo manifest.")
        imgui.text(str(manifest.get("path", "")))
        imgui.text(str(manifest.get("error", "")))
        imgui.end()
        return

    imgui.text(manifest.get("demo_name", "Quantized NTC Demo"))
    imgui.text(f"Scene: {manifest.get('scene', 'N/A')}")
    imgui.text(
        f"Playback: {manifest.get('frames', 'N/A')} frames, "
        f"{manifest.get('fps', 'N/A')} FPS, "
        f"{manifest.get('duration_seconds', 'N/A')} s"
    )

    settings = manifest.get("renderer_settings", {})
    imgui.separator()
    imgui.text("Renderer settings")
    imgui.text(f"NTC rotation: {settings.get('ntc_rotation_mode', 'N/A')}")
    imgui.text(f"NTC translation scale: {settings.get('ntc_translation_scale', 'N/A')}")
    imgui.text(f"Additional 3DGS: {settings.get('additional_3dgs', 'N/A')}")

    imgui.separator()
    imgui.text("Compression modes")

    modes = manifest.get("modes", [])

    # Simple table-like layout using columns for compatibility with old pyimgui versions.
    imgui.columns(7, "qntc_modes_table", True)
    imgui.text("Mode")
    imgui.next_column()
    imgui.text("Size MB")
    imgui.next_column()
    imgui.text("Red. %")
    imgui.next_column()
    imgui.text("Mbps")
    imgui.next_column()
    imgui.text("PSNR")
    imgui.next_column()
    imgui.text("SSIM")
    imgui.next_column()
    imgui.text("LPIPS")
    imgui.next_column()
    imgui.separator()

    for mode in modes:
        metrics = mode.get("metrics", None)

        imgui.text(str(mode.get("name", "N/A")))
        imgui.next_column()

        imgui.text(_qntc_format_float(mode.get("total_scene_mb"), 1))
        imgui.next_column()

        imgui.text(_qntc_format_float(mode.get("reduction_vs_fp32_percent"), 1))
        imgui.next_column()

        imgui.text(_qntc_format_float(mode.get("estimated_full_scene_mbps_for_10s"), 1))
        imgui.next_column()

        imgui.text(_qntc_metric_text(metrics, "psnr_mean", digits=2, inf_for_nan=True))
        imgui.next_column()

        imgui.text(_qntc_metric_text(metrics, "ssim_mean", digits=4))
        imgui.next_column()

        imgui.text(_qntc_metric_text(metrics, "lpips_alex_mean", digits=4))
        imgui.next_column()

    imgui.columns(1)

    imgui.end()


def main():
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort
    global g_show_control_win, g_show_help_win, g_show_camera_win, g_show_qntc_demo_win
    global g_render_mode, g_render_mode_tables
    global g_FVV_path, g_paused, g_reset, g_timestep, g_last_frame_time, g_total_frame, VIDEO_FPS, VIDEO_INTERVAL
    global g_show_fovea_overlay, g_show_fovea_fill, g_fovea_use_center
    global g_fovea_center_x, g_fovea_center_y, g_fovea_inner_deg, g_fovea_outer_deg, g_fovea_dark_alpha

    VIDEO_FPS = float(args.video_fps)
    VIDEO_INTERVAL = 1.0 / max(1e-6, VIDEO_FPS)

    imgui.create_context()
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5

    window = impl_glfw_init()
    impl = GlfwRenderer(window)

    root = tk.Tk()
    root.withdraw()

    # Apply optional fixed camera after the GLFW window exists but before controls.
    # This can load a JSON camera file and/or lock camera input.
    apply_fixed_camera_from_env(g_camera)

    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_window_size_callback(window, window_resize_callback)

    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)
    try:
        from renderer_cuda import CUDARenderer
        g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
    except ImportError:
        pass

    if len(g_renderer_list) > 1:
        g_renderer_idx = BACKEND_CUDA
    else:
        g_renderer_idx = BACKEND_OGL

    g_renderer = g_renderer_list[g_renderer_idx]
    gaussians = util_gau.naive_gaussian()
    update_activated_renderer_state(gaussians)

    g_last_frame_time = time.time()
    load_mask_dir_once()

    live = None
    live_autoplay_started = False
    experiment = None
    if args.priority_experiment:
        from collections import deque
        from pathlib import Path
        from research.quic.priority_transport.live_playback import LivePlayback, write_snapshot

        experiment_dir = Path(args.priority_experiment)
        experiment = LivePlayback(args.frames - 1, VIDEO_FPS, args.priority_prebuffer)
        samples = deque()
        next_feedback = 0.0
        transfer_started = None
        if g_renderer_idx != BACKEND_CUDA:
            raise RuntimeError("live playback experiment requires the CUDA renderer")

    if args.priority_listen is not None:
        from live_priority import LivePriorityState
        g_FVV_path = args.priority_cache
        g_total_frame = args.frames if args.frames > 0 else 300
        live = LivePriorityState(cache_root=args.priority_cache, total_frames_hint=g_total_frame, autoplay=args.autoplay)
        live.start_receiver('127.0.0.1', args.priority_listen,
                            cert=args.priority_cert, key=args.priority_key)
    elif args.tcp_listen is not None:
        g_FVV_path = args.tcp_cache
        g_total_frame = args.frames if args.frames > 0 else 300
        live = LiveTCPState(cache_root=args.tcp_cache, total_frames_hint=g_total_frame, autoplay=args.autoplay)
        live.start_receiver(
            bind_host=args.tcp_bind,
            port=args.tcp_listen,
            clear_cache=args.tcp_clear_cache,
        )

    if live is None:
        autoload_session_if_requested()

    while not glfw.window_should_close(window):
        glfw.poll_events()
        maybe_print_camera_info(g_camera)

        impl.process_inputs()
        imgui.new_frame()

        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose_lazy()
        update_camera_intrin_lazy()

        if live is not None:
            live.process_new_files(g_renderer, g_camera)
            if experiment is None and live.autoplay and (not live_autoplay_started):
                if live.init_ply_ok and live.config_ok and live.max_play >= 0:
                    g_paused = False
                    g_timestep = 0
                    g_last_frame_time = time.time()
                    live_autoplay_started = True
                    reset_mask_runtime_state()

        max_play = (g_total_frame - 1)
        if live is not None and live.max_play >= 0:
            max_play = live.max_play

        current_time = time.time()
        if experiment is not None:
            now = time.monotonic()
            start_file = experiment_dir / "transfer-start.json"
            if transfer_started is None and start_file.is_file():
                transfer_started = json.loads(start_file.read_text())["at"]
            if live.err:
                raise RuntimeError(live.err)
            selected = experiment.select(now, live.init_ply_ok and live.config_ok, max_play)
            if selected is not None:
                g_timestep = selected
            if now >= next_feedback:
                with live._lock:
                    committed = live.total_bytes
                samples.append((now, committed))
                while len(samples) > 2 and samples[1][0] <= now - 2:
                    samples.popleft()
                elapsed = now - samples[0][0]
                rate = (committed - samples[0][1]) / elapsed if elapsed > 0 else 0
                write_snapshot(experiment_dir / "feedback.json", dict(
                    at=now, port=live.bound_port if hasattr(live, "bound_port") else None,
                    buffer_seconds=max(0, max_play - (experiment.frame or 0)) / VIDEO_FPS,
                    committed_Bps=rate, frame=experiment.frame, ready=max_play,
                    stalled=experiment.stall_start is not None,
                ))
                next_feedback = now + 0.1
        elif (current_time - g_last_frame_time) >= VIDEO_INTERVAL and not g_paused:
            loop_playback = os.environ.get("LOOP_PLAYBACK", "0").strip() == "1"

            if g_timestep < max_play:
                g_timestep += 1
            else:
                if loop_playback and max_play > 0:
                    g_timestep = 0
                    g_renderer.fvv_reset()
                    reset_mask_runtime_state()
                else:
                    g_timestep = max_play

            g_last_frame_time = current_time

        if g_reset:
            g_renderer.fvv_reset()
            g_reset = False
            g_last_frame_time = time.time()
            reset_mask_runtime_state()

        if g_timestep > max_play:
            g_timestep = max_play if max_play >= 0 else 0

        apply_opacity_masks_for_current_frame()

        try:
            if experiment is None or selected is not None:
                g_renderer.draw(g_timestep)
        except Exception as e:
            if experiment is not None:
                raise
            if live is not None:
                with live._lock:
                    live.err = f"draw error: {repr(e)}"
            else:
                print("[DRAW] ERROR:", repr(e))

        draw_foveated_overlay()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                _, g_show_control_win = imgui.menu_item("Show Control", None, g_show_control_win)
                _, g_show_help_win = imgui.menu_item("Show Help", None, g_show_help_win)
                _, g_show_camera_win = imgui.menu_item("Show Camera Control", None, g_show_camera_win)
                _, g_show_qntc_demo_win = imgui.menu_item("Show QNTC Demo Panel", None, g_show_qntc_demo_win)
                imgui.end_menu()
            imgui.end_main_menu_bar()

        if g_show_control_win:
            if imgui.begin("Control", True):
                changed, g_renderer_idx = imgui.combo(
                    "backend", g_renderer_idx, ["ogl", "cuda"][: len(g_renderer_list)]
                )
                if changed:
                    g_renderer = g_renderer_list[g_renderer_idx]
                    update_activated_renderer_state(gaussians)
                    if live is not None:
                        g_renderer.NTCs = live.ntc_list
                        g_renderer.additional_3dgs = live.add_list
                    reset_mask_runtime_state()

                imgui.text(f"# of Gaus = {len(getattr(g_renderer, 'gaussians', gaussians))}")
                imgui.text(f"Render FPS = {imgui.get_io().framerate:.1f}")
                imgui.text(f"Video FPS = {VIDEO_FPS:.1f}")
                imgui.text(f"FVV Dir: {g_FVV_path}")
                imgui.text(f"Frame {g_timestep}")

                if args.black_mask_dir:
                    imgui.separator()
                    imgui.text("Opacity masks:")
                    imgui.text(f" dir: {g_black_mask_dir}")
                    imgui.text(f" status: {g_black_mask_status}")
                    imgui.text(f" base mode: {g_current_base_mask_kind}")
                    imgui.text(f" base selected: {g_current_base_mask_count}")
                    imgui.text(f" add selected: {g_current_add_mask_count}")
                    if g_black_mask_error:
                        imgui.text(f" err: {g_black_mask_error}")

                if live is not None:
                    snap = live.ui_snapshot()
                    imgui.separator()
                    imgui.text("QUIC PRIORITY LIVE:" if args.priority_listen is not None else "TCP LIVE:")
                    imgui.text(f" cache: {snap['cache']}")
                    imgui.text(f" status: {snap['status']} ({snap['connected']})")
                    if snap["err"]:
                        imgui.text(f" err: {snap['err']}")
                    imgui.text(f" init_ply: {snap['init_ply']}")
                    imgui.text(f" config: {snap['config']}")
                    imgui.text(f" ready_ntc: {snap['ready_ntc']}")
                    imgui.text(f" ready_add: {snap['ready_add']}")
                    imgui.text(f" max_play: {snap['max_play']}")
                    imgui.text(f" files: {snap['files']} ({snap['gb']:.2f} GB)")

                    imgui.separator()
                    imgui.text("QUIC File Monitor:" if args.priority_listen is not None else "TCP Stream Monitor:")

                    imgui.text(
                        f" rx goodput avg/inst: "
                        f"{snap.get('rx_mbps_avg', 0.0):.1f} / "
                        f"{snap.get('rx_mbps_inst', 0.0):.1f} Mbps"
                    )

                    ntc_done_label = "complete" if snap.get("ntc_transfer_done", False) else "receiving"
                    imgui.text(
                        f" NTC {ntc_done_label}: {snap.get('ntc_files', 0)} files, "
                        f"{snap.get('ntc_mb', 0.0):.1f} MB, "
                        f"{snap.get('ntc_mbps_avg', 0.0):.1f} Mbps, "
                        f"{snap.get('ntc_fps_recv', 0.0):.1f} files/s"
                    )

                    cache_ahead = int(snap.get("max_play", -1)) - int(g_timestep)
                    if cache_ahead < 0:
                        cache_ahead = 0
                    cache_ahead_s = cache_ahead / max(1e-6, VIDEO_FPS)
                    imgui.text(
                        f" cache ahead: {cache_ahead} frames "
                        f"({cache_ahead_s:.2f} s)"
                    )

                    cur_file = str(snap.get("current_file", ""))
                    if cur_file:
                        cur_name = os.path.basename(cur_file)
                        imgui.text(
                            f" receiving: {cur_name} "
                            f"{100.0 * snap.get('current_progress', 0.0):.1f}% "
                            f"({snap.get('current_file_mbps', 0.0):.1f} Mbps)"
                        )
                    else:
                        last_name = os.path.basename(str(snap.get("last_file", "")))
                        if last_name:
                            imgui.text(
                                f" last file: {last_name} "
                                f"{snap.get('last_file_mb', 0.0):.2f} MB, "
                                f"{snap.get('last_file_ms', 0.0):.1f} ms, "
                                f"{snap.get('last_file_mbps', 0.0):.1f} Mbps"
                            )

                    imgui.text(" Required dynamic bandwidth:")
                    imgui.text("  INT8 NTC-only: ~456.7 Mbps")
                    imgui.text("  INT4 NTC-only: ~242.0 Mbps")

                    # Prefer completed NTC transfer rate. While receiving, use current receiver goodput.
                    if snap.get("ntc_transfer_done", False):
                        rx_ref = float(snap.get("ntc_mbps_avg", 0.0))
                    else:
                        rx_ref = max(
                            float(snap.get("rx_mbps_inst", 0.0)),
                            float(snap.get("rx_mbps_avg", 0.0)),
                        )

                    if rx_ref >= 500.0:
                        rec_mode = "INT8"
                    elif rx_ref >= 250.0:
                        rec_mode = "INT4"
                    elif rx_ref > 0.0:
                        rec_mode = "Below INT4 target"
                    else:
                        rec_mode = "Waiting for stream"

                    imgui.text(f" bandwidth supports: {rec_mode}")

                imgui.text("#Frames:")
                imgui.same_line()
                max_frames = max(300, g_total_frame)
                _, g_total_frame = imgui.slider_int("frames", g_total_frame, 1, max_frames)

                if imgui.button("Pause"):
                    g_paused = True
                    g_last_frame_time = time.time()

                imgui.same_line()
                if imgui.button("Play"):
                    g_paused = False
                    g_last_frame_time = time.time()

                imgui.same_line()
                if imgui.button("Reset"):
                    g_paused = True
                    g_reset = True
                    g_timestep = 0
                    g_last_frame_time = time.time()

                imgui.same_line()
                if imgui.button("Step"):
                    g_timestep += 1
                    g_last_frame_time = time.time()

                if imgui.button(label="load ply"):
                    file_path = filedialog.askopenfilename(
                        title="load ply",
                        initialdir="C:\\Users",
                        filetypes=[("ply file", ".ply")],
                    )
                    if file_path:
                        try:
                            gaussians = util_gau.load_ply(file_path)
                            g_renderer.update_gaussian_data(gaussians)
                            g_renderer.sort_and_update(g_camera)
                            reset_mask_runtime_state()
                        except RuntimeError:
                            pass

                imgui.same_line()
                if imgui.button(label="save ply"):
                    file_path = filedialog.asksaveasfilename(
                        title="save ply",
                        initialdir="C:\\Users\\",
                        defaultextension=".ply",
                        filetypes=[("ply file", ".ply")],
                    )
                    if file_path:
                        try:
                            util_3dgstream.save_gau_cuda(g_renderer.gaussians, file_path)
                        except RuntimeError:
                            pass

                imgui.same_line()
                if imgui.button(label="load FVV"):
                    dirp = filedialog.askdirectory(title="load FVV", initialdir="C:\\Users")
                    if dirp:
                        try:
                            g_FVV_path = dirp
                            g_renderer.NTCs = util_3dgstream.load_NTCs(
                                g_FVV_path, g_renderer.gaussians, g_total_frame
                            )
                            g_renderer.additional_3dgs = util_3dgstream.load_Additions(
                                g_FVV_path, g_total_frame
                            )
                            reset_mask_runtime_state()
                        except RuntimeError:
                            pass

                changed, g_camera.fovy = imgui.slider_float(
                    "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
                )
                g_camera.is_intrin_dirty = changed
                update_camera_intrin_lazy()

                changed, g_scale_modifier = imgui.slider_float(
                    "", g_scale_modifier, 0.1, 10, "scale modifier = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset"):
                    g_scale_modifier = 1.0
                    changed = True
                if changed:
                    g_renderer.set_scale_modifier(g_scale_modifier)

                changed, g_render_mode = imgui.combo(
                    "shading", g_render_mode, g_render_mode_tables
                )
                if changed:
                    g_renderer.set_render_mod(g_render_mode - 4)

                if imgui.button(label="sort Gaussians"):
                    g_renderer.sort_and_update(g_camera)
                imgui.same_line()
                changed, g_auto_sort = imgui.checkbox("auto sort", g_auto_sort)
                if g_auto_sort:
                    g_renderer.sort_and_update(g_camera)

                if imgui.button(label="save image"):
                    width, height = glfw.get_framebuffer_size(window)
                    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                    gl.glReadBuffer(gl.GL_FRONT)
                    bufferdata = gl.glReadPixels(
                        0,
                        0,
                        width,
                        height,
                        gl.GL_RGB,
                        gl.GL_UNSIGNED_BYTE,
                    )
                    img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
                    imageio.imwrite("save.png", img[::-1])

                imgui.separator()
                imgui.text("Foveated overlay disabled for QNTC demo recording")

                imgui.end()

        if g_show_camera_win:
            imgui.begin("Camera Control", True)
            imgui.text("Camera state")
            imgui.separator()
            imgui.text(
                f"pos = ({g_camera.position[0]:.6f}, {g_camera.position[1]:.6f}, {g_camera.position[2]:.6f})"
            )
            imgui.text(
                f"target = ({g_camera.target[0]:.6f}, {g_camera.target[1]:.6f}, {g_camera.target[2]:.6f})"
            )
            imgui.text(
                f"up = ({g_camera.up[0]:.6f}, {g_camera.up[1]:.6f}, {g_camera.up[2]:.6f})"
            )
            imgui.text(f"yaw = {g_camera.yaw:.6f}")
            imgui.text(f"pitch = {g_camera.pitch:.6f}")
            imgui.text(f"fov_deg = {np.rad2deg(g_camera.fovy):.6f}")
            imgui.text(f"target_dist = {g_camera.target_dist:.6f}")
            imgui.separator()
            imgui.text("renderer.py args")
            imgui.text(f"--cam_pos={fmt_vec3(g_camera.position)}")
            imgui.text(f"--cam_target={fmt_vec3(g_camera.target)}")
            imgui.text(f"--cam_up={fmt_vec3(g_camera.up)}")
            imgui.text(f"--cam_fov_deg={np.rad2deg(g_camera.fovy):.6f}")

            if imgui.button("Print camera to console"):
                print_camera_state_to_console()

            imgui.separator()
            if imgui.button(label="rot 180"):
                g_camera.flip_ground()

            changed, g_camera.target_dist = imgui.slider_float(
                "t", g_camera.target_dist, 1.0, 8.0, "target dist = %.3f"
            )
            if changed:
                g_camera.update_target_distance()

            changed, g_camera.rot_sensitivity = imgui.slider_float(
                "r", g_camera.rot_sensitivity, 0.002, 0.1, "rotate speed = %.3f"
            )
            imgui.same_line()
            if imgui.button(label="reset r"):
                g_camera.rot_sensitivity = 0.02

            changed, g_camera.trans_sensitivity = imgui.slider_float(
                "m", g_camera.trans_sensitivity, 0.001, 0.03, "move speed = %.3f"
            )
            imgui.same_line()
            if imgui.button(label="reset m"):
                g_camera.trans_sensitivity = 0.01

            changed, g_camera.zoom_sensitivity = imgui.slider_float(
                "z", g_camera.zoom_sensitivity, 0.001, 0.05, "zoom speed = %.3f"
            )
            imgui.same_line()
            if imgui.button(label="reset z"):
                g_camera.zoom_sensitivity = 0.01

            changed, g_camera.roll_sensitivity = imgui.slider_float(
                "ro", g_camera.roll_sensitivity, 0.003, 0.1, "roll speed = %.3f"
            )
            imgui.same_line()
            if imgui.button(label="reset ro"):
                g_camera.roll_sensitivity = 0.03

            imgui.end()

        if g_show_help_win:
            imgui.begin("Help", True)
            imgui.text("Open Gaussian Splatting PLY file by clicking 'load ply'")
            imgui.text("Use left click & move to rotate camera")
            imgui.text("Use right click & move to translate camera")
            imgui.text("Press Q/E to roll camera")
            imgui.text("Use scroll to zoom in/out")
            imgui.text("Use control panel to change setting")
            imgui.text("Open 'Window -> Show Camera Control' to see live camera values")
            imgui.text("Use 'Print camera to console' to get renderer.py arguments")
            imgui.text("Use PRINT_CAMERA=1 to live-print and save camera JSON")
            imgui.text("Use FIXED_CAMERA_JSON + LOCK_CAMERA=1 for metric rendering")
            imgui.text(
                "Use --black_mask_dir to apply framewise/global base masks and add masks with opacity=0"
            )
            imgui.end()

        draw_qntc_demo_panel()

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
        if experiment is not None:
            now = time.monotonic()
            if live.err:
                raise RuntimeError(live.err)
            if selected is not None and g_renderer.current_timestep != selected:
                raise RuntimeError("renderer did not advance to the selected frame")
            experiment.present(now, selected)
            if transfer_started is not None and (
                (experiment.finished and (experiment_dir / "transfer-done.json").exists())
                or now - transfer_started >= args.priority_experiment_timeout
            ):
                report = experiment.report(transfer_started, now)
                with live._lock:
                    report["exact_ntc_uses"] = live.exact_ntc_uses
                    report["missing_ntc_uses"] = live.missing_ntc_uses
                if experiment.finished and report["exact_ntc_uses"] != experiment.updates:
                    raise RuntimeError("presented frame count differs from exact NTC applications")
                write_snapshot(experiment_dir / "playback.json", report)
                break
            glfw.wait_events_timeout(min(0.01, VIDEO_INTERVAL / 4))

    if live is not None:
        live.request_stop()

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny 3DGStream Viewer.")
    parser.add_argument("--hidpi", action="store_true")
    parser.add_argument("--video_fps", type=float, default=30.0)
    parser.add_argument("--tcp_listen", type=int, default=None)
    parser.add_argument("--tcp_bind", default="127.0.0.1")
    parser.add_argument("--tcp_cache", default=r"C:\tmp\gs_stream_cache\live_session")
    parser.add_argument("--tcp_clear_cache", action="store_true")
    parser.add_argument("--priority_listen", type=int, default=None,
                        help="receive bounded-priority QUIC files directly into this player")
    parser.add_argument("--priority_cache", default="data/cache/priority-player")
    parser.add_argument("--priority_experiment", help="directory for native playback experiment telemetry")
    parser.add_argument("--priority_prebuffer", type=float, default=1.0)
    parser.add_argument("--priority_experiment_timeout", type=float, default=240.0)
    parser.add_argument("--priority_cert")
    parser.add_argument("--priority_key")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--autoload_fvv", default="")
    parser.add_argument(
        "--black_mask_dir",
        default="",
        help="Folder containing base_black_framewise/, base_black_global.npy and add_black/",
    )
    parser.add_argument(
        "--base_black_min_count",
        type=int,
        default=1,
        help="If >1 and base_black_counts.npy exists, use counts >= this threshold for global fallback",
    )
    args = parser.parse_args()
    if args.priority_listen is not None:
        if args.tcp_listen is not None:
            parser.error("choose either TCP or priority QUIC reception")
        if not (0 if args.priority_experiment else 1) <= args.priority_listen <= 65535 or not args.priority_cert or not args.priority_key:
            parser.error("priority QUIC requires a valid port, --priority_cert and --priority_key")
    if args.priority_experiment and (
        args.priority_listen is None or args.frames < 2
        or not math.isfinite(args.priority_experiment_timeout)
        or args.priority_experiment_timeout <= 0
    ):
        parser.error("priority experiment requires QUIC, at least two frames, and a positive timeout")
    main()
