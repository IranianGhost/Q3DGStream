"""
Part of the code (CUDA and OpenGL memory transfer) is derived from:
https://github.com/jbaron34/torchwindow/tree/master
"""

# CUDA renderer for dynamic Gaussian motion and rasterization.

import os

from OpenGL import GL as gl
import OpenGL.GL.shaders as shaders  # kept in case other files import it

import util
import util_gau
import numpy as np
import torch

from PIL import Image
from renderer_ogl import GaussianRenderBase
from dataclasses import dataclass
from cuda import cudart as cu
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from raster_compat import compatible_settings


VERTEX_SHADER_SOURCE = """
#version 450

smooth out vec4 fragColor;
smooth out vec2 texcoords;

vec4 positions[3] = vec4[3](
    vec4(-1.0, 1.0, 0.0, 1.0),
    vec4(3.0, 1.0, 0.0, 1.0),
    vec4(-1.0, -3.0, 0.0, 1.0)
);

vec2 texpos[3] = vec2[3](
    vec2(0, 0),
    vec2(2, 0),
    vec2(0, 2)
);

void main() {
    gl_Position = positions[gl_VerID];
    texcoords = texpos[gl_VertexID];
}
"""

# Corrected shader source. Keep this one active.
VERTEX_SHADER_SOURCE = """
#version 450

smooth out vec4 fragColor;
smooth out vec2 texcoords;

vec4 positions[3] = vec4[3](
    vec4(-1.0, 1.0, 0.0, 1.0),
    vec4(3.0, 1.0, 0.0, 1.0),
    vec4(-1.0, -3.0, 0.0, 1.0)
);

vec2 texpos[3] = vec2[3](
    vec2(0, 0),
    vec2(2, 0),
    vec2(0, 2)
);

void main() {
    gl_Position = positions[gl_VertexID];
    texcoords = texpos[gl_VertexID];
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330

smooth in vec2 texcoords;

out vec4 outputColour;

uniform sampler2D texSampler;

void main()
{
    outputColour = texture(texSampler, texcoords);
}
"""


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()

    if value == "":
        return bool(default)

    return value in ("1", "true", "yes", "on", "y")


def _safe_normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalizes quaternions and replaces invalid or zero quaternions with identity.
    """
    if q.numel() == 0:
        return q

    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    norm = torch.linalg.norm(q, dim=1, keepdim=True)

    identity = torch.zeros_like(q)
    identity[:, 0] = 1.0

    q_norm = q / torch.clamp(norm, min=eps)
    q_norm = torch.where(norm > eps, q_norm, identity)

    return q_norm


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiplies quaternions in [w, x, y, z] format.
    """
    a_norm = _safe_normalize_quaternion(a)
    b_norm = _safe_normalize_quaternion(b)

    w1, x1, y1, z1 = a_norm[:, 0], a_norm[:, 1], a_norm[:, 2], a_norm[:, 3]
    w2, x2, y2, z2 = b_norm[:, 0], b_norm[:, 1], b_norm[:, 2], b_norm[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    out = torch.stack([w, x, y, z], dim=1)
    return _safe_normalize_quaternion(out)


def quaternion_scaled_delta(q: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Creates a damped version of an NTC rotation delta.

    scale = 0.0 -> identity rotation
    scale = 1.0 -> original q

    This is kept for diagnostics, but the stable flame_steak setting uses
    ntc_rotation_mode = "none".
    """
    q = _safe_normalize_quaternion(q)

    identity = torch.zeros_like(q)
    identity[:, 0] = 1.0

    out = (1.0 - float(scale)) * identity + float(scale) * q
    return _safe_normalize_quaternion(out)


@dataclass
class GaussianDataCUDA:
    xyz: torch.Tensor
    rot: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh: torch.Tensor

    def __len__(self):
        return len(self.xyz)

    @property
    def sh_dim(self):
        # sh is [N, sh_dim, 3]
        return self.sh.shape[-2]

    @torch.no_grad()
    def get_xyz_bound(self, percentile=86.6):
        half_percentile = (100 - percentile) / 200
        return (
            torch.quantile(self.xyz, half_percentile, dim=0),
            torch.quantile(self.xyz, 1 - half_percentile, dim=0),
        )

    def clone(self):
        return GaussianDataCUDA(
            xyz=self.xyz.clone(),
            rot=self.rot.clone(),
            scale=self.scale.clone(),
            opacity=self.opacity.clone(),
            sh=self.sh.clone(),
        )


@dataclass
class GaussianRasterizationSettingsStorage:
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


def gaus_cuda_from_cpu(gau) -> GaussianDataCUDA:
    """
    Converts util_gau.GaussianData CPU object to CUDA tensors.
    Expected fields: xyz, rot, scale, opacity, sh.
    """
    gaus = GaussianDataCUDA(
        xyz=torch.tensor(gau.xyz).float().cuda().requires_grad_(False),
        rot=torch.tensor(gau.rot).float().cuda().requires_grad_(False),
        scale=torch.tensor(gau.scale).float().cuda().requires_grad_(False),
        opacity=torch.tensor(gau.opacity).float().cuda().requires_grad_(False),
        sh=torch.tensor(gau.sh).float().cuda().requires_grad_(False),
    )

    gaus.rot = _safe_normalize_quaternion(gaus.rot)

    # Ensure SH is [N, sh_dim, 3]
    gaus.sh = gaus.sh.reshape(len(gaus), -1, 3).contiguous()

    return gaus


class CUDARenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()

        self.raster_settings = {
            "image_height": int(h),
            "image_width": int(w),
            "tanfovx": 1.0,
            "tanfovy": 1.0,
            "bg": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"),
            "scale_modifier": 1.0,
            "viewmatrix": None,
            "projmatrix": None,
            "sh_degree": 1,
            "campos": None,
            "prefiltered": False,
            "antialiasing": False,
            "debug": False,
        }

        gl.glViewport(0, 0, int(w), int(h))
        self.program = util.compile_shaders(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)

        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise RuntimeError("OpenGL context may be running on integrated graphics")

        self.vao = gl.glGenVertexArrays(1)
        self.tex = None
        self.cuda_image = None

        self.NTCs = []
        self.additional_3dgs = []
        self.current_timestep = 0

        self.gaussians = None
        self.init_gaussians = None

        # Dump rendered PNG frames when this environment variable is set.
        # Example:
        #   export DUMP_FRAMES_DIR=/path/to/renders_eval/fp32_stable
        self.dump_frames_dir = os.environ.get("DUMP_FRAMES_DIR", "").strip()
        self.dumped_timesteps = set()

        if self.dump_frames_dir:
            os.makedirs(self.dump_frames_dir, exist_ok=True)
            print("[DUMP] saving rendered frames to:", self.dump_frames_dir)

        # Use NTC mask when it has the expected shape.
        self.use_ntc_mask = True

        # Enable NTC diagnostics with:
        #   set NTC_DEBUG=1
        # Disable with:
        #   set NTC_DEBUG=0
        #
        # Default is True here because we are debugging FP32/INT8/INT4 motion.
        self.ntc_debug_mask = _env_bool("NTC_DEBUG", default=True)

        # Print detailed d_xyz motion magnitude every N frames.
        self.ntc_debug_dxyz = _env_bool("NTC_DEBUG_DXYZ", default=True)
        self.ntc_debug_interval = int(os.environ.get("NTC_DEBUG_INTERVAL", "50"))

        # Stable flame_steak setting:
        # - "current" caused strong deformation.
        # - "reverse" looked bad.
        # - "scaled" still left motion-region artifacts.
        # - "none" with translation 0.30 looked stable.
        self.ntc_rotation_mode = "none"

        # Kept for diagnostics if you manually switch ntc_rotation_mode to "scaled".
        self.ntc_rotation_scale = 0.08

        # Stable flame_steak setting.
        self.ntc_translation_scale = 0.42

        # Additions are not used in the stable no-additions scene.
        # Kept here only so original/additions scenes still run if loaded.
        self.additions_opacity_scale = 1.0
        self.additions_scale_scale = 1.0

        self.set_gl_texture(h, w)

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def update_gaussian_data(self, gaus):
        self.gaussians = gaus_cuda_from_cpu(gaus)
        self.init_gaussians = self.gaussians.clone()
        self.current_timestep = 0

        self.raster_settings["sh_degree"] = int(np.round(np.sqrt(self.gaussians.sh_dim))) - 1

    def sort_and_update(self, camera: util.Camera):
        pass

    def set_scale_modifier(self, modifier):
        self.raster_settings["scale_modifier"] = float(modifier)

    def set_render_mod(self, mod: int):
        pass

    def _unregister_current_texture(self):
        if getattr(self, "cuda_image", None) is not None:
            try:
                cu.cudaGraphicsUnregisterResource(self.cuda_image)
            except Exception:
                pass
            self.cuda_image = None

        if getattr(self, "tex", None) is not None:
            try:
                gl.glDeleteTextures([self.tex])
            except Exception:
                pass
            self.tex = None

    def set_gl_texture(self, h, w):
        h = int(h)
        w = int(w)

        if w <= 0 or h <= 0:
            raise RuntimeError(f"Invalid texture size: {w}x{h}")

        self._unregister_current_texture()

        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            w,
            h,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        err, cuda_image = cu.cudaGraphicsGLRegisterImage(
            self.tex,
            gl.GL_TEXTURE_2D,
            cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )

        if err != cu.cudaError_t.cudaSuccess:
            try:
                gl.glDeleteTextures([self.tex])
            except Exception:
                pass
            self.tex = None
            self.cuda_image = None
            raise RuntimeError("Unable to register opengl texture")

        self.cuda_image = cuda_image

    def set_render_reso(self, w, h):
        """
        Safe resize handler.

        GLFW can emit resize events while the window is being created,
        minimized, restored, or closed. Re-registering CUDA/GL textures at
        exactly that moment can fail with 'Unable to register opengl texture'.
        This version ignores invalid sizes, avoids unnecessary re-registration,
        and keeps the previous texture if a resize registration fails.
        """
        w = int(w)
        h = int(h)

        if w <= 0 or h <= 0:
            return

        old_w = int(self.raster_settings.get("image_width", 0) or 0)
        old_h = int(self.raster_settings.get("image_height", 0) or 0)

        gl.glViewport(0, 0, w, h)

        if old_w == w and old_h == h:
            return

        try:
            self.set_gl_texture(h, w)
            self.raster_settings["image_height"] = h
            self.raster_settings["image_width"] = w
        except RuntimeError as e:
            print(f"[RESIZE WARN] Could not re-register CUDA/GL texture after resize: {e}")
            print("[RESIZE WARN] Keeping previous texture. Avoid resizing the window during this test.")
            self.raster_settings["image_height"] = old_h
            self.raster_settings["image_width"] = old_w

    def _prepare_ntc_mask(self, mask: torch.Tensor, expected_n: int):
        if mask is None or not torch.is_tensor(mask):
            return None

        m = mask

        while m.dim() > 1 and m.shape[-1] == 1:
            m = m.squeeze(-1)

        m = m.reshape(-1)

        if m.numel() != expected_n:
            return None

        if m.dtype != torch.bool:
            if torch.is_floating_point(m):
                m = m > 0.5
            else:
                m = m != 0

        return m

    def _apply_rotation_update(self, valid_mask, d_rot):
        """
        Applies NTC rotation update according to self.ntc_rotation_mode.

        Modes:
          none    -> no rotation update
          current -> rot = rot * d_rot
          reverse -> rot = d_rot * rot
          scaled  -> rot = rot * partial(d_rot)
        """
        mode = self.ntc_rotation_mode.lower().strip()

        if mode == "none":
            return

        if mode == "scaled":
            d_rot_to_apply = quaternion_scaled_delta(d_rot, self.ntc_rotation_scale)
        else:
            d_rot_to_apply = d_rot

        if valid_mask is not None:
            if not torch.any(valid_mask):
                return

            if mode == "current" or mode == "scaled":
                self.gaussians.rot[valid_mask] = quaternion_multiply(
                    self.gaussians.rot[valid_mask],
                    d_rot_to_apply[valid_mask],
                )
            elif mode == "reverse":
                self.gaussians.rot[valid_mask] = quaternion_multiply(
                    d_rot_to_apply[valid_mask],
                    self.gaussians.rot[valid_mask],
                )
            else:
                raise ValueError(f"Unknown ntc_rotation_mode: {self.ntc_rotation_mode}")

            return

        if mode == "current" or mode == "scaled":
            self.gaussians.rot = quaternion_multiply(self.gaussians.rot, d_rot_to_apply)
        elif mode == "reverse":
            self.gaussians.rot = quaternion_multiply(d_rot_to_apply, self.gaussians.rot)
        else:
            raise ValueError(f"Unknown ntc_rotation_mode: {self.ntc_rotation_mode}")

    def _print_ntc_dxyz_stats(self, timestep: int, d_xyz: torch.Tensor, valid_mask):
        """
        Prints NTC displacement magnitude diagnostics.

        This tells us whether quantization collapsed the NTC motion.
        """
        if not self.ntc_debug_dxyz:
            return

        interval = max(1, int(self.ntc_debug_interval))

        if timestep % interval != 0:
            return

        with torch.no_grad():
            d = torch.nan_to_num(d_xyz.detach(), nan=0.0, posinf=0.0, neginf=0.0)

            all_norm = torch.linalg.norm(d, dim=1)

            msg = [
                "[NTC DXYZ]",
                f"timestep={timestep}",
                f"mean_abs={float(d.abs().mean().item()):.8f}",
                f"max_abs={float(d.abs().max().item()):.8f}",
                f"mean_norm={float(all_norm.mean().item()):.8f}",
                f"max_norm={float(all_norm.max().item()):.8f}",
            ]

            if valid_mask is not None and torch.any(valid_mask):
                dv = d[valid_mask]
                valid_norm = torch.linalg.norm(dv, dim=1)
                msg.extend(
                    [
                        f"valid_mean_abs={float(dv.abs().mean().item()):.8f}",
                        f"valid_max_abs={float(dv.abs().max().item()):.8f}",
                        f"valid_mean_norm={float(valid_norm.mean().item()):.8f}",
                        f"valid_max_norm={float(valid_norm.max().item()):.8f}",
                    ]
                )

            print(" ".join(msg), flush=True)

    @torch.no_grad()
    def query_NTC(self, xyz: torch.Tensor, timestep: int):
        # Predict the current step's Gaussian motion.
        if self.NTCs is None or len(self.NTCs) == 0:
            return

        if timestep < 0 or timestep >= len(self.NTCs):
            return

        mask, d_xyz, d_rot = self.NTCs[timestep](xyz)

        n = self.gaussians.xyz.shape[0]

        if d_xyz is None or d_rot is None:
            return

        if d_xyz.shape[0] != n or d_rot.shape[0] != n:
            print(
                f"[NTC WARN] timestep={timestep} unexpected delta shapes: "
                f"d_xyz={tuple(d_xyz.shape)}, d_rot={tuple(d_rot.shape)}, n={n}. Skipping."
            )
            return

        d_xyz = torch.nan_to_num(d_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        d_rot = _safe_normalize_quaternion(d_rot)

        valid_mask = self._prepare_ntc_mask(mask, n) if self.use_ntc_mask else None

        if self.ntc_debug_mask and timestep % max(1, int(self.ntc_debug_interval)) == 0:
            if valid_mask is None:
                print(f"[NTC MASK] timestep={timestep} invalid/no mask; applying full update")
            else:
                active = float(valid_mask.float().mean().item())
                print(
                    f"[NTC MASK] timestep={timestep} "
                    f"shape={tuple(mask.shape) if torch.is_tensor(mask) else None} "
                    f"dtype={mask.dtype if torch.is_tensor(mask) else None} "
                    f"active_fraction={active:.6f}",
                    flush=True,
                )

        self._print_ntc_dxyz_stats(timestep, d_xyz, valid_mask)

        if valid_mask is not None:
            if torch.any(valid_mask):
                self.gaussians.xyz[valid_mask] = (
                    self.gaussians.xyz[valid_mask]
                    + self.ntc_translation_scale * d_xyz[valid_mask]
                )
                self._apply_rotation_update(valid_mask, d_rot)
            return

        self.gaussians.xyz = self.gaussians.xyz + self.ntc_translation_scale * d_xyz
        self._apply_rotation_update(None, d_rot)

    @torch.no_grad()
    def _pad_sh_to(self, sh: torch.Tensor, target_sh_dim: int) -> torch.Tensor:
        if sh.shape[1] == target_sh_dim:
            return sh

        if sh.shape[1] > target_sh_dim:
            return sh[:, :target_sh_dim, :].contiguous()

        pad = torch.zeros(
            (sh.shape[0], target_sh_dim - sh.shape[1], sh.shape[2]),
            device=sh.device,
            dtype=sh.dtype,
        )
        return torch.cat([sh, pad], dim=1).contiguous()

    @torch.no_grad()
    def cat_additions(self, timestep: int) -> GaussianDataCUDA:
        if self.additional_3dgs is None or len(self.additional_3dgs) == 0:
            return self.gaussians

        if timestep < 0 or timestep >= len(self.additional_3dgs):
            return self.gaussians

        additions = self.additional_3dgs[timestep]

        additions.rot = _safe_normalize_quaternion(additions.rot)
        self.gaussians.rot = _safe_normalize_quaternion(self.gaussians.rot)

        sh_add = additions.sh
        sh_base = self.gaussians.sh

        if sh_add.dim() != 3 or sh_base.dim() != 3:
            raise RuntimeError(f"Unexpected SH shapes: additions={sh_add.shape}, base={sh_base.shape}")

        if sh_add.shape[1] != sh_base.shape[1]:
            target = max(sh_add.shape[1], sh_base.shape[1])
            sh_add = self._pad_sh_to(sh_add, target)
            sh_base = self._pad_sh_to(sh_base, target)

        add_opacity = torch.clamp(
            additions.opacity * self.additions_opacity_scale,
            min=0.0,
            max=1.0,
        )

        add_scale = torch.clamp(
            additions.scale * self.additions_scale_scale,
            min=1e-8,
        )

        return GaussianDataCUDA(
            xyz=torch.cat([additions.xyz, self.gaussians.xyz], dim=0),
            rot=torch.cat([additions.rot, self.gaussians.rot], dim=0),
            scale=torch.cat([add_scale, self.gaussians.scale], dim=0),
            opacity=torch.cat([add_opacity, self.gaussians.opacity], dim=0),
            sh=torch.cat([sh_add, sh_base], dim=0),
        )

    def fvv_reset(self):
        if self.init_gaussians is not None:
            self.gaussians = self.init_gaussians.clone()
        self.current_timestep = 0

    def update_camera_pose(self, camera: util.Camera):
        view_matrix = camera.get_view_matrix()
        view_matrix[[0, 2], :] = -view_matrix[[0, 2], :]
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings["viewmatrix"] = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings["campos"] = torch.tensor(camera.position).float().cuda()
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()

    def update_camera_intrin(self, camera: util.Camera):
        view_matrix = camera.get_view_matrix()
        view_matrix[[0, 2], :] = -view_matrix[[0, 2], :]
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()
        hfovx, hfovy, focal = camera.get_htanfovxy_focal()
        self.raster_settings["tanfovx"] = hfovx
        self.raster_settings["tanfovy"] = hfovy

    def _rasterize(self, rendered_gaussians: GaussianDataCUDA):
        raster_settings = compatible_settings(GaussianRasterizationSettings, self.raster_settings)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        out = rasterizer(
            means3D=rendered_gaussians.xyz,
            means2D=None,
            shs=rendered_gaussians.sh,
            colors_precomp=None,
            opacities=rendered_gaussians.opacity,
            scales=rendered_gaussians.scale,
            rotations=rendered_gaussians.rot,
            cov3D_precomp=None,
        )

        if isinstance(out, (tuple, list)):
            img = out[0]
            radii = out[1] if len(out) > 1 else None
            return img, radii

        if isinstance(out, dict):
            img = out.get("render", out.get("image", None))
            radii = out.get("radii", None)
            if img is None:
                raise RuntimeError(f"Unexpected rasterizer dict keys: {list(out.keys())}")
            return img, radii

        img = getattr(out, "render", None)
        if img is None:
            img = getattr(out, "image", None)

        radii = getattr(out, "radii", None)

        if img is None:
            raise RuntimeError(f"Unexpected rasterizer return type: {type(out)}")

        return img, radii

    def _max_valid_timestep(self) -> int:
        max_t = 0

        if self.NTCs is not None and len(self.NTCs) > 0:
            max_t = max(max_t, len(self.NTCs))

        if self.additional_3dgs is not None and len(self.additional_3dgs) > 0:
            max_t = max(max_t, len(self.additional_3dgs))

        return max_t

    def _maybe_dump_frame(self, img: torch.Tensor, timestep: int):
        """
        Saves the current rendered RGB image to DUMP_FRAMES_DIR as PNG.

        img is expected to be H x W x 4 float tensor in [0, 1] after alpha concat.
        """
        if not self.dump_frames_dir:
            return

        if timestep in self.dumped_timesteps:
            return

        rgb = img[..., :3]
        rgb = torch.clamp(rgb, 0.0, 1.0)
        rgb_u8 = (rgb * 255.0).byte().detach().cpu().numpy()

        out_path = os.path.join(self.dump_frames_dir, f"{int(timestep):06d}.png")
        Image.fromarray(rgb_u8).save(out_path, compress_level=0)

        self.dumped_timesteps.add(timestep)

        if timestep % 25 == 0:
            print("[DUMP]", out_path)

    def draw(self, timestep: int = 0):
        if self.gaussians is None:
            return

        timestep = int(timestep)
        timestep = max(0, min(timestep, self._max_valid_timestep()))

        if timestep < self.current_timestep:
            self.fvv_reset()

        with torch.no_grad():
            while timestep - self.current_timestep > 0:
                self.query_NTC(self.gaussians.xyz, self.current_timestep)
                self.current_timestep += 1

            if self.current_timestep != 0:
                rendered_gaussians = self.cat_additions(self.current_timestep - 1)
            else:
                rendered_gaussians = self.gaussians

            img, _radii = self._rasterize(rendered_gaussians)

        img = img.permute(1, 2, 0)
        img = torch.concat([img, torch.ones_like(img[..., :1])], dim=-1)
        img = img.contiguous()

        self._maybe_dump_frame(img, timestep)

        height, width = img.shape[:2]

        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to map graphics resource")

        err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            try:
                cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
            except Exception:
                pass
            raise RuntimeError("Unable to get mapped array")

        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            img.data_ptr(),
            4 * 4 * int(width),
            4 * 4 * int(width),
            int(height),
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            try:
                cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
            except Exception:
                pass
            raise RuntimeError("Unable to copy from tensor to texture")

        (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to unmap graphics resource")

        gl.glUseProgram(self.program)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
