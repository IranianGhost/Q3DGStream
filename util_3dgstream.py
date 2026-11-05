# Load scene updates and decode NTC checkpoints for playback.

from __future__ import annotations

import os
import json
import glob
from typing import Optional, Dict, Any, List
from collections import OrderedDict

import numpy as np
import torch

from plyfile import PlyData, PlyElement
from NTC import NeuralTransformationCache
from util_gau import load_ply


def _unpack_signed_int4_to_float(packed, num_values, quantization="offset_signed_int4_packed"):
    # Restore two signed four-bit weights from each packed byte.
    """
    Unpacks packed signed INT4 values stored as uint8.

    Supported formats:
      1. offset_signed_int4_packed:
           stored = q_signed + 8
           q_signed = stored - 8

      2. symmetric_signed_int4_packed / twos_complement_signed_int4_packed:
           negative q values are stored as q + 16
           q_signed = stored if stored < 8 else stored - 16
    """
    packed = packed.detach().cpu().to(torch.uint8).reshape(-1)

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    out_u4 = torch.empty(packed.numel() * 2, dtype=torch.int16)
    out_u4[0::2] = low.to(torch.int16)
    out_u4[1::2] = high.to(torch.int16)

    out_u4 = out_u4[:int(num_values)]

    qmode = str(quantization).lower()

    if (
        "twos" in qmode
        or "two" in qmode
        or "symmetric_signed_int4_packed" in qmode
    ):
        out = torch.where(out_u4 >= 8, out_u4 - 16, out_u4)
    else:
        out = out_u4 - 8

    return out.to(torch.float32)


def _dequantize_int4_ntc_state_if_needed(state):
    """
    Supports checkpoints created by convert_ntc_int4.py.

    Converts:
      model.params_qint4_packed + model.params_int4_scales
    back into:
      model.params float32
    before loading into tiny-cuda-nn.
    """
    if not isinstance(state, dict):
        return state

    if "model.params_qint4_packed" not in state:
        return state

    packed = state["model.params_qint4_packed"]
    scales = state["model.params_int4_scales"]
    shape = state["model.params_int4_shape"]
    num_values = state["model.params_int4_num_values"]
    block_size = state["model.params_int4_block_size"]

    if torch.is_tensor(shape):
        shape = [int(v) for v in shape.reshape(-1).tolist()]
    else:
        shape = [int(v) for v in shape]

    if torch.is_tensor(num_values):
        num_values = int(num_values.reshape(-1)[0].item())
    else:
        num_values = int(num_values)

    if torch.is_tensor(block_size):
        block_size = int(block_size.reshape(-1)[0].item())
    else:
        block_size = int(block_size)

    
    quantization = state.get("model.params_int4_quantization", "offset_signed_int4_packed")
    if torch.is_tensor(quantization):
        quantization = "offset_signed_int4_packed"
    q = _unpack_signed_int4_to_float(packed, num_values, quantization=quantization)
    scales = scales.detach().cpu().to(torch.float32).reshape(-1)

    n = 1
    for v in shape:
        n *= int(v)

    pad = (-n) % block_size
    if pad:
        q_padded = torch.cat([q, torch.zeros(pad, dtype=torch.float32)])
    else:
        q_padded = q

    blocks = q_padded.reshape(-1, block_size)
    out = (blocks * scales[:, None]).reshape(-1)[:n].reshape(shape).contiguous()

    new_state = OrderedDict()

    for k, v in state.items():
        if k == "model.params_qint4_packed":
            continue
        if k.startswith("model.params_int4"):
            continue
        if k == "model.params_int4_quantization":
            continue
        new_state[k] = v

    new_state["model.params"] = out.float()
    return new_state



def _dequantize_int8_ntc_state_if_needed(state):
    """
    Supports checkpoints created by convert_ntc_int8.py.

    Converts:
      model.params_qint8 + model.params_int8_scales
    back into:
      model.params float32
    before loading into tiny-cuda-nn.
    """
    if not isinstance(state, dict):
        return state

    if "model.params_qint8" not in state:
        return state

    q = state["model.params_qint8"]
    scales = state["model.params_int8_scales"]
    shape = state["model.params_int8_shape"]
    block_size = state["model.params_int8_block_size"]

    if torch.is_tensor(shape):
        shape = [int(v) for v in shape.reshape(-1).tolist()]
    else:
        shape = [int(v) for v in shape]

    if torch.is_tensor(block_size):
        block_size = int(block_size.reshape(-1)[0].item())
    else:
        block_size = int(block_size)

    q = q.detach().cpu().to(torch.float32).reshape(-1)
    scales = scales.detach().cpu().to(torch.float32).reshape(-1)

    n = 1
    for v in shape:
        n *= int(v)

    pad = (-n) % block_size
    if pad:
        q_padded = torch.cat([q, torch.zeros(pad, dtype=torch.float32)])
    else:
        q_padded = q

    blocks = q_padded.reshape(-1, block_size)
    out = (blocks * scales[:, None]).reshape(-1)[:n].reshape(shape).contiguous()

    new_state = OrderedDict()

    for k, v in state.items():
        if k == "model.params_qint8":
            continue
        if k.startswith("model.params_int8"):
            continue
        if k == "model.params_int8_quantization":
            continue
        new_state[k] = v

    new_state["model.params"] = out.float()
    return new_state



@torch.no_grad()
def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 1e-6, 1.0 - 1e-6)
    return torch.log(x / (1 - x))


def construct_list_of_attributes(gau_cuda: GaussianDataCUDA) -> List[str]:
    l = ["x", "y", "z", "nx", "ny", "nz"]

    # DC SH coefficients
    for i in range(3):
        l.append(f"f_dc_{i}")

    # Remaining SH coefficients
    for i in range((gau_cuda.sh_dim - 1) * 3):
        l.append(f"f_rest_{i}")

    l.append("opacity")

    for i in range(gau_cuda.scale.shape[1]):
        l.append(f"scale_{i}")

    for i in range(gau_cuda.rot.shape[1]):
        l.append(f"rot_{i}")

    return l


def _normpath(p: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(p)))


def _torch_load_cpu(path: str):
    """
    Compatible torch.load wrapper.

    Some PyTorch versions support weights_only=True and some do not.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_config_json(fvv_path: str) -> Optional[str]:
    """
    Expected: <FVV_path>/NTCs/config.json
    Also checks a few nearby fallback locations.
    """
    fvv_path = _normpath(fvv_path)

    candidates = [
        os.path.join(fvv_path, "NTCs", "config.json"),
        os.path.join(fvv_path, "config.json"),
        os.path.join(os.path.dirname(fvv_path), "NTCs", "config.json"),
        os.path.join(os.path.dirname(fvv_path), "config.json"),
    ]

    for c in candidates:
        if os.path.isfile(c):
            return c

    return None


def _try_extract_config_from_ckpt(ckpt_obj: Any) -> Optional[Dict[str, Any]]:
    """
    If the trainer embedded encoding/network config in the checkpoint dict,
    recover it here. This only works if training saved it.
    """
    if isinstance(ckpt_obj, dict):
        # Common patterns
        for key in ("config", "cfg", "NTC_conf", "ntc_conf"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                d = ckpt_obj[key]
                if "encoding" in d and "network" in d:
                    return d

        # Sometimes stored flat
        if "encoding" in ckpt_obj and "network" in ckpt_obj:
            if isinstance(ckpt_obj["encoding"], dict) and isinstance(ckpt_obj["network"], dict):
                return {
                    "encoding": ckpt_obj["encoding"],
                    "network": ckpt_obj["network"],
                }

    return None


def _extract_state_dict(ckpt_obj: Any):
    """
    Supports both:
      - raw state dict checkpoints
      - checkpoints containing {"state_dict": ...}
    """
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        return ckpt_obj["state_dict"]

    return ckpt_obj


def _convert_fp16_tensors_to_fp32(state):
    """
    Convert FP16 tensors inside a state dict back to FP32.

    This allows compressed FP16 NTC checkpoints to stay smaller on disk while
    keeping tiny-cuda-nn / NeuralTransformationCache runtime compatible.
    """
    if not isinstance(state, dict):
        return state

    try:
        converted = type(state)()
    except Exception:
        converted = OrderedDict()

    for key, value in state.items():
        if torch.is_tensor(value) and value.dtype == torch.float16:
            converted[key] = value.float()
        else:
            converted[key] = value

    return converted


def _extract_bounds_from_ckpt(ckpt_obj: Any, device: torch.device):
    """
    Extract xyz bounds saved by 3DGStream inside the NTC checkpoint.

    Expected checkpoint keys:
      xyz_bound_min
      xyz_bound_max
      model.params

    These bounds are important because NeuralTransformationCache normalizes xyz
    coordinates using them. Recomputing different bounds from the loaded Gaussian
    cloud can make the NTC predict incorrect deformations.
    """
    if not isinstance(ckpt_obj, dict):
        return None, None

    if "xyz_bound_min" not in ckpt_obj or "xyz_bound_max" not in ckpt_obj:
        return None, None

    xyz_min = ckpt_obj["xyz_bound_min"]
    xyz_max = ckpt_obj["xyz_bound_max"]

    if not torch.is_tensor(xyz_min):
        xyz_min = torch.tensor(xyz_min)

    if not torch.is_tensor(xyz_max):
        xyz_max = torch.tensor(xyz_max)

    xyz_min = xyz_min.float().to(device)
    xyz_max = xyz_max.float().to(device)

    return xyz_min, xyz_max


def load_NTCs(FVV_path: str, gau_cuda: GaussianDataCUDA, total_frames: Optional[int] = None):
    """
    Loads NTC_*.pth and builds a NeuralTransformationCache list.

    total_frames:
      - if None: auto = number_of_ntc_files + 1
      - else: loads min(total_frames - 1, found_files)

    Important fixes:
      1. Uses xyz_bound_min / xyz_bound_max from the NTC checkpoint when
         available. These are the bounds used by the NTC normalization.
      2. Supports compressed NTC checkpoints where model.params is stored as
         FP16. It converts FP16 tensors back to FP32 at load time.
    """
    fvv_path = _normpath(FVV_path)
    ntc_dir = os.path.join(fvv_path, "NTCs")

    ntc_paths = sorted(glob.glob(os.path.join(ntc_dir, "NTC_*.pth")))

    if len(ntc_paths) == 0:
        raise FileNotFoundError(
            f"No NTC_*.pth found in: {ntc_dir}\n"
            "Expected files like NTC_000000.pth, NTC_000001.pth, ..."
        )

    if total_frames is None:
        total_frames = len(ntc_paths) + 1

    # Frame 0 is init_3dgs. NTC files represent the following frame deltas.
    ntc_paths = ntc_paths[: max(0, total_frames - 1)]

    if len(ntc_paths) == 0:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load first checkpoint once. It may contain config and bounds.
    first_ckpt = _torch_load_cpu(ntc_paths[0])

    # Load config from config.json or from checkpoint.
    config_path = _find_config_json(fvv_path)
    ntc_conf: Optional[Dict[str, Any]] = None

    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            ntc_conf = json.load(f)

    if ntc_conf is None:
        ntc_conf = _try_extract_config_from_ckpt(first_ckpt)

    if ntc_conf is None:
        raise FileNotFoundError(
            "Could not find NTC config.\n\n"
            "The viewer needs the tinycudann configs (encoding + network), usually at:\n"
            f"  {os.path.join(fvv_path, 'NTCs', 'config.json')}\n\n"
            "But that file is missing, and your NTC_*.pth files do not appear to embed it.\n"
            "Fix: export/copy the config JSON that was used in training into that location."
        )

    if ("encoding" not in ntc_conf) or ("network" not in ntc_conf):
        raise ValueError(
            f"Invalid NTC config. Missing 'encoding' or 'network'. "
            f"Source: {config_path or 'checkpoint'}"
        )

    # Critical fix:
    # Prefer xyz bounds saved inside the checkpoint.
    xyz_min, xyz_max = _extract_bounds_from_ckpt(first_ckpt, device)

    if xyz_min is not None and xyz_max is not None:
        print("[NTC BOUNDS] using xyz_bound_min / xyz_bound_max from checkpoint")
        print("[NTC BOUNDS] xyz_min =", xyz_min.detach().cpu().numpy())
        print("[NTC BOUNDS] xyz_max =", xyz_max.detach().cpu().numpy())
    else:
        print("[NTC BOUNDS] checkpoint bounds not found; falling back to Gaussian quantile bounds")
        xyz_min, xyz_max = gau_cuda.get_xyz_bound()
        xyz_min = xyz_min.float().to(device)
        xyz_max = xyz_max.float().to(device)
        print("[NTC BOUNDS] fallback xyz_min =", xyz_min.detach().cpu().numpy())
        print("[NTC BOUNDS] fallback xyz_max =", xyz_max.detach().cpu().numpy())

    ntcs: List[NeuralTransformationCache] = []

    for _ in ntc_paths:
        import tinycudann as tcnn
        model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=8,
            encoding_config=ntc_conf["encoding"],
            network_config=ntc_conf["network"],
        ).to(device)

        model.eval()

        ntcs.append(
            NeuralTransformationCache(
                model,
                xyz_min,
                xyz_max,
            )
        )

    # Load NTC weights.
    for i, ntc in enumerate(ntcs):
        if i == 0:
            ckpt = first_ckpt
        else:
            ckpt = _torch_load_cpu(ntc_paths[i])

        state = _extract_state_dict(ckpt)
        state = _dequantize_int4_ntc_state_if_needed(state)
        state = _dequantize_int8_ntc_state_if_needed(state)
        state = _convert_fp16_tensors_to_fp32(state)
        ntc.load_state_dict(state, strict=False)

    return ntcs


def load_Additions(FVV_path: str, total_frames: Optional[int] = None):
    """
    Loads additions_*.ply if present.

    Returns:
      list[GaussianDataCUDA]

    If folder does not exist, returns [] so the viewer can still run.
    """
    fvv_path = _normpath(FVV_path)
    add_dir = os.path.join(fvv_path, "additional_3dgs")

    if not os.path.isdir(add_dir):
        return []

    add_paths = sorted(glob.glob(os.path.join(add_dir, "additions_*.ply")))

    if len(add_paths) == 0:
        return []

    if total_frames is None:
        total_frames = len(add_paths) + 1

    add_paths = add_paths[: max(0, total_frames - 1)]

    additions_gaus = [load_ply(p) for p in add_paths]
    from renderer_cuda import gaus_cuda_from_cpu
    additions_gaus_cuda = [gaus_cuda_from_cpu(g) for g in additions_gaus]

    return additions_gaus_cuda


def get_per_frame_3dgs(FVV_path, gau_cuda: GaussianDataCUDA, total_frames: int = 150):
    raise NotImplementedError("This function is not implemented yet")


def save_gau_cuda(gau_cuda: GaussianDataCUDA, path: str):
    xyz = gau_cuda.xyz.detach().cpu().numpy()
    rotation = gau_cuda.rot.detach().cpu().numpy()
    normals = np.zeros_like(xyz)

    f_dc = (
        gau_cuda.sh[:, 0:1, :]
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )

    f_rest = (
        gau_cuda.sh[:, 1:, :]
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )

    opacities = inverse_sigmoid(gau_cuda.opacity).detach().cpu().numpy()
    scale = torch.log(torch.clamp(gau_cuda.scale, min=1e-12)).detach().cpu().numpy()

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(gau_cuda)]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    attributes = np.concatenate(
        (
            xyz,
            normals,
            f_dc,
            f_rest,
            opacities,
            scale,
            rotation,
        ),
        axis=1,
    )

    elements[:] = list(map(tuple, attributes))

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)
