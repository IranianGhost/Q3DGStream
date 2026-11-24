"""The V11 block-affine INT2 projection extended to 4 bits with the same rule."""

import math

import numpy as np


FORMAT = "qntc_sparse_block_affine_v1"


def pack_codes(codes, bits):
    if bits not in (2, 4):
        raise ValueError("Weight precision must be 2 or 4")
    codes = np.asarray(codes)
    if codes.dtype != np.uint8 or np.any(codes >= (1 << bits)):
        raise ValueError(f"Invalid unsigned {bits}-bit weight codes")
    per_byte = 8 // bits
    flat = codes.reshape(-1)
    padded = np.pad(flat, (0, (-flat.size) % per_byte))
    shifts = np.arange(per_byte, dtype=np.uint8) * bits
    return np.bitwise_or.reduce(padded.reshape(-1, per_byte) << shifts, axis=1)


def unpack_codes(packed, count, bits):
    if bits not in (2, 4):
        raise ValueError("Weight precision must be 2 or 4")
    per_byte = 8 // bits
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or count < 1 or packed.size != (count + per_byte - 1) // per_byte:
        raise ValueError("Packed weight buffer has an invalid size or type")
    shifts = np.arange(per_byte, dtype=np.uint8) * bits
    codes = ((packed.reshape(-1, 1) >> shifts) & ((1 << bits) - 1)).reshape(-1)
    if np.any(codes[count:]):
        raise ValueError("Nonzero padding codes")
    return codes[:count]


def expand_sparse(packed, scales, zero_points, indices, shape, block_size, bits):
    count = math.prod(shape)
    if not shape or min(shape) < 1 or block_size < 1:
        raise ValueError("Invalid packed weight shape or block size")
    total = (count + block_size - 1) // block_size
    if (
        indices.ndim != 1 or not len(indices) or indices.dtype.kind != "i"
        or indices[0] < 0 or indices[-1] >= total or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError("Expected sorted, unique kept block indices within range")
    if (
        scales.shape != indices.shape or scales.dtype != np.float32
        or not np.isfinite(scales).all() or np.any(scales <= 0)
        or zero_points.shape != indices.shape or zero_points.dtype != np.uint8
        or np.any(zero_points >= (1 << bits))
    ):
        raise ValueError("Invalid affine scales or zero points")
    codes = unpack_codes(packed, len(indices) * block_size, bits)
    active = codes.reshape(-1, block_size).astype(np.float32)
    active = (active - zero_points[:, None].astype(np.float32)) * scales[:, None]
    if not np.isfinite(active).all():
        raise ValueError("Affine decode produced nonfinite weights")
    dense = np.zeros((total, block_size), dtype=np.float32)
    dense[indices] = active
    return dense.reshape(-1)[:count].reshape(shape)


def project(params, mask, block_size, bits):
    import torch
    from torch.nn.functional import pad

    if bits not in (2, 4):
        raise ValueError("Weight precision must be 2 or 4")
    with torch.no_grad():
        flat = params.detach().float().reshape(-1)
        kept = mask.reshape(-1).bool()
        padding = (-flat.numel()) % block_size
        blocks = pad(flat * kept, (0, padding)).reshape(-1, block_size)
        indices = pad(kept, (0, padding)).reshape(-1, block_size).any(-1).nonzero().flatten()
        active = blocks[indices]
        if not indices.numel() or not torch.isfinite(active).all():
            raise ValueError("Expected nonempty, finite kept weight blocks")
        qmax = (1 << bits) - 1
        low = active.amin(-1).clamp_max(0)
        high = active.amax(-1).clamp_min(0)
        scales = ((high - low) / qmax).clamp_min(1e-12)
        zero_points = (-low / scales).round().clamp(0, qmax).to(torch.uint8)
        codes = (active / scales[:, None] + zero_points[:, None].float()).round().clamp(0, qmax)
        codes = codes.to(torch.uint8)
        dense = torch.zeros_like(blocks)
        dense[indices] = (codes.float() - zero_points[:, None].float()) * scales[:, None]
        values = (dense.reshape(-1)[:flat.numel()] * kept).reshape_as(params)
        return values, (indices, codes, scales, zero_points)


def encode_dense_state(state, bits, block_size, indices, projection):
    import torch

    params = state["model.params"].detach().cpu()
    selected, codes, scales, zero_points = (tensor.detach().cpu() for tensor in projection)
    if not torch.equal(indices.cpu().long(), selected):
        raise ValueError("The packed and trained block masks differ")
    packed = torch.from_numpy(pack_codes(codes.numpy(), bits))
    wire = dict(
        format=FORMAT, bits=bits, block_size=block_size, shape=tuple(params.shape),
        indices=selected.to(torch.int32), codes=packed, scales=scales, zero_points=zero_points,
        xyz_bound_min=state["xyz_bound_min"], xyz_bound_max=state["xyz_bound_max"],
    )
    decoded = decode_to_dense_state(wire, expected_bits=bits)["model.params"]
    if not torch.equal(decoded.to(params.dtype), params):
        raise ValueError("Packed weights do not reproduce the native forward weights")
    return wire


def decode_to_dense_state(wire, expected_bits=None):
    import torch

    bits = wire["bits"]
    if wire["format"] != FORMAT or bits not in (2, 4) or (
        expected_bits is not None and bits != expected_bits
    ):
        raise ValueError("Unexpected packed weight format or precision")
    dense = expand_sparse(
        wire["codes"].numpy(), wire["scales"].numpy(), wire["zero_points"].numpy(),
        wire["indices"].numpy(), wire["shape"], wire["block_size"], bits,
    )
    return {
        "model.params": torch.from_numpy(dense),
        "xyz_bound_min": wire["xyz_bound_min"],
        "xyz_bound_max": wire["xyz_bound_max"],
    }


def code_usage(wire):
    count = wire["indices"].numel() * wire["block_size"]
    codes = unpack_codes(wire["codes"].numpy(), count, wire["bits"])
    return {
        str(code): int(number)
        for code, number in enumerate(np.bincount(codes, minlength=1 << wire["bits"]))
    }
