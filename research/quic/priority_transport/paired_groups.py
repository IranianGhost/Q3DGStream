"""Pack and reconstruct saliency-ranked paired INT2/INT4 affine NTC groups."""

import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from paired_affine import decode_to_dense_state, unpack_codes


MAGIC = b"QNPAIR01"
HEADER = struct.Struct("<8sBBIII")  # magic, bits, tier, update, record count, metadata length


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def record_type(bits):
    if bits not in (2, 4):
        raise ValueError("Expected INT2 or INT4")
    return np.dtype([("id", "<i4"), ("scale", "<f4"), ("zero", "u1"),
                     ("codes", "u1", (64 * bits // 8,))])


def build_groups(state, frame, ranked_ids):
    bits = int(state["bits"])
    if state.get("format") != "qntc_sparse_block_affine_v1" or int(state["block_size"]) != 64:
        raise ValueError("Unexpected paired affine checkpoint format")
    original = state["indices"].detach().cpu().numpy()
    scores = np.asarray(ranked_ids)
    if (original.dtype != np.int32 or original.ndim != 1 or not len(original)
            or np.any(np.diff(original.astype(np.int64)) <= 0)
            or scores.ndim != 1 or len(scores) != len(original)
            or not np.array_equal(np.sort(scores), original)):
        raise ValueError("Priority order does not cover the checkpoint's retained blocks")
    count = len(original)
    packed = state["codes"].detach().cpu().numpy()
    scales = state["scales"].detach().cpu().numpy()
    zeros = state["zero_points"].detach().cpu().numpy()
    code_bytes = 64 * bits // 8
    if (packed.dtype != np.uint8 or packed.size != count * code_bytes
            or scales.dtype != np.float32 or scales.shape != original.shape
            or not np.isfinite(scales).all() or np.any(scales <= 0)
            or zeros.dtype != np.uint8 or zeros.shape != original.shape or np.any(zeros >= 1 << bits)):
        raise ValueError("Invalid packed codes, block scales, or zero points")
    # The source indices are sorted. searchsorted maps saliency IDs to their
    # original packed records without modifying their bit-level contents.
    position = np.searchsorted(original, scores)
    records = np.empty(count, dtype=record_type(bits))
    records["id"] = original[position]
    records["scale"] = scales[position]
    records["zero"] = zeros[position]
    records["codes"] = packed.reshape(count, code_bytes)[position]
    shape = [int(n) for n in state["shape"]]
    if not shape or min(shape) <= 0 or original[-1] >= math.ceil(math.prod(shape) / 64):
        raise ValueError("Invalid dense parameter shape or retained block ID")
    bounds = {name: state[name].detach().cpu().to(dtype=__import__("torch").float32).reshape(-1).tolist()
              for name in ("xyz_bound_min", "xyz_bound_max")}
    if any(len(v) != 3 or not all(math.isfinite(x) for x in v) for v in bounds.values()):
        raise ValueError("Invalid normalization bounds")
    meta = {"format": "qntc_paired_groups_v1", "frame": frame, "bits": bits,
            "block_size": 64, "shape": shape, "retained_blocks": count, **bounds}
    raw_meta = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")
    lengths = [count // 4 + (i < count % 4) for i in range(4)]
    groups = []
    offset = 0
    for tier, length in enumerate(lengths):
        info = raw_meta if tier == 0 else b""
        body = records[offset:offset + length].tobytes()
        groups.append(HEADER.pack(MAGIC, bits, tier, frame, length, len(info)) + info + body)
        offset += length
    return groups


def parse_group(raw, frame, tier):
    if len(raw) < HEADER.size:
        raise ValueError("Truncated group header")
    magic, bits, found_tier, found_frame, count, metadata_bytes = HEADER.unpack_from(raw)
    if magic != MAGIC or bits not in (2, 4) or found_tier != tier or found_frame != frame:
        raise ValueError(f"Mismatched group header for NTC {frame} P{tier}")
    dtype = record_type(bits)
    if (metadata_bytes == 0) != (tier != 0):
        raise ValueError("Only P0 must carry reconstruction metadata")
    if len(raw) != HEADER.size + metadata_bytes + count * dtype.itemsize:
        raise ValueError(f"Invalid group size for NTC {frame} P{tier}")
    meta = json.loads(raw[HEADER.size:HEADER.size + metadata_bytes]) if tier == 0 else None
    records = np.frombuffer(raw, dtype=dtype, count=count, offset=HEADER.size + metadata_bytes)
    return bits, meta, records


def group_precision(folder, frame):
    with (folder / "P0.bin").open("rb") as stream:
        head = stream.read(HEADER.size)
    if len(head) != HEADER.size:
        raise ValueError(f"NTC {frame} has a truncated P0")
    magic, bits, tier, found_frame, _, _ = HEADER.unpack(head)
    if magic != MAGIC or bits not in (2, 4) or tier != 0 or found_frame != frame:
        raise ValueError(f"NTC {frame} has an invalid P0")
    return bits


def assemble(folder, frame_manifest, tiers):
    import torch

    if not isinstance(tiers, int) or isinstance(tiers, bool) or not 1 <= tiers <= 4:
        raise ValueError("Expected between one and four received priority groups")

    frame = int(frame_manifest["frame"])
    bits = group_precision(folder, frame)
    variant = frame_manifest["variants"][str(bits)]
    groups = []
    metadata = None
    for tier in range(tiers):
        raw = (folder / f"P{tier}.bin").read_bytes()
        expected = variant["groups"][f"P{tier}"]
        if len(raw) != expected["payload_bytes"] or sha256(raw) != expected["sha256"]:
            raise ValueError(f"Hash mismatch for NTC {frame} INT{bits} P{tier}")
        found_bits, current_meta, records = parse_group(raw, frame, tier)
        if found_bits != bits:
            raise ValueError(f"Mixed precision groups in NTC {frame}")
        if tier == 0:
            metadata = current_meta
        groups.append(records)
    if (metadata.get("format") != "qntc_paired_groups_v1" or metadata.get("bits") != bits
            or metadata.get("frame") != frame or metadata.get("block_size") != 64
            or metadata.get("retained_blocks") != frame_manifest["retained_blocks"]):
        raise ValueError(f"Invalid metadata in NTC {frame}")
    records = np.concatenate(groups)
    ids = records["id"].astype(np.int64)
    shape = tuple(int(n) for n in metadata["shape"])
    total = math.prod(shape)
    if (len(ids) > metadata["retained_blocks"] or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0) or np.any(ids >= math.ceil(total / 64))):
        raise ValueError(f"Invalid or duplicate blocks in NTC {frame}")
    if (not np.isfinite(records["scale"]).all() or np.any(records["scale"] <= 0)
            or np.any(records["zero"] >= 1 << bits)):
        raise ValueError(f"Invalid block quantization parameters in NTC {frame}")
    packed = np.ascontiguousarray(records["codes"]).reshape(-1)
    codes = unpack_codes(packed, len(ids) * 64, bits).reshape(-1, 64).astype(np.float32)
    values = (codes - records["zero"][:, None].astype(np.float32)) * records["scale"][:, None]
    dense = np.zeros((math.ceil(total / 64), 64), dtype=np.float32)
    dense[ids] = values
    result = {"model.params": torch.from_numpy(dense.reshape(-1)[:total].reshape(shape).copy()),
              "_precision_bits": bits}
    return result, len(ids)


def prepare_frame(int2_file, int4_file, order_file, destination, frame, *, block_index_order=False):
    import torch

    states = {bits: torch.load(file, map_location="cpu", weights_only=True)
              for bits, file in ((2, int2_file), (4, int4_file))}
    with np.load(order_file, allow_pickle=False) as order:
        ids = order["ordered_block_ids"].copy()
        scores = order["priority_scores"].copy()
        if int(order["ntc_index"]) != frame or np.any(np.diff(scores) > 0) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid saliency order for NTC {frame}")
        for bits, file in ((2, int2_file), (4, int4_file)):
            if sha256(file.read_bytes()) != str(order[f"int{bits}_sha256"]):
                raise ValueError(f"The paired INT{bits} checkpoint changed for NTC {frame}")
    if not torch.equal(states[2]["indices"], states[4]["indices"]):
        raise ValueError(f"Paired masks differ for NTC {frame}")
    if block_index_order:
        ids.sort()
    manifest = {"frame": frame, "retained_blocks": len(ids), "variants": {}}
    for bits in (2, 4):
        groups = build_groups(states[bits], frame, ids)
        folder = destination / f"int{bits}"
        folder.mkdir(parents=True)
        records = {}
        for tier, group in enumerate(groups):
            (folder / f"P{tier}.bin").write_bytes(group)
            _, _, decoded = parse_group(group, frame, tier)
            records[f"P{tier}"] = {"payload_bytes": len(group), "block_count": len(decoded),
                                    "sha256": sha256(group)}
        manifest["variants"][str(bits)] = {"groups": records,
                                           "payload_bytes_total": sum(map(len, groups))}
        # Prove lossless group reconstruction against the original affine wire.
        reconstructed, _ = assemble(folder, manifest, 4)
        source = decode_to_dense_state(states[bits], expected_bits=bits)["model.params"]
        if not torch.equal(source, reconstructed["model.params"]):
            raise ValueError(f"INT{bits} group reconstruction differs for NTC {frame}")
    return manifest
