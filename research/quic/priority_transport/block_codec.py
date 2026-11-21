"""Lossless, self-describing PQAT INT4 block transport over 4 Quinn streams.

No model architecture or visual saliency is invented here. `ordered_block_ids`
from the existing 3D saliency artifact selects the P0..P3 group of each
retained 64-value quantization block. The *actual* packed bytes, FP32 scales,
and int32 block IDs from the PQAT checkpoint become 40-byte records.
"""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

MAGIC = b"QNBLK001"
HEADER = struct.Struct("<8sBII") # magic, priority, record_count, metadata bytes
RECORD = struct.Struct("<if32s") # block ID, scale, 64 packed INT4 values
PACKED_KEY = "model.params_sparse_qint4_packed"
SCALES_KEY = "model.params_sparse_int4_scales"
INDICES_KEY = "model.params_sparse_int4_block_indices"
ARRAY_KEYS = {PACKED_KEY, SCALES_KEY, INDICES_KEY}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tensor_bytes(value):
    return value.detach().cpu().contiguous().numpy().tobytes()


def state_hash(state):
    """Canonical ordered field/tensor digest (does not depend on torch.save ZIP bytes)."""
    h = hashlib.sha256()
    for key, value in state.items():
        h.update(key.encode("utf-8") + b"\0")
        if hasattr(value, "detach"):
            h.update(str(value.dtype).encode("ascii") + b"\0")
            h.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
            h.update(tensor_bytes(value))
        elif isinstance(value, str):
            h.update(b"str\0" + value.encode("utf-8"))
        else:
            raise ValueError(f"Unsupported checkpoint field {key}: {type(value)}")
    return h.hexdigest()


def encode_other_field(value):
    if hasattr(value, "detach"):
        return {
            "type": "tensor", "dtype": str(value.dtype).split(".")[-1],
            "shape": list(value.shape), "bytes_hex": tensor_bytes(value).hex(),
        }
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise ValueError(f"Unsupported metadata type: {type(value)}")


def decode_other_field(field):
    if field["type"] == "str":
        return field["value"]
    if field["type"] != "tensor":
        raise ValueError("Unknown metadata field type")
    import torch

    dtype = np.dtype(field["dtype"])
    data = bytes.fromhex(field["bytes_hex"])
    shape = tuple(field["shape"])
    array = np.frombuffer(data, dtype=dtype).copy().reshape(shape)
    return torch.from_numpy(array)


def prepare_ranked(wire_path, order_path, output):
    import torch

    wire_path = Path(wire_path).expanduser().resolve()
    order_path = Path(order_path).expanduser().resolve()
    state = torch.load(wire_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Expected a PQAT INT4 state dict")
    for key in ARRAY_KEYS:
        if key not in state:
            raise ValueError(f"Required PQAT field missing: {key}")
    packed = state[PACKED_KEY].detach().cpu().contiguous()
    scales = state[SCALES_KEY].detach().cpu().contiguous()
    indices = state[INDICES_KEY].detach().cpu().contiguous()
    if packed.dtype != torch.uint8 or scales.dtype != torch.float32 or indices.dtype != torch.int32:
        raise ValueError("Expected uint8 packed values, float32 scales, int32 indices")
    if packed.ndim != 1 or scales.ndim != 1 or indices.ndim != 1:
        raise ValueError("Expected flat PQAT arrays")
    n = indices.numel()
    block_size = int(state["model.params_sparse_int4_block_size"].item())
    if block_size != 64 or packed.numel() != n * 32 or scales.numel() != n:
        raise ValueError("Unexpected packing: expected 64 INT4 values = 32 bytes per retained block")
    if int(state["model.params_sparse_int4_keep_blocks"].item()) != n:
        raise ValueError("keep_blocks differs from indices length")
    if int(state["model.params_sparse_int4_num_values"].item()) != n * block_size:
        raise ValueError("num_values differs from retained blocks x block_size")

    with np.load(order_path, allow_pickle=False) as order:
        ranked_ids = [int(v) for v in order["ordered_block_ids"]]
        priority_scores = np.asarray(order["priority_scores"], dtype=np.float32)
        source_path = Path(str(order["source_wire"])).resolve()
        frame = int(order["ntc_index"])
    if source_path != wire_path:
        raise ValueError(f"Priority source_wire mismatch:\n  order: {source_path}\n  actual: {wire_path}")
    if len(ranked_ids) != n or len(priority_scores) != n:
        raise ValueError(f"Priority length {len(ranked_ids)} / score length {len(priority_scores)} != {n} retained blocks")
    stored_ids = indices.numpy().tolist()
    if len(set(stored_ids)) != n or len(set(ranked_ids)) != n:
        raise ValueError("Duplicate ID in retained blocks or priority ranking")
    if set(stored_ids) != set(ranked_ids):
        missing = sorted(set(stored_ids) - set(ranked_ids))[:12]
        extra = sorted(set(ranked_ids) - set(stored_ids))[:12]
        raise ValueError(f"Priority IDs do not match PQAT indices. Missing {missing}; extra {extra}")

    id_to_position = {int(value): pos for pos, value in enumerate(stored_ids)}
    qbytes, sbytes, ibytes = tensor_bytes(packed), tensor_bytes(scales), tensor_bytes(indices)
    originally_ascending = stored_ids == sorted(stored_ids)
    metadata = {
        "codec": "pqat_int4_blocks_v1",
        "block_size": block_size,
        "ntc_index": frame,
        "keys_in_order": list(state),
        "other_fields": {k: encode_other_field(v) for k, v in state.items() if k not in ARRAY_KEYS},
        "original_ids": "ascending" if originally_ascending else stored_ids,
        "source_state_sha256": state_hash(state),
        "source_wire_sha256": sha(wire_path.read_bytes()),
    }
    meta_bytes = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
    output = Path(output)
    (output / "send").mkdir()
    sizes = []
    score_descending = bool(np.all(np.diff(priority_scores) <= 0))
    groups = {}
    base, leftover = divmod(n, 4)
    cursor = 0
    for priority in range(4):
        count = base + (1 if priority < leftover else 0)
        picked = ranked_ids[cursor:cursor + count]
        cursor += count
        records = bytearray()
        for block_id in picked:
            pos = id_to_position[block_id]
            # Preserve the bit-level representation of ID, scale and qINT4.
            records.extend(ibytes[pos * 4:(pos + 1) * 4])
            records.extend(sbytes[pos * 4:(pos + 1) * 4])
            records.extend(qbytes[pos * 32:(pos + 1) * 32])
        group_meta = meta_bytes if priority == 0 else b""
        group_bytes = HEADER.pack(MAGIC, priority, count, len(group_meta)) + group_meta + records
        (output / "send" / f"P{priority}.bin").write_bytes(group_bytes)
        sizes.append(len(group_bytes))
        groups[f"P{priority}"] = {
            "block_count": count,
            "payload_bytes": len(group_bytes),
            "sha256": sha(group_bytes),
            "first_rank": cursor - count,
            "last_rank": cursor - 1,
        }
    manifest = {
        "frame": frame, "source_wire": str(wire_path), "priority_order": str(order_path),
        "priority_method": "original_ordered_block_ids_equal_count_groups",
        "score_sequence_nonincreasing": score_descending,
        "retained_blocks": n, "block_size": block_size, "record_bytes": RECORD.size,
        "all_block_record_bytes": RECORD.size * n,
        "payload_bytes_total": sum(sizes),
        "source_state_sha256": metadata["source_state_sha256"],
        "groups": groups,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"NTC {frame} | {n} retained blocks x {RECORD.size} B = {RECORD.size * n} B")
    print(f"Priority IDs exactly match the retained block IDs: YES")
    print(f"Priority scores non-increasing in ordered list: {score_descending}")
    for priority in range(4):
        entry = groups[f"P{priority}"]
        print(f"  P{priority}: {entry['block_count']} saliency-ranked blocks, {entry['payload_bytes']} bytes")
    return manifest


def parse_received(directory, manifest):
    expected = manifest["groups"]
    parsed = {}
    meta = None
    for p in range(4):
        group = (Path(directory) / f"P{p}.bin").read_bytes()
        if sha(group) != expected[f"P{p}"]["sha256"]:
            raise ValueError(f"P{p} SHA256 mismatch")
        if len(group) < HEADER.size:
            raise ValueError(f"P{p} missing header")
        magic, packet_priority, record_count, meta_len = HEADER.unpack_from(group)
        if magic != MAGIC or packet_priority != p:
            raise ValueError(f"P{p} header invalid")
        if record_count != expected[f"P{p}"]["block_count"]:
            raise ValueError(f"P{p} record count unexpected")
        if len(group) != HEADER.size + meta_len + record_count * RECORD.size:
            raise ValueError(f"P{p} payload length inconsistent")
        if p == 0:
            if not meta_len:
                raise ValueError("P0 must carry reconstruction metadata")
            meta = json.loads(group[HEADER.size:HEADER.size + meta_len])
        elif meta_len:
            raise ValueError(f"P{p} unexpectedly repeated metadata")
        records = group[HEADER.size + meta_len:]
        for idx in range(record_count):
            start = idx * RECORD.size
            raw = records[start:start + RECORD.size]
            block_id = struct.unpack_from("<i", raw)[0]
            if block_id in parsed:
                raise ValueError(f"Duplicate received block ID {block_id}")
            parsed[block_id] = (raw[4:8], raw[8:40], p)
    if meta is None or len(parsed) != manifest["retained_blocks"]:
        raise ValueError("Missing metadata or retained blocks")
    return meta, parsed


def reconstruct_ranked(received_dir, manifest, output_path):
    import torch

    meta, records = parse_received(received_dir, manifest)
    if meta["codec"] != "pqat_int4_blocks_v1" or meta["block_size"] != 64:
        raise ValueError("Unexpected block codec")
    if meta["original_ids"] == "ascending":
        original_ids = sorted(records)
    else:
        original_ids = meta["original_ids"]
        if len(set(original_ids)) != len(records) or set(original_ids) != set(records):
            raise ValueError("Original retained ID order not recoverable")
    all_id_bytes = bytearray()
    all_scale_bytes = bytearray()
    all_qbytes = bytearray()
    for block_id in original_ids:
        scale, packed, _ = records[block_id]
        all_id_bytes.extend(struct.pack("<i", block_id))
        all_scale_bytes.extend(scale)
        all_qbytes.extend(packed)
    arrays = {
        INDICES_KEY: torch.from_numpy(np.frombuffer(all_id_bytes, dtype="<i4").copy()),
        SCALES_KEY: torch.from_numpy(np.frombuffer(all_scale_bytes, dtype="<f4").copy()),
        PACKED_KEY: torch.from_numpy(np.frombuffer(all_qbytes, dtype="u1").copy()),
    }
    state = OrderedDict()
    for key in meta["keys_in_order"]:
        state[key] = arrays[key] if key in ARRAY_KEYS else decode_other_field(meta["other_fields"][key])
    actual_digest = state_hash(state)
    if actual_digest != meta["source_state_sha256"] or actual_digest != manifest["source_state_sha256"]:
        raise ValueError("Reconstructed tensor-level SHA256 mismatch")
    torch.save(state, output_path)
    return len(records), sum((Path(received_dir) / f"P{i}.bin").stat().st_size for i in range(4))
