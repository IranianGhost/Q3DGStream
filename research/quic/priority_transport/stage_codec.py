"""Build cumulative sparse-INT4 NTC checkpoints from received Quinn blocks.

P0 -> P0+P1 -> P0+P1+P2 -> full. The first three checkpoints include ONLY
received sparse blocks; full tensor shape remains unchanged. Their numerical
rendering behavior is not assumed: verify with the actual viewer/evaluator.
"""

import argparse
from collections import OrderedDict
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import struct
import sys
import time

import numpy as np

from block_codec import (HEADER, MAGIC, RECORD, ARRAY_KEYS, PACKED_KEY,
                         SCALES_KEY, INDICES_KEY, decode_other_field, sha, state_hash)

ROOT = Path(__file__).resolve().parent


def pick_results(args):
    if args.results:
        directory = Path(args.results).expanduser().resolve()
        if not (directory / "manifest.json").is_file():
            raise FileNotFoundError(f"No manifest.json in {directory}")
        return directory
    parent = Path(args.search_dir).expanduser().resolve()
    candidates = sorted(
        (p for p in parent.glob(f"results_block{args.frame}_*")
         if (p / "manifest.json").is_file()
         and (p / args.trial / "received" / "P0.bin").is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No complete Quinn results under {parent}; use --results /exact/results_block114_..."
        )
    print(f"AUTO-SELECTED most recent matching results: {candidates[-1]}")
    return candidates[-1]


def read_group(directory, priority, expected):
    file = directory / f"P{priority}.bin"
    payload = file.read_bytes()
    if sha(payload) != expected["sha256"]:
        raise ValueError(f"Corrupted P{priority}: SHA256 mismatch")
    if len(payload) < HEADER.size:
        raise ValueError(f"P{priority}: incomplete header")
    magic, p, count, metadata_length = HEADER.unpack_from(payload)
    if magic != MAGIC or p != priority or count != expected["block_count"]:
        raise ValueError(f"P{priority}: incompatible header")
    if len(payload) != HEADER.size + metadata_length + count * RECORD.size:
        raise ValueError(f"P{priority}: wrong record length")
    if priority == 0:
        if metadata_length == 0:
            raise ValueError("Missing reconstruction metadata in P0")
        metadata = json.loads(payload[HEADER.size:HEADER.size + metadata_length])
    else:
        if metadata_length:
            raise ValueError(f"Unexpected repeated metadata in P{priority}")
        metadata = None
    raw = payload[HEADER.size + metadata_length:]
    blocks = {}
    for pos in range(count):
        record = raw[pos * RECORD.size:(pos + 1) * RECORD.size]
        block_id = struct.unpack_from("<i", record)[0]
        if block_id in blocks:
            raise ValueError(f"Duplicate block {block_id} within P{priority}")
        blocks[block_id] = (record[4:8], record[8:40])
    return metadata, blocks


def assemble_state(metadata, records, total_retained):
    """Use ONLY provided records, not source checkpoint tensors or unseen groups."""
    import torch

    if metadata["codec"] != "pqat_int4_blocks_v1" or metadata["block_size"] != 64:
        raise ValueError("Unexpected compressed NTC metadata format")
    count = len(records)
    if not 0 < count <= total_retained:
        raise ValueError("Invalid number of received retained blocks")
    if count == total_retained and metadata["original_ids"] != "ascending":
        indices = list(metadata["original_ids"])
        if len(indices) != count or set(indices) != set(records):
            raise ValueError("Original full retained-block order not recoverable")
    else:
        indices = sorted(records)

    ids = bytearray()
    scales = bytearray()
    packed = bytearray()
    for block_id in indices:
        scale_bytes, qbytes = records[block_id]
        ids.extend(struct.pack("<i", block_id))
        scales.extend(scale_bytes)
        packed.extend(qbytes)
    arrays = {
        INDICES_KEY: torch.from_numpy(np.frombuffer(ids, dtype="<i4").copy()),
        SCALES_KEY: torch.from_numpy(np.frombuffer(scales, dtype="<f4").copy()),
        PACKED_KEY: torch.from_numpy(np.frombuffer(packed, dtype="u1").copy()),
    }
    state = OrderedDict()
    for key in metadata["keys_in_order"]:
        state[key] = arrays[key] if key in ARRAY_KEYS else decode_other_field(metadata["other_fields"][key])

    if count != total_retained:
        total_blocks = int(state["model.params_sparse_int4_total_blocks"].item())
        updates = {
            "model.params_sparse_int4_keep_blocks": count,
            "model.params_sparse_int4_num_values": count * metadata["block_size"],
            "model.params_sparse_int4_keep_ratio": count / total_blocks,
        }
        for key, value in updates.items():
            old = state[key]
            state[key] = torch.tensor(value, dtype=old.dtype)
    else:
        if state_hash(state) != metadata["source_state_sha256"]:
            raise ValueError("Full stage differs from original quantized tensors")
    return state


def trial_arrival_times(results_dir, trial_name):
    match = re.fullmatch(r"run(\d+)_(baseline|priority)", trial_name)
    if not match:
        raise ValueError("--trial must look like run001_priority or run001_baseline")
    repeat, mode = int(match.group(1)), match.group(2)
    csv_file = results_dir / "metrics.csv"
    if not csv_file.is_file():
        return {}
    with csv_file.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matching = {
        int(row["group"][1:]): float(row["complete_ms"])
        for row in rows if int(row["repeat"]) == repeat and row["mode"] == mode
    }
    if matching and set(matching) != set(range(4)):
        raise ValueError("Incomplete group completion timings in metrics.csv")
    return matching


def make_stages(results_dir, trial_name, destination):
    import torch

    results_dir = Path(results_dir)
    manifest = json.loads((results_dir / "manifest.json").read_text())
    group_folder = results_dir / trial_name / "received"
    if not group_folder.is_dir():
        raise FileNotFoundError(group_folder)
    arrival = trial_arrival_times(results_dir, trial_name)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)

    total = manifest["retained_blocks"]
    if manifest.get("block_size") != 64 or manifest.get("record_bytes") != 40:
        raise ValueError("Unexpected INT4 record layout")
    accumulated = {}
    meta = None
    summary = []
    for priority in range(4):
        metadata, additions = read_group(group_folder, priority, manifest["groups"][f"P{priority}"])
        if priority == 0:
            meta = metadata
            if meta["source_state_sha256"] != manifest["source_state_sha256"]:
                raise ValueError("P0 metadata and manifest disagree about original checkpoint")
        if set(accumulated) & set(additions):
            raise ValueError(f"Overlapping received block IDs at P{priority}")
        accumulated.update(additions)
        if len(accumulated) > total:
            raise ValueError("More blocks than expected")

        labels = [f"P{k}" for k in range(priority + 1)]
        label = "_".join(labels)
        path = destination / f"stage{priority + 1}_{label}.pth"
        begin = time.perf_counter()
        state = assemble_state(meta, accumulated, total)
        torch.save(state, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if state_hash(loaded) != state_hash(state):
            raise RuntimeError(f"Stage {label} did not survive torch.save/load")
        cpu_ms = (time.perf_counter() - begin) * 1000

        count = len(accumulated)
        total_blocks = int(state["model.params_sparse_int4_total_blocks"].item())
        if int(state["model.params_sparse_int4_keep_blocks"].item()) != count:
            raise ValueError("Partial keep_blocks field is inconsistent")
        if int(state["model.params_sparse_int4_num_values"].item()) != count * 64:
            raise ValueError("Partial num_values field is inconsistent")
        measured_ms = max(arrival[k] for k in range(priority + 1)) if arrival else None
        summary.append({
            "stage": label,
            "received_groups": priority + 1,
            "received_retained_blocks": count,
            "fraction_of_retained_blocks": f"{count / total:.6f}",
            "fraction_of_all_blocks": f"{count / total_blocks:.6f}",
            "last_group_arrival_ms": f"{measured_ms:.3f}" if measured_ms is not None else "",
            "offline_stage_build_save_verify_ms": f"{cpu_ms:.3f}",
            "loaded_checkpoint_sha256": state_hash(loaded),
            "checkpoint": str(path),
        })
        arrived = f"arrival={measured_ms:.2f} ms " if measured_ms is not None else ""
        print(f"{label:15s} {count:4d}/{total} blocks {arrived}checkpoint={path.name} load=OK")

    csv_path = destination / "stage_timeline.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nCHECKPOINTS {destination}\nTIMELINE {csv_path}")
    print("Full-stage tensor SHA256 matched. Partial stages are structurally valid PyTorch states;")
    print("viewer decoding, frame PSNR/SSIM/LPIPS, and actual online reconstruction time are NOT yet verified.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=114)
    parser.add_argument("--results", help="Previously generated results_block114_... folder")
    parser.add_argument("--search-dir", default=str(ROOT.parent / "quinn_ntc_blocks"))
    parser.add_argument("--trial", default="run001_priority")
    parser.add_argument("--out", help="New output folder for four cumulative checkpoints")
    args = parser.parse_args()
    try:
        results = pick_results(args)
        found_frame = json.loads((results / "manifest.json").read_text())["frame"]
        if found_frame != args.frame:
            raise ValueError(f"Selected results contain NTC {found_frame}, not requested NTC {args.frame}")
        name = f"progressive_{args.trial}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output = Path(args.out).expanduser().resolve() if args.out else (results / name)
        make_stages(results, args.trial, output)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
