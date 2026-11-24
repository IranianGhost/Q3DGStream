"""Prepare exact priority groups for paired INT2 and INT4 checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paired_groups import prepare_frame


def prepare(args):
    for path in (args.int2_dir, args.int4_dir, args.orders, args.bootstrap_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    bootstrap = torch.load(args.bootstrap_dir / "NTC_000000.pth", map_location="cpu", weights_only=True)
    bootstrap_bounds = {key: bootstrap[key].detach().cpu() for key in ("xyz_bound_min", "xyz_bound_max")}
    args.out.mkdir(parents=True, exist_ok=False)
    send = args.out / "send"
    send.mkdir()
    manifests = {}
    total = {"2": 0, "4": 0}
    for frame in range(args.frames):
        name = f"NTC_{frame:06}"
        paths = [args.int2_dir / f"{name}.pth", args.int4_dir / f"{name}.pth"]
        # The reused single viewer network retains the initial NTC bounds.
        for path in paths:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if any(not torch.equal(state[key].detach().cpu(), bootstrap_bounds[key]) for key in bootstrap_bounds):
                raise ValueError(f"{path} has different normalization bounds from the viewer model")
        manifest = prepare_frame(*paths, args.orders / f"{name}.npz", send / name, frame,
                                 block_index_order=args.block_index_order)
        manifest["source_wire"] = str(args.bootstrap_dir / f"{name}.pth")
        manifests[str(frame)] = manifest
        for bits in total:
            total[bits] += manifest["variants"][bits]["payload_bytes_total"]
        if (frame + 1) % 25 == 0 or frame + 1 == args.frames:
            print(f"PREPARED AND VERIFIED {frame + 1}/{args.frames} INT2/INT4 pairs", flush=True)
    data = {"frames": args.frames, "start_frame": 0, "payload_bytes_by_precision": total,
            "payload_bytes_total": total["4"], "source_format": "paired block-affine INT2/INT4",
            "priority_method": ("ascending retained block index" if args.block_index_order else
                                "shared saliency order calculated on paired INT4"),
            "manifests": manifests}
    pending = args.out / "sequence_manifest.pending"
    pending.write_text(json.dumps(data, indent=2) + "\n")
    pending.replace(args.out / "sequence_manifest.json")
    print(f"INT2 {total['2']:,} B, INT4 {total['4']:,} B for {args.frames} updates", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int2-dir", type=Path, required=True)
    parser.add_argument("--int4-dir", type=Path, required=True)
    parser.add_argument("--orders", type=Path, required=True)
    parser.add_argument("--bootstrap-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=299)
    parser.add_argument("--block-index-order", action="store_true")
    options = parser.parse_args()
    if options.frames < 1 or options.frames > 299:
        parser.error("--frames must be between 1 and 299")
    prepare(options)
