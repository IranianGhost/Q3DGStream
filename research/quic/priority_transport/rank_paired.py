"""Rank the paired keep20 blocks using the existing per-update saliency tiles."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from dynamic_priority_core import rank_retained_blocks
from paired_affine import decode_to_dense_state


BLOCK_SIZE = 64
HASH_BLOCKS = 30976
TOTAL_BLOCKS = 31120
PAIRED = Path.home() / "gs4-flame-steak/pqat-paired-affine-keep20"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_tiles(xyz, count):
    tiles = [np.arange(len(xyz), dtype=np.int64)]
    while len(tiles) < count:
        _, index = max((len(ids), index) for index, ids in enumerate(tiles) if len(ids) > 1)
        ids = tiles.pop(index)
        axis = int(np.argmax(np.ptp(xyz[ids], axis=0)))
        ordered = ids[np.argsort(xyz[ids, axis], kind="stable")]
        half = len(ordered) // 2
        tiles.extend((ordered[:half], ordered[half:]))
    tiles.sort(key=lambda ids: tuple(np.mean(xyz[ids], axis=0).tolist()))
    return tiles


def load_tile_data(path, frame):
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["ntc_index"]) != frame:
            raise ValueError(f"Priority order update mismatch: {path}")
        tiers = np.array(saved["tile_tiers"], dtype=np.uint8)
        saliency = np.array(saved["tile_saliency"], dtype=np.float64)
        source = str(saved["source_wire"])
    if (tiers.shape != (32,) or saliency.shape != (32,) or (tiers > 3).any()
            or not np.isfinite(saliency).all() or (saliency < 0).any() or saliency.max() <= 0):
        raise ValueError(f"Invalid per-tile saliency or priority: {path}")
    return tiers, saliency, source


def rank_frame(model, torch, xyz, tiles, a, b, frame, old_order, args):
    if a.get("format") != "qntc_sparse_block_affine_v1" or a.get("bits") != 2:
        raise ValueError(f"Update {frame}: unexpected INT2 wire format")
    if b.get("format") != "qntc_sparse_block_affine_v1" or b.get("bits") != 4:
        raise ValueError(f"Update {frame}: unexpected INT4 wire format")
    if not torch.equal(a["indices"], b["indices"]) or tuple(a["shape"]) != tuple(b["shape"]):
        raise ValueError(f"Update {frame}: paired masks or shapes differ")
    if any(not torch.equal(a[key], b[key]) for key in ("xyz_bound_min", "xyz_bound_max")):
        raise ValueError(f"Update {frame}: paired bounds differ")
    ids = b["indices"].detach().cpu().numpy().astype(np.int64)
    if (b["block_size"] != BLOCK_SIZE or len(ids) == 0 or np.any(np.diff(ids) <= 0)
            or ids[0] < 0 or ids[-1] >= TOTAL_BLOCKS):
        raise ValueError(f"Update {frame}: invalid retained block IDs")
    decoded = decode_to_dense_state(b, expected_bits=4)["model.params"]
    if decoded.numel() != TOTAL_BLOCKS * BLOCK_SIZE:
        raise ValueError("Unexpected NTC architecture")
    with torch.no_grad():
        model.params.copy_(decoded.to(device="cuda", dtype=model.params.dtype))
    lo = b["xyz_bound_min"].to(device="cuda", dtype=torch.float32).reshape(1, 3)
    hi = b["xyz_bound_max"].to(device="cuda", dtype=torch.float32).reshape(1, 3)
    if not bool(torch.isfinite(lo).all() and torch.isfinite(hi).all() and (hi > lo).all()):
        raise ValueError(f"Update {frame}: invalid normalization bounds")
    tiers, saliency, source = load_tile_data(old_order, frame)
    hash_ids = ids[ids < HASH_BLOCKS]
    decoder_ids = ids[ids >= HASH_BLOCKS]
    sensitivity = np.zeros((len(tiles), len(hash_ids)), dtype=np.float64)
    selected_gpu = torch.as_tensor(hash_ids, dtype=torch.long, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(args.seed + frame * 100003)
    for tile_index, members in enumerate(tiles):
        sample = np.linspace(0, len(members) - 1, min(len(members), args.samples_per_tile), dtype=np.int64)
        pos = torch.as_tensor(xyz[members[sample]], device="cuda", dtype=torch.float32)
        x = (pos - lo) / (hi - lo)
        x = x[((x >= 0) & (x <= 1)).all(dim=1)].contiguous()
        if len(x) == 0:
            continue
        for _ in range(args.probes):
            model.zero_grad(set_to_none=True)
            y = model(x)[:, :7].float()
            signs = torch.randint(0, 2, y.shape, device="cuda", dtype=torch.int8, generator=generator)
            signs = signs.float().mul_(2).sub_(1)
            loss = (y * signs).sum() / math.sqrt(float(y.numel()))
            loss.backward()
            if model.params.grad is None:
                raise RuntimeError("tiny-cuda-nn produced no parameter gradients")
            gradient = model.params.grad.detach().float().reshape(-1, BLOCK_SIZE)
            weights = model.params.detach().float().reshape(-1, BLOCK_SIZE)
            influence = (gradient[selected_gpu] * weights[selected_gpu]).square().sum(dim=1)
            sensitivity[tile_index] += influence.cpu().numpy() / args.probes
    model.zero_grad(set_to_none=True)
    ordered_hash, scores = rank_retained_blocks(hash_ids, sensitivity, tiers, saliency)
    if not np.isfinite(scores).all() or float(scores.sum()) <= 0:
        raise ValueError(f"Update {frame}: saliency sensitivity is zero or nonfinite")
    ordered = np.concatenate((decoder_ids, ordered_hash)).astype(np.int32)
    decoder_score = float(scores.max()) + max(1.0, abs(float(scores.max())))
    scores = np.concatenate((np.full(len(decoder_ids), decoder_score), scores)).astype(np.float32)
    if len(ordered) != len(ids) or not np.array_equal(np.sort(ordered), ids):
        raise ValueError(f"Update {frame}: ranked IDs do not cover the retained blocks")
    return ordered, scores, tiers, saliency, source


def run(args):
    import torch
    import tinycudann as tcnn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for per-NTC gradient sensitivity")
    sys.path.insert(0, str(args.viewer.resolve()))
    import util_gau

    xyz = np.asarray(util_gau.load_ply(str(args.scene / "init_3dgs.ply")).xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Scene has invalid Gaussian positions")
    tiles = balanced_tiles(xyz, 32)
    # The original ordering files supply the original tile weights. Validate
    # that the old and current scene have identical persistent point order.
    _, _, source = load_tile_data(args.old_orders / "NTC_000000.npz", 0)
    old_scene = Path(source).parent.parent / "scenes/candidate/init_3dgs.ply"
    if old_scene.resolve() != (args.scene / "init_3dgs.ply").resolve():
        if not old_scene.is_file():
            raise FileNotFoundError(f"Original saliency scene is required to verify point order: {old_scene}")
        original_xyz = np.asarray(util_gau.load_ply(str(old_scene)).xyz, dtype=np.float32)
        if not np.array_equal(xyz, original_xyz):
            raise ValueError("Current scene points differ from the scene used for the saved saliency orders")
    config = json.loads((args.scene / "NTCs/config.json").read_text())
    model = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=8,
        encoding_config=config["encoding"], network_config=config["network"]).to("cuda")
    model.train()
    if model.params.numel() != TOTAL_BLOCKS * BLOCK_SIZE:
        raise ValueError("The initialized model does not match the saved NTCs")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "method": "Original per-tile saliency-weighted gradient-times-weight sensitivity",
        "shared_order_source": "Each paired INT4 update, reused for matched INT2 and INT4",
        "scene": str(args.scene), "old_orders": str(args.old_orders),
        "coordinates": "Initial persistent Gaussian positions, not time-updated positions",
        "samples_per_tile": args.samples_per_tile, "probes": args.probes, "seed": args.seed,
        "tiles": 32, "block_size": BLOCK_SIZE,
    }
    metadata_path = args.out / "ranking_method.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError(f"Ranking settings differ from {metadata_path}; use a new output directory")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for frame in range(args.frames):
        output = args.out / f"NTC_{frame:06}.npz"
        int2 = args.int2_dir / f"NTC_{frame:06}.pth"
        int4 = args.int4_dir / f"NTC_{frame:06}.pth"
        old = args.old_orders / f"NTC_{frame:06}.npz"
        hash2, hash4, hash_old = (sha256(path) for path in (int2, int4, old))
        if output.exists():
            with np.load(output, allow_pickle=False) as saved:
                if (int(saved["ntc_index"]) != frame or str(saved["int2_sha256"]) != hash2
                    or str(saved["int4_sha256"]) != hash4 or str(saved["saliency_sha256"]) != hash_old):
                    raise ValueError(f"Stale existing order at {output}; use a new output directory")
            continue
        t0 = time.monotonic()
        a = torch.load(int2, map_location="cpu", weights_only=True)
        b = torch.load(int4, map_location="cpu", weights_only=True)
        ids, scores, tiers, saliency, _ = rank_frame(model, torch, xyz, tiles, a, b, frame, old, args)
        pending = output.with_suffix(".pending")
        with pending.open("wb") as stream:
            np.savez_compressed(stream, ordered_block_ids=ids, priority_scores=scores,
                tile_tiers=tiers, tile_saliency=saliency.astype(np.float32), ntc_index=np.int32(frame),
                source_wire=str(int4), int2_sha256=hash2, int4_sha256=hash4, saliency_sha256=hash_old,
                probe_elapsed_seconds=np.float32(time.monotonic() - t0))
        os.replace(pending, output)
        if frame == 0 or (frame + 1) % 10 == 0 or frame + 1 == args.frames:
            print(f"RANKED {frame + 1}/{args.frames} NTCs, {len(ids):,} blocks, "
                  f"latest {time.monotonic() - t0:.1f} s", flush=True)
    print(f"Priority orders {args.out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--int2-dir", type=Path, default=PAIRED / "int2_keep20/wire")
    p.add_argument("--int4-dir", type=Path, default=PAIRED / "int4_keep20/wire")
    p.add_argument("--old-orders", type=Path, default=Path.home() / "snap/firefox/common/QNTC_dynamic_299_priorities_v1/orders")
    p.add_argument("--scene", type=Path, default=Path.home() / "qntc_scenes/flame_steak_official")
    p.add_argument("--viewer", type=Path, default=Path.home() / "QNTC_Linux_Package/3DGStreamViewer")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frames", type=int, default=299)
    p.add_argument("--samples-per-tile", type=int, default=128)
    p.add_argument("--probes", type=int, default=2)
    p.add_argument("--seed", type=int, default=114)
    a = p.parse_args()
    if a.frames < 1 or a.frames > 299 or a.samples_per_tile < 16 or not 1 <= a.probes <= 16:
        p.error("Invalid frame, sample or probe count")
    run(a)
