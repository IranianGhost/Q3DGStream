#!/usr/bin/env python3
"""Standalone dynamic-299 QUIC sender. Uses prevalidated .qblk stages and real Gaussian additions.

Does not modify the existing viewer, renderer, sender, QUIC receiver, or scene data.
Only for local 127.0.0.1 demonstration; application-ordered file delivery is not
QUIC transport-level stream prioritization.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

# Shared locations keep experiments independent of the author's machine.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import q3dg_paths as paths
import ssl
import struct
import tempfile
import time

BASE = paths.EXPERIMENT_DIR
ORIGINAL = paths.SCENE_DIR
STAGES = paths.STAGES_DIR
CACHE = paths.CACHE_DIR / 'qntc_dynamic_priority_cache'
HEADER = struct.Struct('!5sIBHH')
RECORD = struct.Struct('!If32s')
MAGIC = b'QNP01'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.digest()


def validate_stage(path, frame, stage):
    b = path.read_bytes()
    if len(b) < HEADER.size:
        raise ValueError(f'short stage: {path}')
    magic, idx, st, expected, count = HEADER.unpack_from(b)
    if (magic, idx, st, expected) != (MAGIC, frame, stage, 1556):
        raise ValueError(f'bad stage header: {path}')
    if not 0 <= count <= 1556 or len(b) != HEADER.size + count*RECORD.size:
        raise ValueError(f'bad stage size: {path}')
    ids = set()
    for pos in range(count):
        block_id, scale, _ = RECORD.unpack_from(b, HEADER.size + pos*RECORD.size)
        if block_id >= 31120 or block_id in ids or not (scale >= 0 and scale < float('inf')):
            raise ValueError(f'bad block ID / scale / duplicate: {path}')
        ids.add(block_id)
    return ids


def collect(args):
    # Validate every packed stage before sending any data.
    scene = args.compressed_scene.expanduser().resolve()
    orig = args.original_scene.expanduser().resolve()
    stages = args.stages.expanduser().resolve()
    init = scene / 'init_3dgs.ply'
    orig_init = orig / 'init_3dgs.ply'
    config = scene / 'NTCs/config.json'
    first = args.wire.expanduser().resolve() / f'NTC_{args.start:06d}.pth'
    for p in (init, orig_init, config, first):
        if not p.is_file():
            raise FileNotFoundError(p)
    if sha256(init) != sha256(orig_init):
        raise RuntimeError('compressed and original initial Gaussians are NOT byte identical; refusing to mix additions')
    cfg = json.loads(config.read_text())
    if 'encoding' not in cfg or 'network' not in cfg:
        raise RuntimeError('invalid NTC configuration')
    additions = []
    frames = []
    nbytes = init.stat().st_size + config.stat().st_size
    for frame in range(args.start, args.end + 1):
        add = orig / 'additional_3dgs' / f'additions_{frame:06d}.ply'
        if not add.is_file():
            raise FileNotFoundError(add)
        additions.append(add)
        nbytes += add.stat().st_size
        parts = []
        all_ids = set()
        for part in range(4):
            p = stages / f'NTC_{frame:06d}_P{part}.qblk'
            if not p.is_file():
                raise FileNotFoundError(p)
            ids = validate_stage(p, frame, part)
            if all_ids.intersection(ids):
                raise ValueError(f'frame {frame}: duplicate block IDs across stages')
            all_ids.update(ids)
            nbytes += p.stat().st_size
            parts.append(p)
        if len(all_ids) != 1556:
            raise ValueError(f'frame {frame}: got {len(all_ids)} blocks; expected 1556')
        frames.append((frame, add, parts))
    return init, config, first, frames, nbytes


def storage_check(args, nbytes):
    cache = args.cache.expanduser()
    if cache.name != 'qntc_dynamic_priority_cache' or cache.is_symlink():
        raise RuntimeError('cache must be a real dedicated path named qntc_dynamic_priority_cache')
    root = cache.parent.resolve()
    if not root.is_dir() or cache.exists() and not cache.is_dir():
        raise RuntimeError(f'cache parent/directory unavailable: {cache}')
    free = os.statvfs(root)
    available = free.f_bavail * free.f_frsize
    required = int(nbytes * 1.15 + 128 * 1024 * 1024)
    print(f'[PREFLIGHT] Frames={args.start}..{args.end}, additions={args.end-args.start+1}, stage files={(args.end-args.start+1)*4}', flush=True)
    print(f'[PREFLIGHT] bytes={nbytes/2**20:.2f} MiB, free={available/2**20:.1f} MiB, required={required/2**20:.1f} MiB', flush=True)
    if required > available:
        raise RuntimeError('insufficient free space for dedicated progressive QUIC cache')


async def send_priority_scene(proto, init, config, boundfile, frames, args):
    """Opt-in chunk scheduler. Keep metadata/additions/frame dependencies intact."""
    from research.quic.priority_transport.transport import FileJob, send_files
    options = dict(policy=args.policy, max_inflight=args.max_inflight, timeout=getattr(args,"chunk_timeout",120))
    rows = []
    async def batch(jobs, phase):
        result = await send_files(proto, jobs, **options)
        for row in result:
            row['phase'] = phase
        rows.extend(result)
    await batch([FileJob(boundfile, 'NTCs/progressive/bounds.json', 0),
                 FileJob(init, 'init_3dgs.ply', 0),
                 FileJob(config, 'NTCs/config.json', 0)], 'metadata')
    if args.warmup_delay:
        await asyncio.sleep(args.warmup_delay)
    for idx, addition, stages in frames:
        await batch([FileJob(addition, f'additional_3dgs/additions_{idx:06d}.ply', 0)], f'addition-{idx}')
        await batch([FileJob(path, f'NTCs/progressive/NTC_{idx:06d}_P{st}.qblk', st)
                     for st, path in enumerate(stages)], f'ntc-{idx}')
        print(f'[PRIORITY CHUNKS] NTC={idx:03d}: all files committed', flush=True)
    return rows


async def transfer(args, scene_data):
    from aioquic.asyncio import connect
    from aioquic.quic.configuration import QuicConfiguration
    if args.transport == 'priority':
        from research.quic.priority_transport.transport import ALPN
    else:
        from quic_transport import ALPN, send_file
    import torch

    init, config, first, frames, _ = scene_data
    state = torch.load(first, map_location='cpu', weights_only=True)
    bounds = {
        'xyz_bound_min': state['xyz_bound_min'].cpu().float().tolist(),
        'xyz_bound_max': state['xyz_bound_max'].cpu().float().tolist(),
    }
    conf = QuicConfiguration(is_client=True, alpn_protocols=ALPN)
    if args.transport == 'priority':
        conf.congestion_control_algorithm=args.cc
        conf.max_datagram_size=args.datagram_size
        conf.load_verify_locations(str(args.cert))
    else:
        conf.verify_mode = ssl.CERT_NONE    # existing legacy loopback behavior
    conf.server_name = 'localhost'

    with tempfile.TemporaryDirectory(prefix='qntc_dynamic_priority_bounds_') as tmp:
        boundfile = Path(tmp) / 'bounds.json'
        boundfile.write_text(json.dumps(bounds))
        async with connect('127.0.0.1', args.port, configuration=conf) as proto:
            if args.transport == 'priority':
                rows = await send_priority_scene(proto, init, config, boundfile, frames, args)
                if args.transport_log:
                    args.transport_log.parent.mkdir(parents=True, exist_ok=True)
                    args.transport_log.write_text(json.dumps(rows, indent=2))
                return
            # Initial metadata and scene, then true original additions before each NTC update.
            await send_file(proto, 'NTCs/progressive/bounds.json', boundfile, timeout=120)
            await send_file(proto, 'init_3dgs.ply', init, timeout=120)
            await send_file(proto, 'NTCs/config.json', config, timeout=120)
            if args.warmup_delay:
                await asyncio.sleep(args.warmup_delay)
            for idx, add, stages in frames:
                await send_file(proto, f'additional_3dgs/additions_{idx:06d}.ply', add, timeout=120)
                for st, path in enumerate(stages):
                    await send_file(proto, f'NTCs/progressive/NTC_{idx:06d}_P{st}.qblk', path, timeout=120)
                    if args.stage_delay:
                        await asyncio.sleep(args.stage_delay)
                if idx == frames[0][0] or idx == frames[-1][0] or idx % 10 == 0:
                    print(f'[QUIC PRIORITY] NTC={idx:03d}, original addition + P0/P1/P2/P3 committed', flush=True)
    print(f'[QUIC COMPLETE] {len(frames)} full staged INT4 updates and matching original additions delivered', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compressed-scene', type=Path, default=BASE/'scenes/candidate')
    p.add_argument('--original-scene', type=Path, default=ORIGINAL)
    p.add_argument('--wire', type=Path, default=BASE/'wire')
    p.add_argument('--stages', type=Path, default=STAGES)
    p.add_argument('--cache', type=Path, default=CACHE)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=29)
    p.add_argument('--port', type=int, default=5002)
    p.add_argument('--stage-delay', type=float, default=0.0)
    p.add_argument('--warmup-delay', type=float, default=2.0)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--transport', choices=['legacy', 'priority'], default='legacy')
    p.add_argument('--policy', choices=['fifo', 'strict', 'weighted'], default='strict')
    p.add_argument('--max-inflight', type=int, default=128)
    p.add_argument('--cc',choices=['reno','cubic'],default='cubic')
    p.add_argument('--datagram-size',type=int,default=1400)
    p.add_argument('--chunk-timeout',type=float,default=120)
    p.add_argument('--cert', type=Path, help='trusted localhost certificate for priority transport')
    p.add_argument('--transport-log', type=Path)
    a = p.parse_args()
    if not 1 <= a.max_inflight <= 128:
        p.error('--max-inflight must be 1..128')
    if a.transport == 'priority' and (not a.cert or not a.cert.is_file()):
        p.error('--transport priority requires --cert PATH')
    if a.transport == 'priority' and a.stage_delay:
        p.error('--stage-delay is only available with legacy transport')
    if not (0 <= a.start <= a.end <= 298) or a.stage_delay < 0 or a.warmup_delay < 0:
        p.error('require 0 <= start <= end <= 298 and nonnegative delays')
    if a.start != 0:
        p.error('this receiver experiment must start at NTC 000')
    if not (1 <= a.port <= 65535):
        p.error('invalid port')
    data = collect(a)
    storage_check(a, data[4])
    if a.preflight_only:
        print('[READY] All source additions, stage headers and block IDs validated. Nothing transmitted.', flush=True)
        return
    asyncio.run(transfer(a, data))

if __name__ == '__main__':
    main()
