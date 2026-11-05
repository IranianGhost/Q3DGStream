"""Send real original NTC checkpoints directly to Publio's priority-enabled player."""

import argparse
import asyncio
import json
from pathlib import Path
import tempfile
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from .transport import ALPN, FileJob, send_files, send_whole_files


def collect_scene(scene, updates):
    scene = Path(scene)
    if updates < 1:
        raise ValueError("updates must be positive")
    init = scene / "init_3dgs.ply"
    config = scene / "NTCs/config.json"
    config_data = json.loads(config.read_text())
    if "encoding" not in config_data or "network" not in config_data:
        raise ValueError("invalid model config")
    additions = (scene / "additional_3dgs").is_dir()
    frames = []
    for frame in range(updates):
        ntc = scene / f"NTCs/NTC_{frame:06d}.pth"
        add = (
            scene / f"additional_3dgs/additions_{frame:06d}.ply" if additions else None
        )
        frames.append((frame, ntc, add))
    for path in [init, config] + [
        p for _, ntc, add in frames for p in [ntc, add] if p is not None
    ]:
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(path)
    return init, config, frames, additions


async def send_scene(connection, scene, updates=3, mode="chunked", max_inflight=16):
    init, config, frames, additions = collect_scene(scene, updates)
    all_rows = []

    async def transfer(jobs, phase):
        if mode == "whole":
            rows = await send_whole_files(connection, jobs)
        else:
            rows = await send_files(
                connection,
                jobs,
                policy="strict",
                max_inflight=max_inflight,
                timeout=120,
            )
        all_rows.extend(dict(r, phase=phase) for r in rows)

    with tempfile.TemporaryDirectory() as temporary:
        manifest = Path(temporary) / "priority_manifest.json"
        manifest.write_text(
            json.dumps(
                dict(
                    require_additions=additions,
                    updates=updates,
                    checkpoint_format="original",
                    complete_updates_only=True,
                )
            )
        )
        await transfer(
            [
                FileJob(manifest, "priority_manifest.json", 0),
                FileJob(init, "init_3dgs.ply", 0),
                FileJob(config, "NTCs/config.json", 0),
            ],
            "metadata",
        )
        for frame, ntc, addition in frames:
            if addition is not None:
                await transfer(
                    [
                        FileJob(
                            addition, f"additional_3dgs/additions_{frame:06d}.ply", 0
                        )
                    ],
                    f"addition-{frame}",
                )
            await transfer(
                [FileJob(ntc, f"NTCs/NTC_{frame:06d}.pth", 0)], f"ntc-{frame}"
            )
            print(f"[ORIGINAL NTC] update {frame} committed", flush=True)
    return all_rows


async def run(args):
    conf = QuicConfiguration(
        is_client=True, alpn_protocols=ALPN, server_name="localhost"
    )
    conf.load_verify_locations(args.cert)
    async with connect("127.0.0.1", args.port, configuration=conf) as connection:
        rows = await send_scene(
            connection, args.scene, args.updates, args.mode, args.max_inflight
        )
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--cert", required=True)
    p.add_argument("--port", type=int, default=5003)
    p.add_argument("--updates", type=int, default=3)
    p.add_argument("--mode", choices=["whole", "chunked"], default="chunked")
    p.add_argument("--max-inflight", type=int, default=16)
    p.add_argument("--log", type=Path)
    asyncio.run(run(p.parse_args()))
