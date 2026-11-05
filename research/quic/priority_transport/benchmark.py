"""Real local QUIC tests using synthetic records in upstream QNP01 format."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

from aioquic.asyncio import connect, serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.logger import QuicFileLogger
from q3dgstream import HEADER, RECORD, parse_qblk
from .transport import ALPN, FileJob, Receiver, send_files, file_hash


def fixtures(root, frames=32, seed=1):
    rng = random.Random(seed)
    jobs = []
    for frame in range(frames):
        ids = rng.sample(range(31120), 1556)
        for priority in range(4):
            records = [
                (i, rng.uniform(0.001, 0.05), rng.randbytes(32))
                for i in ids[priority * 389 : (priority + 1) * 389]
            ]
            data = HEADER.pack(
                b"QNP01", frame, priority, 1556, len(records)
            ) + b"".join(RECORD.pack(*r) for r in records)
            remote = f"NTCs/progressive/NTC_{frame:06d}_P{priority}.qblk"
            path = root / remote
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            parse_qblk(data)
            jobs.append(FileJob(path, remote, priority, 0 if priority == 3 else 150))
    # Availability stress: P3 arrives first, urgent stages arrive later.
    return sorted(jobs, key=lambda j: (j.release_ms, j.priority, j.remote))


async def run(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    (out / "qlog").mkdir()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        jobs = fixtures(root / "source", args.frames, args.seed)
        receiver = Receiver(root / "cache")
        serverconf = QuicConfiguration(is_client=False, alpn_protocols=ALPN)
        serverconf.load_cert_chain(args.cert, args.key)
        server = await serve(
            "127.0.0.1", 0, configuration=serverconf, stream_handler=receiver.handler
        )
        port = server._transport.get_extra_info("sockname")[1]
        clientconf = QuicConfiguration(
            is_client=True,
            alpn_protocols=ALPN,
            server_name="localhost",
            quic_logger=QuicFileLogger(str(out / "qlog")),
        )
        clientconf.load_verify_locations(args.cert)
        try:
            async with connect(
                "127.0.0.1", port, configuration=clientconf
            ) as connection:
                if args.scenario == "pipeline":
                    from research.quic.dynamic_priority_adapter.send_dynamic_priority import (
                        send_priority_scene,
                    )

                    init = root / "init.ply"
                    init.write_text(
                        "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"
                    )
                    config = root / "config.json"
                    config.write_text("{}")
                    bounds = root / "bounds.json"
                    bounds.write_text(
                        '{"xyz_bound_min":[0,0,0],"xyz_bound_max":[1,1,1]}'
                    )
                    byname = {j.remote: j for j in jobs}
                    frames = []
                    for frame in range(args.frames):
                        add = root / f"addition-{frame}.ply"
                        add.write_bytes(init.read_bytes())
                        frames.append(
                            (
                                frame,
                                add,
                                [
                                    byname[
                                        f"NTCs/progressive/NTC_{frame:06d}_P{p}.qblk"
                                    ].path
                                    for p in range(4)
                                ],
                            )
                        )
                    options = SimpleNamespace(
                        policy=args.policy,
                        max_inflight=args.max_inflight,
                        warmup_delay=0,
                    )
                    rows = await send_priority_scene(
                        connection, init, config, bounds, frames, options
                    )
                    # Completed stage files of one frame precede the next frame's additions.
                    events = receiver.events
                    for frame in range(args.frames):
                        addition = f"additional_3dgs/additions_{frame:06d}.ply"
                        added = max(
                            e["received"] for e in events if e["path"] == addition
                        )
                        stages = [
                            e["received"]
                            for e in events
                            if f"NTC_{frame:06d}_" in e["path"]
                        ]
                        assert added < min(stages)
                        if frame:
                            previous = max(
                                e["received"]
                                for e in events
                                if f"NTC_{frame-1:06d}_" in e["path"]
                            )
                            assert previous < added
                else:
                    rows = await send_files(
                        connection,
                        jobs,
                        policy=args.policy,
                        max_inflight=args.max_inflight,
                    )
            # Byte identity and upstream QNP01 decoder, plus dense INT4 decode parity.
            from research.quic.progressive_299_v2.qntc_progressive_wire import (
                decode_stage,
                decode_records_into,
                TOTAL,
                BLOCK,
            )
            import numpy as np

            for job in jobs:
                actual = root / "cache" / job.remote
                assert file_hash(actual) == file_hash(job.path)
                assert parse_qblk(actual.read_bytes()) == parse_qblk(
                    job.path.read_bytes()
                )
                _, _, records = decode_stage(actual.read_bytes())
                dense = np.zeros((TOTAL, BLOCK), np.float32)
                decode_records_into(dense, records)
                _, _, reference = decode_stage(job.path.read_bytes())
                expected = np.zeros_like(dense)
                decode_records_into(expected, reference)
                np.testing.assert_array_equal(dense, expected)
            assert not receiver.errors
            summary = dict(
                policy=args.policy,
                profile=args.profile,
                scenario=args.scenario,
                seed=args.seed,
                frames=args.frames,
                files=len(jobs),
                chunks=len(rows),
                valid_files=len(jobs),
                errors=receiver.errors,
                classes={},
            )
            if args.scenario == "stress":
                for priority in range(4):
                    group = [r for r in rows if r["priority"] == priority]
                    waits = sorted(r["received_ms"] - r["release_ms"] for r in group)
                    summary["classes"][priority] = dict(
                        latency_p95_ms=waits[int(0.95 * (len(waits) - 1))],
                        deadline_500ms_misses=sum(v > 500 for v in waits),
                    )
            (out / "events.json").write_text(json.dumps(rows, indent=2))
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary), flush=True)
        finally:
            server.close()
            await receiver.close()
            (out / "receiver-events.json").write_text(
                json.dumps(receiver.events, indent=2)
            )
            (out / "receiver-errors.json").write_text(
                json.dumps(receiver.errors, indent=2)
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--cert", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--policy", choices=["fifo", "strict", "weighted"], default="strict")
    p.add_argument("--profile", default="native")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--max-inflight", type=int, default=16)
    p.add_argument("--scenario", choices=["stress", "pipeline"], default="stress")
    asyncio.run(asyncio.wait_for(run(p.parse_args()), 120))
