"""Original trained NTC transport comparison, with no saliency proxy or recompression."""

import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import time
from aioquic.asyncio import connect, serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.logger import QuicFileLogger
from .transport import ALPN, Receiver, FileJob, file_hash, send_files, send_whole_files
from .player_bridge import PlayerCacheBridge


async def run(args):
    import torch

    source = args.scene / "NTCs/NTC_000000.pth"
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    assert set(checkpoint) == {"model.params", "xyz_bound_min", "xyz_bound_max"}
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be new/empty")
    out.mkdir(parents=True, exist_ok=True)
    (out / "qlog").mkdir()
    with tempfile.TemporaryDirectory() as temporary:
        cache = Path(temporary)
        notifications = []
        bridge = PlayerCacheBridge(cache, notifications.append)
        rx = Receiver(cache, on_file=bridge.committed)
        conf = QuicConfiguration(is_client=False, alpn_protocols=ALPN)
        conf.load_cert_chain(args.cert, args.key)
        server = await serve(
            "127.0.0.1", 0, configuration=conf, stream_handler=rx.handler
        )
        client = QuicConfiguration(
            is_client=True,
            alpn_protocols=ALPN,
            server_name="localhost",
            quic_logger=QuicFileLogger(str(out / "qlog")),
        )
        client.load_verify_locations(args.cert)
        try:
            async with connect(
                "127.0.0.1",
                server._transport.get_extra_info("sockname")[1],
                configuration=client,
            ) as connection:
                job = FileJob(source, "NTCs/NTC_000000.pth", 0)
                start = time.monotonic()
                if args.mode == "whole":
                    rows = await send_whole_files(connection, [job], timeout=180)
                else:
                    rows = await send_files(
                        connection, [job], policy="strict", max_inflight=16, timeout=180
                    )
                ack_ms = (time.monotonic() - start) * 1000
                commit_ms = max(e["received"] for e in rx.events) - start
            received = cache / job.remote
            assert file_hash(source) == file_hash(received)
            restored = torch.load(received, map_location="cpu", weights_only=True)
            assert all(torch.equal(checkpoint[k], restored[k]) for k in checkpoint)
            assert notifications == [job.remote] and not rx.errors
            report = dict(
                mode=args.mode,
                profile=args.profile,
                repeat=args.repeat,
                bytes=source.stat().st_size,
                sha256=file_hash(source),
                parameters=checkpoint["model.params"].numel(),
                checkpoint_commit_ms=commit_ms * 1000,
                ack_ms=ack_ms,
                streams=len(rows),
                valid=True,
                scope="One original trained FP32 NTC, steady-state transfer after handshake, no startup scene or rendering",
            )
            (out / "summary.json").write_text(json.dumps(report, indent=2))
            (out / "events.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(report), flush=True)
        except Exception as exc:
            failure = dict(
                mode=args.mode,
                profile=args.profile,
                repeat=args.repeat,
                valid=False,
                error=repr(exc),
                receiver_errors=rx.errors,
                committed_files=sum(e["complete"] for e in rx.events),
                received_chunks=len(rx.events),
            )
            (out / "failure.json").write_text(json.dumps(failure, indent=2))
            raise
        finally:
            server.close()
            await rx.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cert", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--mode", choices=["whole", "chunked"], required=True)
    p.add_argument("--profile", default="native")
    p.add_argument("--repeat", type=int, default=1)
    asyncio.run(asyncio.wait_for(run(p.parse_args()), 240))
