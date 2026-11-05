"""Headless local receiver. Use a new dedicated cache for every experiment."""

import argparse
import asyncio
from pathlib import Path
from aioquic.asyncio import serve
from aioquic.quic.configuration import QuicConfiguration
from .transport import ALPN, Receiver


async def run(args):
    if args.cache.exists() and any(args.cache.iterdir()):
        raise ValueError("cache must be empty")
    receiver = Receiver(args.cache)
    conf = QuicConfiguration(is_client=False, alpn_protocols=ALPN)
    conf.load_cert_chain(args.cert, args.key)
    server = await serve(
        "127.0.0.1", args.port, configuration=conf, stream_handler=receiver.handler
    )
    print(f"Listening on QUIC localhost:{args.port}; cache={args.cache}", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        server.close()
        await receiver.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--cert", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--port", type=int, default=5003)
    asyncio.run(run(p.parse_args()))
