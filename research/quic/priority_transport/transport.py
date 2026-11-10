"""Verified chunk transport for Q3DGStream files. ALPN intentionally differs from legacy."""

import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import time

from .core import CHUNK_SIZE, FileAssembler, Scheduler

ALPN = ["q3dg-priority-chunks/1"]


@dataclass(frozen=True)
class FileJob:
    path: Path
    remote: str
    priority: int
    release_ms: float = 0


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class Receiver:
    def __init__(self, cache, on_file=None, processing_delay=0):
        self.assembler = FileAssembler(cache)
        self.on_file = on_file
        self.tasks = set()
        self.events = []
        self.errors = []
        self.processing_delay = processing_delay

    def handler(self, reader, writer):
        t = asyncio.create_task(self.receive(reader, writer))
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

    async def receive(self, reader, writer):
        try:
            n = struct.unpack("!I", await asyncio.wait_for(reader.readexactly(4), 30))[
                0
            ]
            if not 0 < n <= 4096:
                raise ValueError("invalid header size")
            h = json.loads(await asyncio.wait_for(reader.readexactly(n), 30))
            if h.get("whole_file") is True:
                total = h["total"]
                if (
                    type(total) is not int
                    or not 0 < total <= 1024**3
                    or h["size"] != total
                ):
                    raise ValueError("invalid whole-file size")
                for offset in range(0, total, CHUNK_SIZE):
                    payload = await asyncio.wait_for(
                        reader.readexactly(min(CHUNK_SIZE, total - offset)), 30
                    )
                    part = dict(
                        h,
                        offset=offset,
                        size=len(payload),
                        chunk_sha256=hashlib.sha256(payload).hexdigest(),
                    )
                    if offset + len(payload) == total and await asyncio.wait_for(
                        reader.read(1), 30
                    ):
                        raise ValueError("unexpected trailing bytes")
                    complete = self.assembler.accept(part, payload)
            else:
                if type(h["size"]) is not int or not 0 < h["size"] <= CHUNK_SIZE:
                    raise ValueError("invalid payload size")
                payload = await asyncio.wait_for(reader.readexactly(h["size"]), 30)
                if await asyncio.wait_for(reader.read(1), 30):
                    raise ValueError("unexpected trailing bytes")
                processing_started = time.monotonic()
                if self.processing_delay:
                    await asyncio.sleep(self.processing_delay)
                complete = self.assembler.accept(h, payload)
            if complete and self.on_file is not None:
                await asyncio.to_thread(self.on_file, h["path"])
            received = time.monotonic()
            self.events.append(
                dict(
                    path=h["path"],
                    offset=h["offset"],
                    priority=h["priority"],
                    received=received,
                    complete=complete,
                )
            )
            writer.write(
                (
                    json.dumps(dict(ok=True, received=received, complete=complete,
                                    processing_ms=(received - processing_started) * 1000
                                    if not h.get("whole_file") else None))
                    + "\n"
                ).encode()
            )
            writer.write_eof()
        except Exception as exc:
            self.errors.append(repr(exc))
            try:
                writer.write(
                    (json.dumps(dict(ok=False, error=repr(exc))) + "\n").encode()
                )
                writer.write_eof()
            except Exception:
                pass

    async def close(self):
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*list(self.tasks), return_exceptions=True)
        self.assembler.close()


async def send_files(connection, jobs, *, policy="strict", max_inflight=128, timeout=120, on_event=None):
    """Send file chunks. Completion means receiver validated and atomically committed files.

    Read files lazily per chunk. Inputs must remain unchanged during transfer.
    Receipt timestamps use receiver monotonic time, comparable only on the same host.
    """
    if not 1 <= max_inflight <= 128:
        raise ValueError("max_inflight must be 1..128")
    scheduler = Scheduler(policy)
    jobs = list(jobs)
    if len({job.remote for job in jobs}) != len(jobs):
        raise ValueError("duplicate file name")
    pending = []
    for job in jobs:
        if job.priority not in range(4) or job.release_ms < 0:
            raise ValueError("bad priority/release")
        size = job.path.stat().st_size
        if not 0 < size <= 1024**3:
            raise ValueError("file size must be 1 byte..1 GiB")
        pending.append((job, size, file_hash(job.path)))
    started = time.monotonic()
    active = set()
    rows = []
    completed = set()

    async def send(item, selected):
        job, total, digest, offset = item
        with job.path.open("rb") as f:
            f.seek(offset)
            payload = f.read(min(CHUNK_SIZE, total - offset))
        h = dict(
            path=job.remote,
            total=total,
            offset=offset,
            size=len(payload),
            sha256=digest,
            chunk_sha256=hashlib.sha256(payload).hexdigest(),
            priority=job.priority,
        )
        encoded = json.dumps(h, separators=(",", ":")).encode()
        if len(encoded) > 4096:
            raise ValueError("header too long")
        row = dict(
            path=job.remote,
            priority=job.priority,
            offset=offset,
            size=len(payload),
            release_ms=job.release_ms,
            selected_ms=selected,
        )
        rows.append(row)
        reader, writer = await connection.create_stream()
        row["stream_id"] = writer.get_extra_info("stream_id")
        row["submitted_ms"] = (time.monotonic() - started) * 1000
        writer.write(struct.pack("!I", len(encoded)) + encoded + payload)
        writer.write_eof()
        if on_event:
            on_event("submitted", dict(row))
        response = json.loads(await asyncio.wait_for(reader.readline(), timeout))
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "receiver rejected chunk"))
        row["received_ms"] = (response["received"] - started) * 1000
        row["acked_ms"] = (time.monotonic() - started) * 1000
        row["processing_ms"] = response.get("processing_ms")
        if on_event:
            on_event("acked", dict(row))
        if response["complete"]:
            completed.add(job.remote)

    try:
        while pending or active or any(scheduler.q) or scheduler.fifo:
            elapsed = (time.monotonic() - started) * 1000
            for entry in pending[:]:
                job, size, digest = entry
                if job.release_ms <= elapsed:
                    for offset in range(0, size, CHUNK_SIZE):
                        scheduler.add(job.priority, (job, size, digest, offset))
                    pending.remove(entry)
            while len(active) < max_inflight:
                item = scheduler.pop()
                if item is None:
                    break
                active.add(asyncio.create_task(send(item[1], elapsed)))
            if active:
                done, active = await asyncio.wait(
                    active, timeout=0.005, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    task.result()
            else:
                await asyncio.sleep(0.001)
        if completed != {job.remote for job in jobs}:
            raise RuntimeError("missing committed file confirmation")
        return rows
    finally:
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)


async def send_whole_files(connection, jobs, *, timeout=120):
    """Controlled sequential-file reference, not the unavailable legacy wire code.

    One stream per complete file. Await commit before starting the next file.
    Keeps up to a whole file in aioquic's pending buffer, as a simple file sender.
    """
    rows = []
    started = time.monotonic()
    for job in jobs:
        remaining = job.release_ms / 1000 - (time.monotonic() - started)
        if remaining > 0:
            await asyncio.sleep(remaining)
        size = job.path.stat().st_size
        if not 0 < size <= 1024**3:
            raise ValueError("invalid file size")
        h = dict(
            path=job.remote,
            size=size,
            total=size,
            offset=0,
            priority=job.priority,
            sha256=file_hash(job.path),
            whole_file=True,
        )
        header = json.dumps(h, separators=(",", ":")).encode()
        if len(header) > 4096:
            raise ValueError("header too long")
        reader, writer = await connection.create_stream()
        selected = (time.monotonic() - started) * 1000
        row = dict(
            path=job.remote,
            priority=job.priority,
            release_ms=job.release_ms,
            selected_ms=selected,
            submitted_ms=selected,
            stream_id=writer.get_extra_info("stream_id"),
            size=size,
        )
        writer.write(struct.pack("!I", len(header)) + header)
        with job.path.open("rb") as f:
            while block := f.read(1024 * 1024):
                writer.write(block)
        writer.write_eof()
        response = json.loads(await asyncio.wait_for(reader.readline(), timeout))
        if not response.get("ok") or not response.get("complete"):
            raise RuntimeError(response)
        row["received_ms"] = (response["received"] - started) * 1000
        row["acked_ms"] = (time.monotonic() - started) * 1000
        rows.append(row)
    return rows
