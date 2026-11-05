"""Application scheduling and atomic file reconstruction, independent of rendering."""

import collections
import hashlib
import os
import tempfile
from pathlib import Path, PurePosixPath

CHUNK_SIZE = 8192


class Scheduler:
    """Choose the next application chunk; P0 is highest priority."""

    def __init__(self, policy):
        if policy not in ("fifo", "strict", "weighted"):
            raise ValueError(policy)
        self.policy = policy
        self.q = [collections.deque() for _ in range(4)]
        self.fifo = collections.deque()
        self.cycle = [0] * 8 + [1] * 4 + [2] * 2 + [3]
        self.pos = 0

    def add(self, p, value):
        if p not in range(4):
            raise ValueError(p)
        if self.policy == "fifo":
            self.fifo.append((p, value))
        else:
            self.q[p].append(value)

    def pop(self):
        if self.policy == "fifo":
            return self.fifo.popleft() if self.fifo else None
        if self.policy == "strict":
            for p, q in enumerate(self.q):
                if q:
                    return p, q.popleft()
            return None
        for _ in self.cycle:
            p = self.cycle[self.pos]
            self.pos = (self.pos + 1) % len(self.cycle)
            if self.q[p]:
                return p, self.q[p].popleft()
        return None


class FileAssembler:
    """Local experiment receiver. Owns temporary files, never publishes partial data."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.pending = {}
        self.completed = set()

    def accept(self, h, payload):
        name = h["path"]
        rel = PurePosixPath(name)
        if not name or rel.is_absolute() or ".." in rel.parts or "\\" in name:
            raise ValueError("unsafe relative path")
        dst = self.root.joinpath(*rel.parts)
        if not dst.resolve().is_relative_to(self.root):
            raise ValueError("path escapes cache")
        for ancestor in [dst, *dst.parents]:
            if ancestor == self.root:
                break
            if ancestor.is_symlink():
                raise ValueError("symlink in output path")
        size, total, offset = h["size"], h["total"], h["offset"]
        if any(type(v) is not int for v in (size, total, offset)):
            raise ValueError("non-integer length")
        if (
            not 0 < total <= 1024**3
            or offset < 0
            or offset % CHUNK_SIZE
            or offset >= total
        ):
            raise ValueError("invalid range")
        if size != min(CHUNK_SIZE, total - offset) or len(payload) != size:
            raise ValueError("invalid chunk size")
        if hashlib.sha256(payload).hexdigest() != h["chunk_sha256"]:
            raise ValueError("chunk hash mismatch")
        if name in self.completed:
            raise ValueError("file already committed")
        if name not in self.pending:
            if len(self.pending) >= 128:
                raise ValueError("too many partial files")
            if dst.exists():
                raise ValueError("refusing to overwrite cache file")
            dst.parent.mkdir(parents=True, exist_ok=True)
            f = tempfile.NamedTemporaryFile(
                dir=dst.parent, suffix=".part", delete=False
            )
            self.pending[name] = dict(
                file=f,
                path=Path(f.name),
                dst=dst,
                total=total,
                sha256=h["sha256"],
                offsets=set(),
            )
        state = self.pending[name]
        if (state["total"], state["sha256"]) != (total, h["sha256"]) or offset in state[
            "offsets"
        ]:
            raise ValueError("inconsistent or duplicate chunk")
        f = state["file"]
        f.seek(offset)
        f.write(payload)
        state["offsets"].add(offset)
        if len(state["offsets"]) != (total + CHUNK_SIZE - 1) // CHUNK_SIZE:
            return False
        f.flush()
        f.seek(0)
        digest = hashlib.sha256()
        while data := f.read(1024 * 1024):
            digest.update(data)
        if digest.hexdigest() != state["sha256"]:
            raise ValueError("file hash mismatch")
        f.close()
        os.replace(state["path"], state["dst"])
        del self.pending[name]
        self.completed.add(name)
        return True

    def close(self):
        for state in self.pending.values():
            state["file"].close()
            state["path"].unlink(missing_ok=True)
        self.pending.clear()
