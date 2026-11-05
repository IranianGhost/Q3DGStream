from pathlib import Path
import hashlib
import pytest
from research.quic.priority_transport.core import Scheduler, FileAssembler


def test_late_urgent_and_weighted_progress():
    q = Scheduler("strict")
    q.add(3, "low")
    q.add(0, "urgent")
    assert q.pop() == (0, "urgent")
    q = Scheduler("weighted")
    for p in range(4):
        for i in range(20):
            q.add(p, i)
    picked = [q.pop()[0] for _ in range(15)]
    assert [picked.count(p) for p in range(4)] == [8, 4, 2, 1]


def header(data, offset, size, name="NTCs/progressive/NTC_000000_P0.qblk"):
    chunk = data[offset : offset + size]
    return dict(
        path=name,
        total=len(data),
        offset=offset,
        size=len(chunk),
        sha256=hashlib.sha256(data).hexdigest(),
        chunk_sha256=hashlib.sha256(chunk).hexdigest(),
    )


def test_reassembly_is_atomic_and_out_of_order(tmp_path):
    a = FileAssembler(tmp_path)
    data = b"a" * 8192 + b"b" * 17
    h = header(data, 8192, 8192)
    assert not a.accept(h, data[8192:])
    assert not (tmp_path / h["path"]).exists()
    assert a.accept(header(data, 0, 8192), data[:8192])
    assert (tmp_path / h["path"]).read_bytes() == data
    a.close()


@pytest.mark.parametrize("name", ["../bad", "/tmp/bad", "x/../bad"])
def test_bad_paths(tmp_path, name):
    a = FileAssembler(tmp_path)
    with pytest.raises(ValueError):
        a.accept(header(b"abc", 0, 8192, name), b"abc")
    a.close()


def test_bad_hash_and_duplicate(tmp_path):
    a = FileAssembler(tmp_path)
    data = b"a" * 9000
    h = header(data, 0, 8192)
    with pytest.raises(ValueError):
        a.accept(h, b"z" * 8192)
    a.accept(h, data[:8192])
    with pytest.raises(ValueError):
        a.accept(h, data[:8192])
    a.close()
    assert not list(tmp_path.rglob("*.part"))


def test_file_hash_failure_never_commits(tmp_path):
    a = FileAssembler(tmp_path)
    h = header(b"abc", 0, 8192)
    h["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        a.accept(h, b"abc")
    assert not (tmp_path / h["path"]).exists()
    a.close()
