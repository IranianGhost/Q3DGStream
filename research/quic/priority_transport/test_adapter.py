import asyncio
from types import SimpleNamespace
from research.quic.dynamic_priority_adapter import send_dynamic_priority as adapter


def test_adapter_preserves_metadata_addition_and_frame_barriers(tmp_path, monkeypatch):
    calls = []

    async def fake(connection, jobs, **options):
        calls.append(
            ([job.remote for job in jobs], [job.priority for job in jobs], options)
        )
        return []

    monkeypatch.setattr("research.quic.priority_transport.transport.send_files", fake)
    args = SimpleNamespace(policy="weighted", max_inflight=4, warmup_delay=0)
    p = tmp_path / "file"
    p.write_bytes(b"x")
    frames = [(i, p, [p] * 4) for i in range(2)]
    asyncio.run(adapter.send_priority_scene(object(), p, p, p, frames, args))
    assert len(calls) == 5
    assert calls[0][0] == [
        "NTCs/progressive/bounds.json",
        "init_3dgs.ply",
        "NTCs/config.json",
    ]
    assert calls[1][0] == ["additional_3dgs/additions_000000.ply"]
    assert calls[2][1] == [0, 1, 2, 3]
    assert calls[3][0] == ["additional_3dgs/additions_000001.ply"]
    assert calls[4][0][-1] == "NTCs/progressive/NTC_000001_P3.qblk"
    assert calls[2][2]["policy"] == "weighted"
