import json
from pathlib import Path
import pytest
from research.quic.priority_transport.player_bridge import (
    PlayerCacheBridge,
    ready_display_frame,
)
from q3dgstream import HEADER, RECORD


def test_playback_requires_contiguous_ntcs_and_matching_additions():
    assert ready_display_frame({0, 2}, {0, 1, 2}, 4) == 1
    assert ready_display_frame({0, 1, 2}, {0}, 4) == 1
    assert ready_display_frame({0, 1, 2}, {0, 1, 2}, 4) == 3
    assert ready_display_frame(set(), set(), 4) == 0


def test_qblk_waits_for_every_stage_and_preserves_bounds(tmp_path):
    import torch

    events = []
    bridge = PlayerCacheBridge(tmp_path, events.append)
    folder = tmp_path / "NTCs/progressive"
    folder.mkdir(parents=True)
    bounds = {"xyz_bound_min": [-1, -2, -3], "xyz_bound_max": [1, 2, 3]}
    (folder / "bounds.json").write_text(json.dumps(bounds))
    for stage in [3, 1, 0, 2]:
        ids = range(stage * 389, (stage + 1) * 389)
        raw = HEADER.pack(b"QNP01", 0, stage, 1556, 389) + b"".join(
            RECORD.pack(i, 0.25, bytes([0x89]) * 32) for i in ids
        )
        name = f"NTCs/progressive/NTC_000000_P{stage}.qblk"
        (tmp_path / name).write_bytes(raw)
        bridge.committed(name)
        if stage != 2:
            assert not events
    assert events == ["NTCs/NTC_000000.pth"]
    checkpoint = torch.load(tmp_path / events[0], weights_only=True)
    assert checkpoint["xyz_bound_min"].tolist() == bounds["xyz_bound_min"]
    assert checkpoint["model.params"].numel() == 31120 * 64
    assert checkpoint["model.params"][:2].tolist() == [0.25, 0.0]


def test_original_checkpoint_notification_and_stage_name_mismatch(tmp_path):
    events = []
    bridge = PlayerCacheBridge(tmp_path, events.append)
    p = tmp_path / "NTCs"
    p.mkdir()
    (p / "NTC_000000.pth").write_bytes(b"opaque checkpoint")
    bridge.committed("NTCs/NTC_000000.pth")
    assert events == ["NTCs/NTC_000000.pth"]
    name = "NTCs/progressive/NTC_000000_P0.qblk"
    (tmp_path / name).parent.mkdir()
    (tmp_path / name).write_bytes(HEADER.pack(b"QNP01", 1, 0, 1556, 0))
    with pytest.raises(ValueError):
        bridge.committed(name)
