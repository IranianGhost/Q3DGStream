"""CPU-only live transport/receiver edge cases (no GPU required)."""

import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from block_codec import HEADER, MAGIC, RECORD
from live_client import (atomic_json, highest_ready_tier, monitor_future,
                         percentile, received_state)


class LiveClientTests(unittest.TestCase):
    def test_p95_and_single_measurement(self):
        self.assertIsNone(percentile([], 0.95))
        self.assertEqual(percentile([8], .95), 8)
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], .95), 4.8)

    def test_complete_group_detection_never_bypasses_p0(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            m = {"groups": {f"P{i}": {} for i in range(4)}}
            (path / "P1.bin").write_bytes(b"P1")
            self.assertEqual(highest_ready_tier(path, m), 0)
            (path / "P0.bin").write_bytes(b"P0")
            self.assertEqual(highest_ready_tier(path, m), 2)
            (path / "P3.bin").write_bytes(b"P3")
            self.assertEqual(highest_ready_tier(path, m), 2)

    def test_contiguous_buffer_enforces_cumulative_order(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)
            for i in (0, 2, 3):
                folder=path / f"NTC_{i:06}"
                folder.mkdir()
                (folder / "P0.bin").write_bytes(b"ok")
            self.assertEqual(monitor_future(path, 0, 4), 1)
            self.assertEqual(monitor_future(path, 1, 4), 0)

    def test_received_group_uses_only_verified_wire_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)
            metadata = b'{"source_state_sha256":"original"}'
            record=RECORD.pack(9, 0.1, b"a"*32)
            packet=HEADER.pack(MAGIC,0,1,len(metadata))+metadata+record
            (path / "P0.bin").write_bytes(packet)
            frame_m = {"source_state_sha256": "original", "retained_blocks": 4,
                       "groups": {"P0": {"sha256":hashlib.sha256(packet).hexdigest(), "block_count":1}}}
            with patch("live_client.assemble_state", return_value="state") as builder:
                self.assertEqual(received_state(path, frame_m, 1), ("state", 1))
                self.assertEqual(set(builder.call_args.args[1]), {9})
            (path / "P0.bin").write_bytes(packet+b"bad")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                received_state(path, frame_m, 1)


if __name__ == "__main__":
    unittest.main()
