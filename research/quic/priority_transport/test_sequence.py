"""CPU-only tests for the multi-NTC preparation and validation harness."""

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import run_sequence


class SequenceTests(unittest.TestCase):
    def test_requires_every_wire_and_priority_file(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            wire, priority = root / "wire", root / "orders"
            wire.mkdir()
            priority.mkdir()
            for i in range(3):
                (wire / f"NTC_{i:06}.pth").write_bytes(b"placeholder")
                (priority / f"NTC_{i:06}.npz").write_bytes(b"placeholder")
            args = Namespace(wire_dir=wire, priority_dir=priority, start_frame=0, frames=3)
            self.assertEqual(len(run_sequence.paths_for(args)), 3)
            (priority / "NTC_000001.npz").unlink()
            with self.assertRaises(FileNotFoundError):
                run_sequence.paths_for(args)

    def test_preparation_preserves_four_payloads_per_frame(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "results"
            output.mkdir()
            selection = [(i, root / f"source{i}.pth", root / f"order{i}.npz") for i in range(3)]

            def fake_prepare(wire, order, frame_dir):
                i = int(Path(wire).stem.removeprefix("source"))
                source = frame_dir / "send"
                source.mkdir()
                for group in range(4):
                    (source / f"P{group}.bin").write_bytes(bytes([i, group]))
                return {
                    "frame": i, "score_sequence_nonincreasing": True,
                    "payload_bytes_total": 8,
                    "groups": {f"P{p}": {"payload_bytes": 2} for p in range(4)},
                }

            with patch("run_sequence.prepare_ranked", side_effect=fake_prepare):
                manifests, size = run_sequence.prepare_all(selection, output)
            self.assertEqual(size, 24)
            self.assertEqual(sorted(manifests), [0, 1, 2])
            for frame in range(3):
                path = output / "send" / f"NTC_{frame:06}"
                for group in range(4):
                    self.assertEqual((path / f"P{group}.bin").read_bytes(), bytes([frame, group]))
                self.assertFalse((path / "send").exists())
            metadata = json.loads((output / "sequence_manifest.json").read_text())
            self.assertEqual(metadata["payload_bytes_total"], 24)

    def test_structured_timing_line(self):
        self.assertEqual(
            run_sequence.TIMING_RE.fullmatch("TIMING 114 0 58.123 59.891").groups(),
            ("114", "0", "58.123", "59.891"),
        )
        self.assertIsNone(run_sequence.TIMING_RE.fullmatch("TIMING 114 4 1.1 2.2"))


if __name__ == "__main__":
    unittest.main()
