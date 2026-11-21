import hashlib
from pathlib import Path
import tempfile
import unittest

from run_deadline import DECISION, FRAME, verify_received
from deadline_policy import select_tiers


class DeadlineParserTests(unittest.TestCase):
    def test_5mbps_prefers_p0_at_30fps(self):
        decision = select_tiers([17261, 15577, 15577, 15577],
                                now_ms=0, deadline_ms=150,
                                effective_mbps=4.6, fps=30, rtt_ms=20,
                                backlog_bytes=0, client_compute_ms=15)
        self.assertEqual(decision.tiers, 1)
        self.assertTrue(decision.deadline_feasible)

    def test_20mbps_allows_full(self):
        decision = select_tiers([17261, 15577, 15577, 15577],
                                now_ms=0, deadline_ms=150,
                                effective_mbps=18.4, fps=30, rtt_ms=20,
                                backlog_bytes=0, client_compute_ms=15)
        self.assertEqual(decision.tiers, 4)

    def test_overload_never_silently_drops_delta(self):
        decision = select_tiers([17261, 15577, 15577, 15577],
                                now_ms=100, deadline_ms=110,
                                effective_mbps=1.2, fps=30, rtt_ms=20,
                                backlog_bytes=20000, client_compute_ms=15)
        self.assertEqual(decision.tiers, 1)
        self.assertFalse(decision.deadline_feasible)

    def test_gpu_too_slow_cannot_claim_30fps(self):
        decision = select_tiers([17261, 15577, 15577, 15577],
                                now_ms=0, deadline_ms=500,
                                effective_mbps=18.4, fps=30, rtt_ms=20,
                                backlog_bytes=0, client_compute_ms=55)
        self.assertEqual(decision.tiers, 1)
        self.assertFalse(decision.deadline_feasible)

    def test_decision_metrics(self):
        line = ("DECISION frame=114 tiers=1 bytes=17261 capacity_mbps=4.600 "
                "measured_goodput_mbps=4.301 cwnd=16384 rtt_ms=20.00 "
                "min_rtt_ms=19.50 jitter_sample_ms=1.10 "
                "backlog_ms=29.33 client_ms=11.50 gpu_source=external_client_json "
                "deadline_ms=3950.00 loss_delta=0 sent_delta=17 "
                "loss_fraction=0.00000 congestion_delta=1")
        match = DECISION.fullmatch(line)
        self.assertIsNotNone(match)
        self.assertEqual(int(match[1]), 114)
        self.assertEqual(int(match[2]), 1)
        self.assertEqual(float(match[11]), 11.5)

    def test_frame_deadline(self):
        match = FRAME.fullmatch(
            "FRAME_COMPLETE 114 tiers=1 t_ms=3940.000 deadline_ms=3950.000 late=false")
        self.assertIsNotNone(match)
        self.assertEqual(match[5], "false")

    def test_exact_group_hash_and_missing_tier(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "priority/received/NTC_000114"
            directory.mkdir(parents=True)
            bodies = [b"actual ranked P0", b"real P1 bytes"]
            for n, raw in enumerate(bodies):
                (directory / f"P{n}.bin").write_bytes(raw)
            manifest = {114: {
                "groups": {f"P{n}": {"sha256": hashlib.sha256(raw).hexdigest()}
                           for n, raw in enumerate(bodies)}}}
            self.assertEqual(verify_received(root, manifest, {114: 2}), {114: 2})
            (directory / "P1.bin").write_bytes(b"corrupted")
            with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                verify_received(root, manifest, {114: 2})

    def test_extra_group_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "priority/received/NTC_000001"
            directory.mkdir(parents=True)
            (directory / "P0.bin").write_bytes(b"P0")
            (directory / "P1.bin").write_bytes(b"not selected")
            manifest = {1: {"groups": {
                "P0": {"sha256": hashlib.sha256(b"P0").hexdigest()}}}}
            with self.assertRaisesRegex(RuntimeError, "Unexpected group count"):
                verify_received(root, manifest, {1: 1})


if __name__ == "__main__":
    unittest.main()
