"""Pure-Python tests: no GPU, network namespaces, or Rust compiler needed."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from evaluate_readiness import (candidate_bytes, estimated_rate_mbps, evaluate,
                                gated_gpu_prediction_ms, predict_transport_ms)
from run_deadline import PREDICT_CONTEXT, SENDER_EPOCH


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frame_row(frame=0, selected=2):
    return {
        "frame": frame, "selected_tiers": selected,
        "payload_bytes": selected * 10000,
        "sender_decision_ms": frame * (1000 / 30),
        "sender_outstanding_bytes": 0 if frame == 0 else 8000,
        "stage_p0_bytes": 10000, "stage_p1_bytes": 10000,
        "stage_p2_bytes": 10000, "stage_p3_bytes": 10000,
        "sender_to_receiver_clock_offset_ms": 2.0,
        "initial_capacity_ceiling_mbps": 5,
        "sender_utilization": 0.9, "rtt_ms": 20,
        "cwnd_bytes": 10000, "available_mbps": 3.2,
        "observed_rtt_jitter_sample_ms": 0.0,
        "predicted_decode_ms": 6.65, "predicted_inference_ms": 2.6,
        "predicted_render_ms": 6.3, "client_compute_ms": 15.55,
        "deadline_ms_receiver": 150 + frame * 1000 / 30,
        "p0_complete_ms_connection": 30 + frame * 100,
        "p1_complete_ms_connection": 100 + frame * 100,
        "p2_complete_ms_connection": "", "p3_complete_ms_connection": "",
    }


class ReadinessTest(unittest.TestCase):
    def test_instrumentation_regex(self):
        sample = ("PREDICT_CONTEXT frame=114 send_ms=3820.123 outstanding_bytes=16384 "
                  "stage_bytes=17259,19000,22000,26000 initial_mbps=10.000 "
                  "utilization=0.900 decode_ms=6.650 inference_ms=2.600 "
                  "render_ms=6.300 buffer_frames=2 last_applied_ntc=110")
        self.assertIsNotNone(PREDICT_CONTEXT.fullmatch(sample))
        self.assertEqual(PREDICT_CONTEXT.fullmatch(sample)[14], "110")
        self.assertIsNotNone(SENDER_EPOCH.fullmatch("SENDER_EPOCH unix_ms=1795180000000.000"))

    def test_models_and_receiver_clock(self):
        row = frame_row()
        self.assertEqual(candidate_bytes(row, 2), 20000)
        self.assertAlmostEqual(estimated_rate_mbps(row, "configured"), 4.5)
        self.assertAlmostEqual(estimated_rate_mbps(row, "cwnd"), 3.6)
        self.assertAlmostEqual(estimated_rate_mbps(row, "hybrid"), 3.2)
        self.assertAlmostEqual(predict_transport_ms(row, 1, "hybrid"), 37.0)
        self.assertAlmostEqual(predict_transport_ms(row, 2, "hybrid"), 62.0)
        row["sender_to_receiver_clock_offset_ms"] = -1.0
        self.assertAlmostEqual(predict_transport_ms(row, 1, "hybrid"), 34.0)

    def test_gpu_tick_gate_and_split_work(self):
        row = frame_row()
        tick = 1000 / 30
        before = gated_gpu_prediction_ms(row, 130, 150, tick, include_render=False)
        after = gated_gpu_prediction_ms(row, 155, 150, tick, include_render=False)
        self.assertAlmostEqual(before, 159.25)
        self.assertAlmostEqual(after, 150 + tick + 9.25)
        self.assertAlmostEqual(gated_gpu_prediction_ms(row, 130, 150, tick, True), 165.55)

    def test_evaluate_transport_and_catchup_gpu(self):
        with tempfile.TemporaryDirectory() as scratch:
            run = Path(scratch) / "rep_01" / "concurrent_priority"
            trial = run / "priority"
            csv_write(trial / "per_frame_deadline.csv", [frame_row(0), frame_row(1)])
            csv_write(trial / "live" / "apply_events.csv", [
                {"ntc": 0, "tick": 1, "applied_tiers": 1,
                 "inference_ready_ms": "204.0", "rendered_this_tick": 0,
                 "render_ready_ms": ""},
                {"ntc": 1, "tick": 1, "applied_tiers": 2,
                 "inference_ready_ms": "213.0", "rendered_this_tick": 1,
                 "render_ready_ms": "221.0"},
            ])
            report = evaluate(run, fps=30)
            self.assertEqual(report["transport_frames"], 2)
            self.assertEqual(report["gpu_applied_ntcs"], 2)
            self.assertEqual(report["gpu_individually_rendered_ntcs"], 1)
            self.assertEqual(report["gpu_applied_but_not_individually_rendered"], 1)
            with (trial / "readiness" / "transport_predictions.csv").open() as handle:
                transport = list(csv.DictReader(handle))
            with (trial / "readiness" / "gpu_inference_predictions.csv").open() as handle:
                inference = list(csv.DictReader(handle))
            with (trial / "readiness" / "gpu_display_predictions.csv").open() as handle:
                display = list(csv.DictReader(handle))
            self.assertEqual(len(transport), 2 * 2 * 3)
            self.assertEqual(len(inference), 2 * 3)
            self.assertEqual(len(display), 1 * 3)
            self.assertEqual(display[0]["frame"], "1")
            self.assertEqual(json.loads((trial / "readiness" / "readiness_summary.json").read_text())
                             ["gpu_instrumented"], True)

    def test_single_frame_needs_fps_without_config(self):
        with tempfile.TemporaryDirectory() as scratch:
            trial = Path(scratch) / "priority"
            csv_write(trial / "per_frame_deadline.csv", [frame_row()])
            with self.assertRaisesRegex(ValueError, "requires --fps"):
                evaluate(trial)
            self.assertEqual(evaluate(trial, fps=30)["transport_frames"], 1)

    def test_selected_tiers_must_match_payload(self):
        with tempfile.TemporaryDirectory() as scratch:
            trial = Path(scratch) / "priority"
            row = frame_row()
            row["payload_bytes"] = 20001
            csv_write(trial / "per_frame_deadline.csv", [row])
            with self.assertRaisesRegex(ValueError, "disagree"):
                evaluate(trial, fps=30)


if __name__ == "__main__":
    unittest.main()
