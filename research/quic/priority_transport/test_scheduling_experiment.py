import csv
from pathlib import Path
import tempfile
import unittest

from run_deadline import GROUP, WRITE_BEGIN, WRITE_END, QUINN_PRIORITY, SCHEDULING_MODES
from run_scheduling_experiment import (MODES, QUINN_POLICIES, summarize,
                                       write_interaction_comparison,
                                       write_native_priority_deltas)


class SchedulingExperimentTests(unittest.TestCase):
    def test_end_marker_is_latched_across_out_of_order_group_completions(self):
        # An EOF task can complete before the last data stream. In the Rust
        # receive loop, later ordinary groups must not reset received_eof.
        source = (Path(__file__).parent / "src/main.rs").read_text()
        self.assertIn("received_eof |= done??;", source)
        self.assertNotIn("received_eof = done??;", source)

    def test_all_experiment_modes_are_distinct_from_adaptive(self):
        self.assertEqual(set(MODES), {
            "original", "sequential", "concurrent", "amirali_fifo",
            "amirali_strict", "amirali_weighted", "p0_first",
        })
        self.assertTrue(set(MODES).issubset(set(SCHEDULING_MODES)))
        self.assertNotIn("adaptive", MODES)
        self.assertEqual(QUINN_POLICIES, ("equal", "tiered"))

    def test_chunk_modes_use_quinn_and_application_ack(self):
        source = (Path(__file__).parent / "src/main.rs").read_text()
        self.assertIn("const CHUNK_SIZE: usize = 8192;", source)
        self.assertIn('"amirali_strict" =>', source)
        self.assertIn('"amirali_weighted" =>', source)
        self.assertIn('"amirali_fifo" =>', source)
        self.assertIn('let value = native_priority_value(job.tier, tiered);', source)
        self.assertIn('send.set_priority(value)?;', source)
        self.assertIn('if receipt != b"OK"', source)
        self.assertIn('mpsc::unbounded_channel::<ChunkJob>()', source)

    def test_group_arrival_and_write_start_logs(self):
        arrival = GROUP.fullmatch("GROUP_COMPLETE 114 tier=3 t_ms=201.987")
        start = WRITE_BEGIN.fullmatch("GROUP_WRITE_BEGIN 114 tier=0 t_ms=33.333")
        end = WRITE_END.fullmatch("GROUP_WRITE_END 114 tier=0 t_ms=33.555")
        priority = QUINN_PRIORITY.fullmatch("QUINN_PRIORITY frame=114 tier=0 value=30")
        self.assertEqual((int(arrival[1]), int(arrival[2]), float(arrival[3])),
                         (114, 3, 201.987))
        self.assertEqual((int(start[1]), int(start[2]), float(start[3])),
                         (114, 0, 33.333))
        self.assertEqual((int(end[1]), int(end[2]), float(end[3])),
                         (114, 0, 33.555))
        self.assertEqual((int(priority[1]), int(priority[2]), int(priority[3])),
                         (114, 0, 30))

    def test_p0_first_waits_for_quinn_acceptance_before_enhancement_writes(self):
        source = (Path(__file__).parent / "src/main.rs").read_text()
        code = source.split('let first_tier = if mode == "p0_first_priority"')[1]
        self.assertLess(code.index('write_ntc_group(p0, frame, 0, selected'),
                        code.index('for (offset, stream) in streams.into_iter()'))
        self.assertIn('fn native_priority_value(tier: usize, tiered: bool) -> i32', source)

    def test_aggregate_all_four_real_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "per_frame_deadline.csv"
            rows = []
            for frame, p0, full in [(114, 70.0, 150.0), (115, 105.0, 180.0)]:
                rows.append({
                    "frame": frame,
                    "selected_tiers": "4",
                    "sha256_group_verified": "yes",
                    "scheduling_mode": "concurrent",
                    "quinn_priority_policy": "tiered",
                    "p0_complete_ms_connection": str(p0),
                    "p1_complete_ms_connection": str(p0 + 10),
                    "p2_complete_ms_connection": str(p0 + 20),
                    "p3_complete_ms_connection": str(full),
                    "p0_write_start_ms_sender": str(p0 - 30.0),
                    "p0_delivery_after_write_end_ms": "18.1",
                    "p0_lead_before_full_ms": str(full - p0),
                    "completion_ms": str(full),
                    "write_start_spread_ms": "0.08",
                    "p0_deadline_missed": "False",
                    "deadline_missed": "False",
                    "payload_bytes": "63992",
                })
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            item = summarize(path, "concurrent", "tiered", 1, 30.0)
            self.assertEqual(item["frames"], 2)
            self.assertEqual(item["median_p0_lead_before_full_ms"], 77.5)
            self.assertEqual(item["payload_bytes"], 127984)
            self.assertEqual(item["p0_deadline_misses"], 0)
            self.assertEqual(item["median_concurrent_write_start_spread_ms"], 0.08)
            self.assertEqual(item["p0_completed_before_other_groups"], 2)
            self.assertEqual(item["median_p0_delivery_after_write_end_ms"], 18.1)
            with self.assertRaisesRegex(RuntimeError, "Invalid group receipt or mode"):
                summarize(path, "sequential", "tiered", 1, 30.0)
            with self.assertRaisesRegex(RuntimeError, "Invalid group receipt or mode"):
                summarize(path, "concurrent", "equal", 1, 30.0)

            fixed = summarize(path, "concurrent", "tiered", 1, 30.0,
                              tier_selection="fixed", fixed_tiers=4)
            self.assertEqual(fixed["selected_tier_histogram"],
                             '{"P0..P0":0,"P0..P1":0,"P0..P2":0,"P0..P3":2}')

    def test_native_priority_delta_is_within_same_admission_method(self):
        with tempfile.TemporaryDirectory() as folder:
            equal = dict(mode="concurrent", quinn_priority="equal", repetition=1, frames=30,
                         p0_deadline_misses=25, full_deadline_misses=30,
                         live_played_current_ticks=5)
            tiered = {**equal, "quinn_priority": "tiered", "p0_deadline_misses": 10,
                      "live_played_current_ticks": 19}
            output = Path(folder) / "delta.csv"
            write_native_priority_deltas(output, [equal, tiered])
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["mode"], "concurrent")
            self.assertEqual(float(rows[0]["quinn_priority_gain_p0_percentage_points"]), 50.0)
            self.assertEqual(int(rows[0]["quinn_priority_gain_current_ticks"]), 14)

    def test_tier_selection_supported_by_transport(self):
        source = (Path(__file__).parent / "src/main.rs").read_text()
        self.assertIn('"--tier-selection"', source)
        self.assertIn('"--fixed-tiers"', source)
        self.assertIn('o.tier_selection == "adaptive"', source)


if __name__ == "__main__":
    unittest.main()
