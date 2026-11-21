import csv
import tempfile
import unittest
from pathlib import Path

from analyze_arrival_order import read_orders, summarize


class ArrivalOrderTests(unittest.TestCase):
    def write_sample(self, directory, rows):
        source = Path(directory) / "per_frame_all.csv"
        with source.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return source

    def sample(self):
        return {
            "repetition": 1, "mode": "concurrent", "quinn_priority": "tiered",
            "selected_tiers": 4, "frame": 114,
            "p0_complete_ms_connection": 15,
            "p1_complete_ms_connection": 12,
            "p2_complete_ms_connection": 14,
            "p3_complete_ms_connection": 13,
            "p0_net_ready_ms_connection": 10,
            "p1_net_ready_ms_connection": 11,
            "p2_net_ready_ms_connection": 12,
            "p3_net_ready_ms_connection": 13,
        }

    def test_quic_order_is_distinct_from_disk_order(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = read_orders(self.write_sample(directory, [self.sample()]))
        self.assertEqual(rows[0]["net_ready_order"], "P0>P1>P2>P3")
        self.assertEqual(rows[0]["disk_ready_order"], "P1>P3>P2>P0")
        self.assertTrue(rows[0]["p0_first"])
        self.assertEqual(rows[0]["timestamp_source"], "quic_reassembled")
        self.assertEqual(summarize(rows)[0]["p0_first_pct"], 100.0)

    def test_older_csv_has_explicit_disk_fallback(self):
        item = self.sample()
        for tier in range(4):
            del item[f"p{tier}_net_ready_ms_connection"]
        with tempfile.TemporaryDirectory() as directory:
            rows = read_orders(self.write_sample(directory, [item]))
        self.assertEqual(rows[0]["timestamp_source"], "disk_commit_only")
        self.assertEqual(rows[0]["net_ready_order"], "P1>P3>P2>P0")
        self.assertFalse(rows[0]["p0_first"])

    def test_equal_timestamps_do_not_count_as_unambiguous_first(self):
        item = self.sample()
        item["p1_net_ready_ms_connection"] = 10
        with tempfile.TemporaryDirectory() as directory:
            rows = read_orders(self.write_sample(directory, [item]))
        self.assertTrue(rows[0]["tied_net_timestamps"])
        self.assertFalse(rows[0]["p0_first"])


if __name__ == "__main__":
    unittest.main()
