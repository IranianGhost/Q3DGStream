"""CPU-only tests of exact-tier prefetch, ordering and bounded cache lifetime."""

import tempfile
import time
from pathlib import Path
import unittest
from contextlib import nullcontext

from ntc_preparation import NTCPreparation, AsyncCUDAJobs, PreparedNTC, gpu_prefetch_guard_ms


class FakeStream:
    def __init__(self):
        self.waits = []
        self.syncs = 0

    def wait_event(self, event):
        self.waits.append(event)

    def synchronize(self):
        self.syncs += 1


class FakeEvent:
    def __init__(self, enable_timing=False):
        self.recorded = False
        self.complete = False

    def record(self, stream):
        self.recorded = True

    def query(self):
        return self.complete

    def elapsed_time(self, other):
        assert self.complete and other.complete
        return 6.5


class FakeCuda:
    def __init__(self):
        self.events = []
        self.playout = FakeStream()

    def Stream(self):
        return FakeStream()

    def Event(self, enable_timing=False):
        event = FakeEvent(enable_timing)
        self.events.append(event)
        return event

    def stream(self, stream):
        return nullcontext()

    def current_stream(self):
        return self.playout


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


def until(predicate, seconds=1.0):
    limit = time.monotonic() + seconds
    while time.monotonic() < limit:
        if predicate():
            return True
        time.sleep(.002)
    return False


class PreparationTests(unittest.TestCase):
    def test_async_slots_and_event_driven_consumption(self):
        fake = FakeTorch()
        calls = []

        def dequantize(quantized):
            calls.append(quantized)
            return {"model.params": f"dense-{quantized}"}

        gpu = AsyncCUDAJobs(fake, dequantize)
        e0 = PreparedNTC(0, 2, "f0p2", .5)
        e1 = PreparedNTC(1, 2, "f1p2", .5)
        e2 = PreparedNTC(2, 2, "f2p2", .5)
        first = gpu.launch(e0)
        second = gpu.launch(e1)
        self.assertIsNone(gpu.launch(e2))
        self.assertIsNone(gpu.launch(e0))
        self.assertEqual(calls, ["f0p2", "f1p2"])
        dense, ready = gpu.consume(first)
        self.assertEqual(dense, "dense-f0p2")
        self.assertFalse(ready)
        self.assertEqual(fake.cuda.playout.waits, [first.finish_event])
        gpu.retire(1)
        self.assertEqual(len(gpu.jobs), 2)  # Never recycle in-flight device buffers.
        first.start_event.complete = first.finish_event.complete = True
        second.start_event.complete = second.finish_event.complete = True
        gpu.retire(1)
        self.assertEqual(first.gpu_ms, 6.5)
        self.assertNotIn(first.key, gpu.jobs)
        self.assertIsNotNone(gpu.launch(e2))
        _, ready = gpu.consume(second)
        self.assertTrue(ready)

    def test_future_exact_tier_selection(self):
        with tempfile.TemporaryDirectory() as path:
            worker = NTCPreparation(Path(path), {str(i): {} for i in range(4)},
                                    0, 4, lambda f, m: 2,
                                    lambda f, m, t: (f.name, 1), ahead=4)
            worker.start()
            try:
                self.assertTrue(until(lambda: worker.get(3, 2) is not None))
                self.assertEqual(worker.pending_dense(0, required_tiers=2).frame, 0)
                self.assertEqual(worker.pending_dense(3, required_tiers=2,
                                                     excluded={(0, 2), (1, 2)}).frame, 2)
                self.assertIsNone(worker.pending_dense(3, required_tiers=1))
            finally:
                worker.stop()

    def test_dynamic_guard_admits_work_between_30fps_ticks(self):
        self.assertEqual(gpu_prefetch_guard_ms([]), 18.0)
        # A 6-7ms CUDA stage now fits a ~15ms idle gap (v1 needed 45ms).
        self.assertEqual(gpu_prefetch_guard_ms([6, 6.5, 7]), 12.0)
        # A costly outlier raises the admission threshold automatically.
        self.assertGreater(gpu_prefetch_guard_ms([6, 7, 23]), 20)
        self.assertEqual(gpu_prefetch_guard_ms([7] * 20, safety_ms=7), 14)

    def test_cpu_stage_upgrades_are_exact_and_preserve_future(self):
        with tempfile.TemporaryDirectory() as path:
            availability = {0: 0, 1: 0, 2: 0}
            seen = []

            def tiers(folder, manifest):
                return availability[int(folder.name.split("_")[-1])]

            def assemble(folder, manifest, num):
                frame = int(folder.name.split("_")[-1])
                seen.append((frame, num))
                return f"f{frame}p{num}", 42

            worker = NTCPreparation(Path(path), {str(i): {} for i in range(3)},
                                    0, 3, tiers, assemble, ahead=2, poll_s=.001)
            worker.start()
            try:
                availability[0] = 1
                self.assertTrue(until(lambda: worker.get(0, 1) is not None))
                self.assertEqual(worker.get(0, 1).quantized, "f0p1")
                self.assertIsNone(worker.get(0, 2))
                availability[0] = 2
                self.assertTrue(until(lambda: worker.get(0, 2) is not None))
                self.assertEqual(worker.get(0, 2).quantized, "f0p2")
                self.assertEqual(worker.pending_dense(0).tiers, 2)
                worker.get(0, 2).dense = "already-decoded"
                # Do not go back and waste GPU time on obsolete P0-only.
                self.assertIsNone(worker.pending_dense(0))
                # Once cumulative frame 0 is applied, discard its versions.
                worker.advance(1)
                self.assertIsNone(worker.get(0, 1))
                self.assertIsNone(worker.get(0, 2))
                availability[2] = 1
                self.assertTrue(until(lambda: worker.get(2, 1) is not None))
                self.assertNotIn((0, 3), seen)
            finally:
                worker.stop()

    def test_worker_failure_propagates(self):
        with tempfile.TemporaryDirectory() as path:
            worker = NTCPreparation(Path(path), {"0": {}}, 0, 1,
                                    lambda folder, manifest: 1,
                                    lambda folder, manifest, tiers: 1 / 0,
                                    poll_s=.001)
            worker.start()
            self.assertTrue(until(lambda: worker._error is not None))
            with self.assertRaisesRegex(RuntimeError, "division by zero"):
                worker.stop()


if __name__ == "__main__":
    unittest.main()
