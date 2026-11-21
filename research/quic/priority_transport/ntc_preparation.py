"""Exact-tier CPU preparation and optional event-driven CUDA prefetch.

This worker never touches CUDA, OpenGL, a renderer, or the Gaussian scene. The
playout thread alone may dequantize prepared states onto a GPU and, *at the
scheduled tick*, install their weights and run cumulative NTC inference.
AsyncCUDAJobs must be owned by that playout thread; the CPU worker never uses CUDA.
"""

from dataclasses import dataclass
from pathlib import Path
import math
import threading
import time


def gpu_prefetch_guard_ms(samples, safety_ms=5.0, minimum_ms=12.0,
                          initial_ms=18.0):
    """Conservative idle-time admission based on recent actual CUDA costs.

    Unlike v1's fixed 45ms threshold, this admits a ~6-7ms dequantization
    during a ~13-15ms gap between 30 FPS ticks. An observed slow CUDA stage
    increases the guard automatically. No GPU work is ever guaranteed to fit.
    """
    if not samples:
        return max(initial_ms, minimum_ms)
    values = sorted(float(x) for x in samples if x > 0)
    if not values:
        return max(initial_ms, minimum_ms)
    rank = (len(values) - 1) * .95
    lo, hi = math.floor(rank), math.ceil(rank)
    p95 = values[lo] + (values[hi] - values[lo]) * (rank - lo)
    return max(minimum_ms, p95 + safety_ms)


@dataclass
class PreparedNTC:
    frame: int
    tiers: int
    quantized: object
    assembly_ms: float
    dense: object = None
    dequant_ms: float = 0.0


@dataclass
class CUDAJob:
    """Own the sources and results until the CUDA completion event fires."""
    prepared: PreparedNTC
    dense: object
    start_event: object
    finish_event: object
    submitted_at: float
    enqueue_ms: float
    completion_observed_at: float | None = None
    gpu_ms: float | None = None
    used: bool = False

    @property
    def key(self):
        return self.prepared.frame, self.prepared.tiers


class AsyncCUDAJobs:
    """At most two complete future weight sets, prepared on a non-default stream.

    Never installs the weights or mutates the Gaussian scene. In particular,
    event.query() only *observes* GPU progress and cannot force a host sync.
    The playout thread waits on the matching event before consuming tensors.
    """

    def __init__(self, torch_module, dequantize, max_slots=2):
        if max_slots != 2:
            raise ValueError("The experiment uses exactly two staging slots")
        self.torch = torch_module
        self.dequantize = dequantize
        self.stream = torch_module.cuda.Stream()
        self.max_slots = max_slots
        self.jobs: dict[tuple[int, int], CUDAJob] = {}

    def poll(self):
        """Update completion and GPU timings without waiting for the device."""
        for job in self.jobs.values():
            if job.completion_observed_at is None and job.finish_event.query():
                job.completion_observed_at = time.monotonic()
                job.gpu_ms = job.start_event.elapsed_time(job.finish_event)

    def retire(self, next_unapplied):
        self.poll()
        # Never release a CUDA source/destination while its kernel is in flight.
        for key, job in list(self.jobs.items()):
            if job.prepared.frame < next_unapplied and job.completion_observed_at is not None:
                del self.jobs[key]

    def launch(self, candidate: PreparedNTC):
        key = candidate.frame, candidate.tiers
        if key in self.jobs or len(self.jobs) >= self.max_slots:
            return None
        now = time.monotonic()
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        with self.torch.cuda.stream(self.stream):
            start.record(self.stream)
            dense = self.dequantize(candidate.quantized)["model.params"]
            end.record(self.stream)
        job = CUDAJob(candidate, dense, start, end, now,
                      (time.monotonic() - now) * 1000)
        self.jobs[key] = job
        return job

    def get(self, frame, tiers):
        return self.jobs.get((frame, tiers))

    def consume(self, job):
        """Order cross-stream GPU use; caller syncs its own stream for timing."""
        self.poll()
        was_ready = job.completion_observed_at is not None
        self.torch.cuda.current_stream().wait_event(job.finish_event)
        job.used = True
        return job.dense, was_ready

    def finish_logging(self):
        """Outside all playout ticks, collect timings for surviving jobs."""
        self.stream.synchronize()
        self.poll()


class NTCPreparation:
    """Cache exact `(NTC, contiguous tiers)` states in a bounded lookahead.

    The worker never silently downgrades a tier: the playout thread must ask
    for the exact tier count available at its tick and decode synchronously if
    that exact cache entry is not ready.
    """

    def __init__(self, received_dir: Path, manifests: dict, first: int, last: int,
                 tier_count, assemble, ahead: int = 4, poll_s: float = .003):
        self.received_dir = received_dir
        self.manifests = manifests
        self.first = first
        self.last = last
        self.tier_count = tier_count
        self.assemble = assemble
        self.ahead = ahead
        self.poll_s = poll_s
        self._focus = first
        self._entries: dict[tuple[int, int], PreparedNTC] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._worker = threading.Thread(target=self._run, name="ntc-cpu-prefetch", daemon=True)

    def start(self):
        self._worker.start()

    def stop(self):
        self._stop.set()
        if self._worker.ident is not None:
            self._worker.join(timeout=3)
            if self._worker.is_alive():
                raise RuntimeError("NTC CPU preparation worker did not stop")
        self.check_error()

    def check_error(self):
        with self._lock:
            err = self._error
        if err is not None:
            raise RuntimeError(f"NTC CPU preparation failed: {err}") from err

    def advance(self, next_unapplied: int):
        with self._lock:
            self._focus = next_unapplied
            for key in list(self._entries):
                if key[0] < next_unapplied:
                    del self._entries[key]

    def get(self, frame: int, tiers: int):
        with self._lock:
            return self._entries.get((frame, tiers))

    def pending_dense(self, before_frame: int, required_tiers: int = 0,
                      excluded=frozenset()):
        """Select oldest most-complete future state, optionally exact-tier only."""
        with self._lock:
            latest_by_frame = {}
            for entry in self._entries.values():
                if (not self._focus <= entry.frame <= before_frame or
                        (required_tiers and entry.tiers != required_tiers) or
                        (entry.frame, entry.tiers) in excluded):
                    continue
                previous = latest_by_frame.get(entry.frame)
                if previous is None or entry.tiers > previous.tiers:
                    latest_by_frame[entry.frame] = entry
            eligible = [e for e in latest_by_frame.values() if e.dense is None]
            if not eligible:
                return None
            return min(eligible, key=lambda e: e.frame)

    def _run(self):
        try:
            while not self._stop.is_set():
                with self._lock:
                    focus = self._focus
                    known = set(self._entries)
                did_work = False
                for frame in range(focus, min(self.last, focus + self.ahead)):
                    if self._stop.is_set():
                        return
                    folder = self.received_dir / f"NTC_{frame:06}"
                    tiers = self.tier_count(folder, self.manifests[str(frame)])
                    if not tiers or (frame, tiers) in known:
                        continue
                    t0 = time.perf_counter()
                    state, _ = self.assemble(folder, self.manifests[str(frame)], tiers)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    with self._lock:
                        if frame >= self._focus:
                            self._entries[(frame, tiers)] = PreparedNTC(frame, tiers, state, elapsed)
                    did_work = True
                    break  # Favor checking progress/stop between allocations.
                if not did_work:
                    self._stop.wait(self.poll_s)
        except BaseException as e:
            with self._lock:
                self._error = e
