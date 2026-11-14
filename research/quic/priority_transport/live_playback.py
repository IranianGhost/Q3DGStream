"""Opt-in clock and feedback for frames presented by the native viewer."""
import json
import math
from pathlib import Path


class LivePlayback:
    def __init__(self, updates, fps, prebuffer):
        if updates < 1 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("positive update count and finite positive fps required")
        if not math.isfinite(prebuffer) or prebuffer < 0:
            raise ValueError("finite nonnegative prebuffer required")
        self.updates = updates
        self.interval = 1 / fps
        self.minimum = min(updates, max(1, math.ceil(prebuffer * fps)))
        self.frame = None
        self.first_presented = None
        self.last_presented = None
        self.stall_start = None
        self.stalls = []
        self.ready_at = {}
        self.frames = []
        self.finished = False

    def select(self, now, scene_ready, ready):
        for frame in range(1, min(ready, self.updates) + 1):
            self.ready_at.setdefault(frame, now)
        if self.frame is None:
            return 0 if scene_ready and ready >= self.minimum else None
        if self.finished or now < self.last_presented + self.interval:
            return self.frame
        next_frame = self.frame + 1
        if next_frame > ready:
            if self.stall_start is None:
                self.stall_start = self.last_presented + self.interval
            return self.frame
        if self.stall_start is not None:
            if ready - self.frame < min(self.minimum, self.updates - self.frame):
                return self.frame
        return next_frame

    def present(self, now, frame):
        if frame is None or frame == self.frame:
            return
        if frame != (0 if self.frame is None else self.frame + 1):
            raise ValueError("presentation must advance contiguously")
        if self.stall_start is not None:
            self.stalls.append((self.stall_start, now))
            self.stall_start = None
        if self.first_presented is None:
            self.first_presented = now
        self.frame = frame
        self.last_presented = now
        self.frames.append(now)
        self.finished = frame == self.updates

    def report(self, started, now):
        stalls = list(self.stalls)
        if self.stall_start is not None:
            stalls.append((self.stall_start, now))
        late = None if self.first_presented is None else sum(
            self.ready_at.get(i, math.inf) > self.first_presented + i * self.interval
            for i in range(1, self.updates + 1)
        )
        return dict(
            frames_presented=len(self.frames),
            planned_frames=self.updates + 1,
            finished=self.finished,
            startup_seconds=None if self.first_presented is None else self.first_presented - started,
            elapsed_seconds=now - started,
            stall_count=len(stalls),
            stall_seconds=sum(end - start for start, end in stalls),
            stalls_seconds=[[start - started, end - started] for start, end in stalls],
            late_or_missing_to_initial_schedule=late,
            presentations_seconds=[t - started for t in self.frames],
            readiness_seconds={i: t - started for i, t in self.ready_at.items()},
        )


def choose_optional_bytes(snapshot, now, required_bytes, interval, target_buffer, maximum):
    """Admit optional bytes only when fresh viewer feedback leaves spare capacity."""
    if not snapshot or not 0 <= now - snapshot["at"] <= 1:
        return 0
    if snapshot["buffer_seconds"] < target_buffer:
        return 0
    budget = int(snapshot["committed_Bps"] * interval * 0.8) - required_bytes
    return max(0, min(maximum, budget))


def write_snapshot(path, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state))
    temporary.replace(path)
