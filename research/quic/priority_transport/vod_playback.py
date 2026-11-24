"""Buffer-based start and pause/resume for on-demand NTC playout."""

import time


class VodClock:
    def __init__(self, request_time, fps, startup_frames, total_frames, feedback_ms):
        self.request_time = request_time
        self.period = 1 / fps
        self.startup_frames = min(startup_frames, total_frames)
        self.feedback_interval = feedback_ms / 1000
        self.phase = "startup"
        self.next_due = None
        self.first_render = None
        self.last_render = None
        self.rendered_frames = 0
        self.stalls = []

    def wait_for_buffer(self, ready_frames, minimum, report):
        next_report = 0.0
        while ready_frames() < minimum:
            now = time.monotonic()
            if now >= next_report:
                report()
                next_report = now + self.feedback_interval
            time.sleep(0.003)

    def start(self, ready_frames, report):
        self.wait_for_buffer(ready_frames, self.startup_frames, report)
        self.next_due = time.monotonic()
        self.phase = "playing"
        report()

    def wait_for_update(self, update, ready_frames, report):
        if ready_frames():
            return
        stalled_at = time.monotonic()
        self.phase = "rebuffering"
        self.next_due = None
        self.wait_for_buffer(ready_frames, 1, report)
        resumed_at = time.monotonic()
        self.stalls.append({
            "ntc": update,
            "start_ms_from_request": round((stalled_at - self.request_time) * 1000, 3),
            "duration_ms": round((resumed_at - stalled_at) * 1000, 3),
        })
        self.phase = "playing"
        self.next_due = resumed_at
        report()

    def advance(self, rendered_at):
        if self.first_render is None:
            self.first_render = rendered_at
        self.last_render = rendered_at
        self.rendered_frames += 1
        self.next_due = max(self.next_due + self.period, rendered_at)

    def summary(self):
        total = sum(stall["duration_ms"] for stall in self.stalls)
        duration = self.last_render - self.first_render
        unpaused = duration - total / 1000
        return {
            "startup_buffer_frames": self.startup_frames,
            "startup_delay_ms": round((self.first_render - self.request_time) * 1000, 3),
            "rebuffering_count": len(self.stalls),
            "rebuffering_ms": round(total, 3),
            "longest_stall_ms": max((stall["duration_ms"] for stall in self.stalls), default=0.0),
            "render_fps_excluding_rebuffering": round((self.rendered_frames - 1) / unpaused, 3)
            if unpaused > 0 else None,
        }
