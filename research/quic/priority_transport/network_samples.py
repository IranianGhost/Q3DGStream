"""Sample the egress bottleneck in its temporary network namespace."""

import json
import subprocess
import time


SAMPLE_INTERVAL = .2


def queue_sample(namespace, device):
    start = time.time_ns()
    raw = subprocess.check_output(
        ["sudo", "-n", "ip", "netns", "exec", namespace,
         "tc", "-s", "-j", "qdisc", "show", "dev", device], timeout=3)
    finish = time.time_ns()
    (netem,) = [qdisc for qdisc in json.loads(raw) if qdisc["kind"] == "netem"]
    return {
        "unix_ms": (start + finish) / 2e6,
        "sample_duration_ms": (finish - start) / 1e6,
        "dequeued_bytes": netem["bytes"],
        "dequeued_packets": netem["packets"],
        "backlog_bytes": netem["backlog"],
        "backlog_packets": netem["qlen"],
        "drops": netem["drops"],
    }


def sample_bottleneck(network, stop, samples, errors):
    try:
        next_sample = time.monotonic()
        while True:
            samples.append(queue_sample(network["client_ns"], network["client_dev"]))
            if stop.is_set():
                return
            next_sample += SAMPLE_INTERVAL
            if next_sample < time.monotonic():
                next_sample = time.monotonic() + SAMPLE_INTERVAL
            stop.wait(max(0, next_sample - time.monotonic()))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        errors.append(str(exc))
