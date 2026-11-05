"""Audit integrated transport artifacts and summarize observed results."""

import argparse
import json
from pathlib import Path
import re
import statistics


def stage_latencies(rows, priority):
    files = {}
    for row in rows:
        if row["priority"] == priority:
            value = row["received_ms"] - row["release_ms"]
            files[row["path"]] = max(files.get(row["path"], 0), value)
    return sorted(files.values())


def analyze(root):
    lines = [
        "Q3DGStream integrated priority transport",
        "Synthetic QNP01 records, not trained NTC weights.",
        "Late-arrival stress across 32 frame IDs. This workload does not simulate playable recurrent frames.",
        "All policies use the new transport. The absent legacy transport is not a measured baseline.",
        "P0 CHUNK p95 receipt latency: median of three runs (milliseconds).",
        "",
        "Profile          FIFO    Strict  Weighted   Improvement   P0 first-send rank FIFO/strict",
    ]
    stage_lines = ["", "P0 COMPLETE STAGE p95 latency (ms): FIFO / strict / weighted"]
    total_files = total_chunks = 0
    for profile in ["clean", "bandwidth", "delay", "jitter", "loss", "burst"]:
        stages = {}
        latency = {}
        ranks = {}
        for policy in ["fifo", "strict", "weighted"]:
            samples = []
            stage_samples = []
            order = []
            for seed in range(1, 4):
                folder = root / f"{profile}-{seed}-{policy}"
                s = json.loads((folder / "summary.json").read_text())
                rows = json.loads((folder / "events.json").read_text())
                assert (
                    s["files"] == s["valid_files"] == 128
                    and s["chunks"] == len(rows) == 256
                )
                assert not s["errors"] and not json.loads(
                    (folder / "receiver-errors.json").read_text()
                )
                assert all(
                    r["release_ms"]
                    <= r["selected_ms"]
                    <= r["submitted_ms"]
                    <= r["received_ms"]
                    <= r["acked_ms"]
                    for r in rows
                )
                pending = {(r["path"], r["offset"]): r for r in rows}
                for row in sorted(rows, key=lambda r: r["submitted_ms"]):
                    ready = [
                        v
                        for v in pending.values()
                        if v["release_ms"] <= row["selected_ms"]
                    ]
                    if policy == "strict":
                        assert row["priority"] == min(r["priority"] for r in ready)
                    del pending[(row["path"], row["offset"])]
                qlogs = list((folder / "qlog").glob("*.qlog"))
                assert len(qlogs) == 1
                trace = json.loads(qlogs[0].read_text())["traces"][0]
                assert trace["vantage_point"]["type"] == "client"
                first = []
                seen = set()
                finished = set()
                for e in trace["events"]:
                    if e["name"] != "transport:packet_sent":
                        continue
                    for frame in e["data"].get("frames", []):
                        if frame["frame_type"] != "stream":
                            continue
                        sid = frame["stream_id"]
                        if sid not in seen:
                            first.append(sid)
                            seen.add(sid)
                        if frame.get("fin"):
                            finished.add(sid)
                assert seen == finished == {r["stream_id"] for r in rows}
                priorities = {r["stream_id"]: r["priority"] for r in rows}
                order.append(
                    statistics.median(
                        i + 1 for i, sid in enumerate(first) if priorities[sid] == 0
                    )
                )
                if profile in ("loss", "burst"):
                    assert (
                        sum(
                            map(
                                int,
                                re.findall(
                                    r"dropped (\d+)", (folder / "netem.txt").read_text()
                                ),
                            )
                        )
                        > 0
                    )
                samples.append(s["classes"]["0"]["latency_p95_ms"])
                waits = stage_latencies(rows, 0)
                stage_samples.append(waits[int(0.95 * (len(waits) - 1))])
                total_files += s["files"]
                total_chunks += len(rows)
            stages[policy] = statistics.median(stage_samples)
            latency[policy] = statistics.median(samples)
            ranks[policy] = statistics.median(order)
        improvement = 100 * (1 - latency["strict"] / latency["fifo"])
        stage_lines.append(
            f"{profile:12} {stages['fifo']:.1f} / {stages['strict']:.1f} / {stages['weighted']:.1f}"
        )
        lines.append(
            f"{profile:12} {latency['fifo']:8.1f} {latency['strict']:8.1f} {latency['weighted']:8.1f} {improvement:9.1f}% {ranks['fifo']:14.1f}/{ranks['strict']:.1f}"
        )
    lines += stage_lines
    pipeline = json.loads((root / "pipeline/summary.json").read_text())
    assert pipeline["valid_files"] == 8 and not pipeline["errors"]
    for profile in ["clean", "bandwidth", "delay", "jitter", "loss", "burst"]:
        summary = json.loads(
            (root / f"pipeline-{profile}" / "summary.json").read_text()
        )
        assert (
            summary["valid_files"] == 8
            and summary["chunks"] == 21
            and not summary["errors"]
        )
    lines += [
        "",
        f"PASS: 54 stress runs, {total_files} QNP01 files, {total_chunks} chunks.",
        "All file hashes, upstream QNP01 parsing and dense INT4 decode comparisons passed.",
        "PASS: real QUIC 2-frame adapter pipeline on all six profiles, additions/stage/frame barriers.",
        "First-send rank is the order of first STREAM-frame appearance in sender qlog.",
        "It verifies endpoint packet construction order, not NIC departure or WAN arrival ordering.",
        "",
        "Limitations: synthetic weights and empty Gaussian fixtures; no GPU rendering;",
        "same-host clocks; three stochastic repeats without matched loss traces;",
        "application admission scheduling only; reliable delivery without deadline cancellation;",
        "16-chunk ACK window can limit throughput; strict can starve low-priority queues;",
        "new ALPN requires the new headless receiver; legacy viewer compatibility unverified.",
        "Network stress allows independent file arrivals across frame IDs. Production adapter",
        "preserves per-frame barriers. Its current 8-chunk NTC batch fits entirely inside",
        "the 16-chunk window, so its pipeline test does not demonstrate a priority latency gain.",
        "Positive, neutral and negative performance differences are reported without filtering.",
        "",
        "Sources: benchmark.py, matrix.py, events.json, qlog/, netem.txt in this run.",
        "https://www.rfc-editor.org/rfc/rfc9000.html#section-2.3",
        "https://aioquic.readthedocs.io/en/latest/asyncio.html",
        "https://man7.org/linux/man-pages/man8/tc-netem.8.html",
    ]
    report = "\n".join(lines) + "\n"
    (root / "REPORT.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/outputs/priority-matrix"))
    analyze(p.parse_args().root)
