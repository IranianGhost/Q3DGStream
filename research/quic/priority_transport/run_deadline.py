"""Run QUIC deadline-aware PQAT NTC streaming, optionally with live CUDA playback.

--live starts the GPU viewer BEFORE the QUIC connection, consumes the actual
received groups at 30 FPS, and feeds measured CUDA/GL timings back to Quinn.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import nullcontext

from net_link import isolated_link, parse_rate_profile, apply_rate_profile
from run_ntc import DEFAULT_WIRE_DIR, pump, within_namespace
from run_sequence import prepare_all, paths_for, DEFAULT_PRIORITY_DIR

HERE = Path(__file__).resolve().parent
DECISION = re.compile(
    r"^DECISION frame=(\d+) tiers=(\d+) bytes=(\d+) capacity_mbps=([\d.]+) "
    r"measured_goodput_mbps=([\d.]+) cwnd=(\d+) rtt_ms=([\d.]+) "
    r"min_rtt_ms=([\d.]+) jitter_sample_ms=([\d.]+) "
    r"backlog_ms=([\d.]+) client_ms=([\d.]+) gpu_source=(\S+) "
    r"deadline_ms=([\d.]+) loss_delta=(\d+) sent_delta=(\d+) "
    r"loss_fraction=([\d.]+) congestion_delta=(\d+)$"
)
FRAME = re.compile(
    r"^FRAME_COMPLETE (\d+) tiers=(\d+) t_ms=([\d.]+) "
    r"deadline_ms=([\d.]+) late=(true|false)$"
)
GROUP = re.compile(r"^GROUP_COMPLETE (\d+) tier=([0-3]) t_ms=([\d.]+)$")
NET_READY = re.compile(r"^GROUP_NET_READY (\d+) tier=([0-3]) t_ms=([\d.]+)$")
WRITE_BEGIN = re.compile(r"^GROUP_WRITE_BEGIN (\d+) tier=([0-3]) t_ms=([\d.]+)$")
WRITE_END = re.compile(r"^GROUP_WRITE_END (\d+) tier=([0-3]) t_ms=([\d.]+)$")
QUINN_PRIORITY = re.compile(r"^QUINN_PRIORITY frame=(\d+) tier=([0-3]) value=(-?\d+)$")
PREDICT_CONTEXT = re.compile(
    r"^PREDICT_CONTEXT frame=(\d+) send_ms=([\d.]+) outstanding_bytes=(\d+) "
    r"stage_bytes=(\d+),(\d+),(\d+),(\d+) initial_mbps=([\d.]+) utilization=([\d.]+) "
    r"decode_ms=([\d.]+) inference_ms=([\d.]+) render_ms=([\d.]+) "
    r"buffer_frames=(\d+) last_applied_ntc=(-?\d+)$"
)
SENDER_EPOCH = re.compile(r"^SENDER_EPOCH unix_ms=([\d.]+)$")

SCHEDULING_MODES = (
    "adaptive", "sequential_equal", "concurrent_equal",
    "concurrent_priority", "p0_first_priority", "sequential_priority",
    "sequential", "concurrent", "p0_first",
    "original", "amirali_fifo", "amirali_strict", "amirali_weighted",
)


def verify_received(output, manifest, chosen):
    """Verify every transported actual group against its exact input hash."""
    frames = {}
    for frame, selected in chosen.items():
        if frame not in manifest:
            raise RuntimeError(f"Unknown frame {frame}")
        received_dir = output / "priority" / "received" / f"NTC_{frame:06}"
        for priority in range(selected):
            file = received_dir / f"P{priority}.bin"
            actual = hashlib.sha256(file.read_bytes()).hexdigest()
            expected = manifest[frame]["groups"][f"P{priority}"]["sha256"]
            if actual != expected:
                raise RuntimeError(f"Checksum mismatch for NTC {frame} P{priority}")
        if (files := list(received_dir.glob("P*.bin"))) and len(files) != selected:
            raise RuntimeError(f"Unexpected group count NTC {frame}: {len(files)} != {selected}")
        frames[frame] = selected
    return frames


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        if not rows:
            raise RuntimeError(f"No records for {path.name}")
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if args.frames < 1 or args.start_frame < 0 or args.timeout <= 0:
        raise ValueError("Invalid frame count, start, or timeout")
    if args.live and args.start_frame != 0:
        raise ValueError("--live requires start-frame 0 (cumulative history is necessary)")
    if args.live and args.metrics_json:
        raise ValueError("--metrics-json cannot be combined with --live (the viewer supplies it)")
    if args.progressive and not args.live:
        raise ValueError("--progressive requires --live")
    profile = parse_rate_profile(args.rate_profile)
    if profile and not args.netns:
        raise ValueError("--rate-profile requires --netns for isolated tc changes")
    selection = paths_for(args)
    output = (Path(args.out).expanduser().resolve() if args.out else
              HERE / f"results_deadline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}")
    output.mkdir(parents=True, exist_ok=False)
    manifests, total_bytes = prepare_all(selection, output)
    p0_mbps = sum(m["groups"]["P0"]["payload_bytes"] for m in manifests.values()) * (
        args.fps * 8 / args.frames / 1e6)
    full_mbps = total_bytes * args.fps * 8 / args.frames / 1e6
    print(f"INPUT {args.frames} NTCs, {total_bytes/1e6:.3f} MB; "
          f"required at {args.fps:g}FPS: P0={p0_mbps:.3f}Mbps full={full_mbps:.3f}Mbps "
          "(excluding QUIC/UDP/IP overhead)", flush=True)
    if args.prepare_only:
        print(f"PREPARED {output}"); return
    if not shutil.which("cargo"):
        raise RuntimeError("Cargo unavailable. Build/run on your Ubuntu laptop.")
    subprocess.run(["cargo", "build", "--release"], cwd=HERE, check=True)
    binary = HERE / "target" / "release" / "quinn-ntc-deadline"
    trial = output / "priority"
    trial.mkdir()
    received = trial / "received"
    received.mkdir()
    cert = trial / "cert.der"
    live_folder = trial / "live"
    metrics_file = trial / "live_metrics.json"
    session_clock = trial / "connection_clock.json"
    initial_link_rate = profile[0][1] if profile else args.rate_mbps
    initial_estimate = (args.initial_mbps if args.initial_mbps is not None else
                        max(mbps for _, mbps in profile) if profile else args.rate_mbps)
    link = (isolated_link(initial_link_rate, args.delay_ms, args.loss_pct) if args.netns
            else nullcontext({"host": "127.0.0.1"}))
    gpu_proc = None
    with link as network:
        common = [
            "--host", network["host"], "--port", str(args.port),
            "--start-frame", str(args.start_frame), "--frames", str(args.frames),
            "--fps", str(args.fps), "--feedback-ms", str(args.feedback_ms),
            "--playout-ms", str(args.playout_ms),
            "--initial-mbps", str(initial_estimate),
            "--scheduling-mode", args.scheduling_mode,
            "--quinn-priority", args.quinn_priority,
            "--tier-selection", args.tier_selection,
            "--fixed-tiers", str(args.fixed_tiers),
            "--max-inflight", str(args.max_inflight),
            "--tier-delays-ms", args.tier_delays_ms,
            "--delivery-timeout-ms", str(args.delivery_timeout_ms),
            "--utilization", str(args.utilization),
            "--inference-ms", str(args.inference_ms),
            "--render-ms", str(args.render_ms), "--decode-ms", str(args.decode_ms),
        ]
        recv_cmd = within_namespace(
            [str(binary), "receiver", "--cert", str(cert),
             "--received-dir", str(received), *common,
             "--session-clock-file", str(session_clock),
             *(["--metrics-json", str(metrics_file)] if args.live else
               (["--metrics-json", str(Path(args.metrics_json).resolve())] if args.metrics_json else []))],
            network.get("server_ns"))
        send_cmd = within_namespace(
            [str(binary), "sender", "--cert", str(cert),
             "--payload-dir", str(output / "send"), *common],
            network.get("client_ns"))
        events = queue.Queue()
        receiver = None
        sender = None
        all_logs = []
        rate_stop = threading.Event()
        rate_errors = []
        rate_thread = None
        start = time.monotonic()
        try:
            if args.live:
                live_cmd = [
                    sys.executable, str(HERE / "live_client.py"),
                    "--out", str(live_folder), "--received-dir", str(received),
                    "--manifest", str(output / "sequence_manifest.json"),
                    "--clock-file", str(session_clock), "--metrics-json", str(metrics_file),
                    "--scene-base", str(Path(args.scene_base).expanduser().resolve()),
                    "--viewer", str(Path(args.viewer).expanduser().resolve()),
                    "--wire-dir", str(Path(args.wire_dir).expanduser().resolve()),
                    "--start-frame", str(args.start_frame), "--frames", str(args.frames),
                    "--fps", str(args.fps), "--playout-ms", str(args.playout_ms),
                    "--feedback-ms", str(args.feedback_ms),
                    "--width", str(args.width), "--height", str(args.height),
                    "--capture-every", str(args.capture_every),
                    "--preparation-mode", args.preparation_mode,
                    "--prefetch-frames", str(args.prefetch_frames),
                    "--target-tiers", str(args.fixed_tiers if args.tier_selection == "fixed" else 0),
                    *(["--progressive"] if args.progressive else []),
                    *(["--show-window"] if args.show_window else []),
                    "--preview-port", str(args.preview_port),
                    "--preview-linger-seconds", str(args.preview_linger_seconds),
                    *(["--save-stages"] if args.save_stages else []),
                ]
                gpu_proc = subprocess.Popen(live_cmd, cwd=HERE, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, bufsize=1)
                threading.Thread(target=pump, args=("GPU", gpu_proc, events), daemon=True).start()
                print("[LIVE] Preloading CUDA renderer before QUIC connects...", flush=True)
                preload_start = time.monotonic()
                while not (live_folder / "ready.json").is_file():
                    if gpu_proc.poll() is not None:
                        while not events.empty(): all_logs.append(events.get_nowait())
                        raise RuntimeError(f"Live CUDA renderer failed ({gpu_proc.returncode}): {all_logs[-15:]}")
                    if time.monotonic() - preload_start > args.ready_timeout:
                        raise TimeoutError("CUDA viewer preload timeout")
                    time.sleep(0.02)
                print("[LIVE] Viewer ready; starting Quinn receiver/sender", flush=True)
            receiver = subprocess.Popen(recv_cmd, cwd=HERE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
            threading.Thread(target=pump, args=("RECEIVER", receiver, events), daemon=True).start()
            receiver_start = time.monotonic()
            while not cert.exists():
                if receiver.poll() is not None:
                    raise RuntimeError(f"receiver exited {receiver.returncode}")
                if time.monotonic() - receiver_start > 20:
                    raise TimeoutError("Receiver did not become ready")
                time.sleep(0.02)
            sender = subprocess.Popen(send_cmd, cwd=HERE, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
            threading.Thread(target=pump, args=("SENDER", sender, events), daemon=True).start()
            if len(profile) > 1:
                rate_thread = threading.Thread(
                    target=apply_rate_profile,
                    args=(profile, network, args.delay_ms, args.loss_pct,
                          rate_stop, time.monotonic(), rate_errors), daemon=True)
                rate_thread.start()
            while True:
                if rate_errors:
                    raise RuntimeError(f"Bandwidth-profile controller failed: {rate_errors}")
                if time.monotonic() - start > args.timeout:
                    raise TimeoutError(f"Deadline test exceeded {args.timeout}s")
                try:
                    who, line = events.get(timeout=0.1)
                except queue.Empty:
                    who, line = None, None
                if line and line != "__EXIT__":
                    all_logs.append((who, line))
                    if line.startswith(("SENDER_FINISH", "ALL_COMPLETE", "DEADLINE_INFEASIBLE")):
                        print(f"[{who}] {line}", flush=True)
                    elif who == "GPU" and line.startswith(("[LIVE]", "[LIVE DONE]", "[GPU READY]", "[PREVIEW]")):
                        print(line, flush=True)
                    elif line.startswith("FRAME_COMPLETE") and len(
                        [e for e in all_logs if e[1].startswith("FRAME_COMPLETE")]) % 25 == 0:
                        print(f"[{who}] {line}", flush=True)
                for proc, name in ((sender, "sender"), (receiver, "receiver")):
                    if proc.poll() is not None and proc.returncode != 0:
                        raise RuntimeError(f"{name} exited {proc.returncode}: {all_logs[-12:]}")
                if gpu_proc is not None and gpu_proc.poll() is not None and gpu_proc.returncode != 0:
                    raise RuntimeError(f"Live viewer failed {gpu_proc.returncode}: {all_logs[-20:]}")
                if sender.poll() is not None and receiver.poll() is not None:
                    break
            if gpu_proc is not None:
                # Network delivery may finish before the final playout tick.
                while gpu_proc.poll() is None:
                    if time.monotonic() - start > args.timeout:
                        raise TimeoutError("Live viewer did not complete playout")
                    try:
                        who, line = events.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if line and line != "__EXIT__":
                        all_logs.append((who, line))
                        if who == "GPU" and line.startswith(("[LIVE]", "[LIVE DONE]", "[PREVIEW]")):
                            print(line, flush=True)
                if gpu_proc.returncode != 0:
                    raise RuntimeError(f"Live viewer failed {gpu_proc.returncode}: {all_logs[-15:]}")
            while not events.empty():
                who, line = events.get_nowait()
                if line and line != "__EXIT__": all_logs.append((who, line))
        finally:
            rate_stop.set()
            if rate_thread is not None:
                rate_thread.join(timeout=2)
            for proc in (sender, receiver, gpu_proc):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try: proc.wait(timeout=3)
                    except subprocess.TimeoutExpired: proc.kill()

    decisions, arrivals, group_arrivals, group_write_starts, feedback_lines = {}, {}, {}, {}, []
    group_net_ready = {}
    group_write_ends, native_priorities = {}, {}
    predict_contexts = {}
    sender_epoch_unix_ms = None
    for who, line in all_logs:
        if who == "SENDER" and (match := SENDER_EPOCH.fullmatch(line)):
            sender_epoch_unix_ms = float(match[1])
        if who == "SENDER" and (match := PREDICT_CONTEXT.fullmatch(line)):
            frame = int(match[1])
            if frame in predict_contexts:
                raise RuntimeError(f"Duplicate sender prediction context NTC {frame}")
            predict_contexts[frame] = {
                "sender_decision_ms": float(match[2]),
                "sender_outstanding_bytes": int(match[3]),
                **{f"stage_p{p}_bytes": int(match[p+4]) for p in range(4)},
                "initial_capacity_ceiling_mbps": float(match[8]),
                "sender_utilization": float(match[9]),
                "predicted_decode_ms": float(match[10]),
                "predicted_inference_ms": float(match[11]),
                "predicted_render_ms": float(match[12]),
                "feedback_buffer_frames": int(match[13]),
                "feedback_last_applied_ntc": int(match[14]),
            }
        if who == "SENDER" and (match := DECISION.match(line)):
            frame = int(match[1])
            decisions[frame] = {
                "frame": frame, "selected_tiers": int(match[2]),
                "payload_bytes": int(match[3]), "available_mbps": float(match[4]),
                "client_goodput_mbps": float(match[5]), "cwnd_bytes": int(match[6]),
                "rtt_ms": float(match[7]), "min_rtt_ms": float(match[8]),
                "observed_rtt_jitter_sample_ms": float(match[9]),
                "backlog_ms": float(match[10]), "client_compute_ms": float(match[11]),
                "gpu_source": match[12], "deadline_ms": float(match[13]),
                "loss_delta_packets": int(match[14]),
                "sent_delta_packets": int(match[15]),
                "loss_fraction_in_observation_window": float(match[16]),
                "congestion_delta_events": int(match[17]),
            }
        if who == "RECEIVER" and (match := FRAME.match(line)):
            arrivals[int(match[1])] = {
                "frame": int(match[1]), "received_tiers": int(match[2]),
                "completion_ms": float(match[3]), "deadline_ms_receiver": float(match[4]),
                "deadline_missed": match[5] == "true",
            }
        if who == "RECEIVER" and (match := GROUP.fullmatch(line)):
            key = (int(match[1]), int(match[2]))
            if key in group_arrivals:
                raise RuntimeError(f"Duplicate group arrival {key}")
            group_arrivals[key] = float(match[3])
        if who == "RECEIVER" and (match := NET_READY.fullmatch(line)):
            key = (int(match[1]), int(match[2]))
            if key in group_net_ready:
                raise RuntimeError(f"Duplicate network-ready group {key}")
            group_net_ready[key] = float(match[3])
        if who == "SENDER" and (match := WRITE_BEGIN.fullmatch(line)):
            key = (int(match[1]), int(match[2]))
            if key in group_write_starts:
                raise RuntimeError(f"Duplicate group write start {key}")
            group_write_starts[key] = float(match[3])
        if who == "SENDER" and (match := WRITE_END.fullmatch(line)):
            key = (int(match[1]), int(match[2]))
            if key in group_write_ends:
                raise RuntimeError(f"Duplicate group write end {key}")
            group_write_ends[key] = float(match[3])
        if who == "SENDER" and (match := QUINN_PRIORITY.fullmatch(line)):
            key = (int(match[1]), int(match[2]))
            if key in native_priorities:
                raise RuntimeError(f"Duplicate Quinn priority record {key}")
            native_priorities[key] = int(match[3])
        if who == "RECEIVER" and line.startswith("CLIENT_FEEDBACK"):
            feedback_lines.append(line)
    if len(decisions) != args.frames or len(arrivals) != args.frames:
        raise RuntimeError(f"Missing decisions/receipts: {len(decisions)}/{len(arrivals)} of {args.frames}")
    if len(predict_contexts) != args.frames or sender_epoch_unix_ms is None:
        raise RuntimeError(f"Missing sender prediction context: {len(predict_contexts)}/{args.frames}")
    if not session_clock.is_file():
        raise RuntimeError("Missing receiver connection epoch; cannot compare cross-process times")
    receiver_epoch_unix_ms = float(json.loads(session_clock.read_text())["unix_ms"])
    sender_to_receiver_ms = sender_epoch_unix_ms - receiver_epoch_unix_ms
    verified = verify_received(output, manifests,
                               {f: d["selected_tiers"] for f, d in decisions.items()})
    rows = []
    previous_completion = None
    late_run = 0
    for frame in sorted(decisions):
        if decisions[frame]["selected_tiers"] != arrivals[frame]["received_tiers"]:
            raise RuntimeError(f"Selected/received stage count disagree at NTC {frame}")
        completion = arrivals[frame]["completion_ms"]
        interval = None if previous_completion is None else completion - previous_completion
        late_run = late_run + 1 if arrivals[frame]["deadline_missed"] else 0
        count = decisions[frame]["selected_tiers"]
        missing = [tier for tier in range(count) if (frame, tier) not in group_arrivals
                   or (frame, tier) not in group_write_starts]
        if missing:
            raise RuntimeError(f"Missing per-group timing NTC {frame}: {missing}")
        missing_net = [tier for tier in range(count) if (frame, tier) not in group_net_ready]
        if missing_net:
            raise RuntimeError(f"Missing network-ready group timing NTC {frame}: {missing_net}")
        for tier in range(count):
            if group_net_ready[(frame, tier)] > group_arrivals[(frame, tier)]:
                raise RuntimeError(f"Network-ready timestamp after disk commit: {frame} P{tier}")
        if args.quinn_priority in ("equal", "tiered"):
            for tier in range(count):
                key = (frame, tier)
                if key not in native_priorities:
                    raise RuntimeError(f"Missing Quinn priority frame={frame} P{tier}")
                value = ((3 - tier) * 10) if args.quinn_priority == "tiered" else 0
                if native_priorities[key] != value:
                    raise RuntimeError(f"Wrong Quinn priority frame={frame} P{tier}: "
                                       f"{native_priorities[key]} != {value}")
        if args.scheduling_mode in ("concurrent_equal", "concurrent_priority", "p0_first_priority",
                                    "sequential", "concurrent", "p0_first"):
            for tier in range(count):
                key = (frame, tier)
                if key not in group_write_ends:
                    raise RuntimeError(f"Missing group write completion frame={frame} P{tier}")
                if group_write_ends[key] < group_write_starts[key]:
                    raise RuntimeError(f"Negative Quinn write interval frame={frame} P{tier}")
            if args.scheduling_mode in ("p0_first_priority", "p0_first") and count > 1:
                if not all(group_write_starts[(frame, tier)] >= group_write_ends[(frame, 0)]
                           for tier in range(1, count)):
                    raise RuntimeError(f"P0-first ordering was not enforced for frame={frame}")
        spread = (max(group_write_starts[(frame, tier)] for tier in range(count)) -
                  min(group_write_starts[(frame, tier)] for tier in range(count)))
        net_order = sorted(range(count), key=lambda tier: (group_net_ready[(frame, tier)], tier))
        commit_order = sorted(range(count), key=lambda tier: (group_arrivals[(frame, tier)], tier))
        rows.append({
            **decisions[frame], **arrivals[frame], **predict_contexts[frame],
            "sender_to_receiver_clock_offset_ms": f"{sender_to_receiver_ms:.3f}",
            "scheduling_mode": args.scheduling_mode,
            "quinn_priority_policy": args.quinn_priority,
            "net_ready_order": ">".join(f"P{tier}" for tier in net_order),
            "disk_ready_order": ">".join(f"P{tier}" for tier in commit_order),
            "p0_net_ready_rank": net_order.index(0) + 1,
            "net_ready_spread_ms": f"{max(group_net_ready[(frame, tier)] for tier in range(count)) - min(group_net_ready[(frame, tier)] for tier in range(count)):.3f}",
            **{f"p{tier}_net_ready_ms_connection":
               f"{group_net_ready[(frame, tier)]:.3f}" if tier < count else ""
               for tier in range(4)},
            **{f"p{tier}_net_to_disk_ms":
               f"{group_arrivals[(frame, tier)] - group_net_ready[(frame, tier)]:.3f}"
               if tier < count else "" for tier in range(4)},
            **{f"p{tier}_complete_ms_connection":
               f"{group_arrivals[(frame, tier)]:.3f}" if tier < count else ""
               for tier in range(4)},
            **{f"p{tier}_write_start_ms_sender":
               f"{group_write_starts[(frame, tier)]:.3f}" if tier < count else ""
               for tier in range(4)},
            **{f"p{tier}_write_end_ms_sender":
               f"{group_write_ends[(frame, tier)]:.3f}" if (frame, tier) in group_write_ends else ""
               for tier in range(4)},
            **{f"p{tier}_quinn_priority":
               native_priorities.get((frame, tier), "") for tier in range(4)},
            **{f"p{tier}_delivery_after_write_end_ms":
               f"{group_arrivals[(frame, tier)] - group_write_ends[(frame, tier)] - sender_to_receiver_ms:.3f}"
               if (frame, tier) in group_write_ends else ""
               for tier in range(4)},
            "write_start_spread_ms": f"{spread:.3f}",
            "p0_lead_before_full_ms": f"{completion - group_arrivals[(frame, 0)]:.3f}",
            "p0_deadline_missed": group_arrivals[(frame, 0)] > arrivals[frame]["deadline_ms_receiver"],
            "arrival_interval_ms": "" if interval is None else f"{interval:.3f}",
            "arrival_jitter_abs_ms": "" if interval is None else f"{abs(interval - 1000.0 / args.fps):.3f}",
            "consecutive_deadline_miss_burst": late_run,
            "sha256_group_verified": "yes",
        })
        previous_completion = completion
    write_csv(trial / "per_frame_deadline.csv", rows)
    (trial / "feedback.log").write_text("\n".join(feedback_lines) + "\n")
    (trial / "transport.log").write_text("\n".join(f"[{who}] {line}" for who, line in all_logs) + "\n")
    summary = {
        "frames": args.frames, "fps_target": args.fps,
        "scheduling_mode": args.scheduling_mode,
        "quinn_priority_policy": args.quinn_priority,
        "tier_selection": args.tier_selection, "fixed_tiers": args.fixed_tiers,
        "rate_profile": args.rate_profile,
        "concurrent_write_start_mean_spread_ms": sum(float(r["write_start_spread_ms"]) for r in rows) / len(rows),
        "p0_deadline_missed_transport": sum(row["p0_deadline_missed"] for row in rows),
        "selected_tier_counts": {f"P0..P{j-1}": sum(v == j for v in verified.values())
                                 for j in range(1, 5)},
        "deadline_missed_transport": sum(row["deadline_missed"] for row in rows),
        "mean_usable_mbps_estimate": sum(r["available_mbps"] for r in rows) / len(rows),
        "payload_bytes_verified": sum(r["payload_bytes"] for r in rows),
        "source": "actual QUIC transport; deadlines are connection-relative synthetic playout",
        "critical_limitations": [
            "No live 3DGS GPU rendering or measured QoE during streaming",
            "GPU times are placeholders unless externally measured JSON is provided",
            "Sender never drops cumulative NTCs; late P0 updates are reported",
            "Initial capacity is configured; cwnd/RTT bound the dynamic estimate",
        ],
    }
    if args.scheduling_mode != "adaptive" and args.tier_selection == "full":
        summary["critical_limitations"] = [
            "All P0-P3 groups forced for every NTC; deadline-aware tier selection disabled",
            "These results compare application admission and Quinn stream priority, not adaptive QoE",
            "Write completion is not packet transmission; packet-level scheduling requires Quinn tracing",
            "Concurrent submission synchronizes application writers, not simultaneous UDP packets",
            "No live CUDA playback; deadlines are transport-only synthetic playout targets",
        ]
    elif args.scheduling_mode != "adaptive" and args.tier_selection == "fixed":
        summary["critical_limitations"] = [
            "Tier count is fixed for this scenario; live feedback does not choose the payload",
            "All writing modes transmit the same number of tiers for every NTC",
            "Playback or visual fidelity must be measured with --live and exact FP32 references",
        ]
    elif args.scheduling_mode != "adaptive" and args.tier_selection == "adaptive":
        summary["critical_limitations"] = [
            "Feedback-driven tier choices can differ across scheduling modes",
            "The built-in conservative bandwidth estimate starts from configured initial capacity",
            "No stale enhancement-stream cancellation in this prototype",
        ]
    (trial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.live:
        summary["live_gpu"] = json.loads((live_folder / "playback_summary.json").read_text())
        summary["source"] = "Actual QUIC delivery + live CUDA/OpenGL playback"
        summary["critical_limitations"] = [
            "Real rendered quality requires PNG capture and separate FP32 comparison",
            "Only local shared-host namespace emulation tested; clocks assume same host",
            "Cumulative NTC backlog is applied in order, which may cause GPU tick overruns",
            "Post-deadline enhancement tiers are not retroactively applied",
        ]
        (trial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"VERIFIED {len(verified)}/{args.frames} selected real INT4 group payloads SHA256=OK")
    print(f"TIERS {summary['selected_tier_counts']} missed transport deadlines={summary['deadline_missed_transport']}")
    print(f"CSV {trial / 'per_frame_deadline.csv'}")
    print(f"RESULTS {trial}")
    if args.live:
        print(f"LIVE CSV {live_folder / 'playback_frames.csv'}")
        print(f"LIVE SUMMARY {live_folder / 'playback_summary.json'}")
        print("NOTE: Real GPU playback measured; run evaluate_live.py separately for image quality.")
    else:
        print("NOTE: Transport-only run; no real-time GPU playback or render PSNR.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--frames", type=int, default=299)
    p.add_argument("--wire-dir", default=str(DEFAULT_WIRE_DIR))
    p.add_argument("--priority-dir", default=str(DEFAULT_PRIORITY_DIR))
    p.add_argument("--out")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--netns", action="store_true")
    p.add_argument("--rate-mbps", type=float, default=5)
    p.add_argument("--delay-ms", type=float, default=10)
    p.add_argument("--loss-pct", type=float, default=1)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--initial-mbps", type=float,
                   help="Optional initial capacity ceiling; profile runs default to maximum profile rate")
    p.add_argument("--scheduling-mode", choices=SCHEDULING_MODES, default="adaptive",
                   help="Adaptive, original stage-by-stage, Quinn writes, or Amirali chunk admission")
    p.add_argument("--quinn-priority", choices=("equal", "tiered", "legacy"), default="legacy",
                   help="Native Quinn priority independent of application admission; matrix uses equal/tiered")
    p.add_argument("--tier-selection", choices=("full", "fixed", "adaptive"), default="full",
                   help="All tiers, a shared fixed tier count, or feedback-driven tier selection")
    p.add_argument("--fixed-tiers", type=int, choices=range(1, 5), default=4)
    p.add_argument("--max-inflight", type=int, choices=range(1, 129), default=16,
                   help="Maximum in-flight chunks for Amirali's application-level policies")
    p.add_argument("--rate-profile", default="",
                   help="netns only: comma-separated elapsed_seconds:Mbps, e.g. 0:20,3:5,6:10")
    p.add_argument("--delivery-timeout-ms", type=int, default=30000,
                   help="Max wait for all application-level group receipts (default: 30000)")
    p.add_argument("--feedback-ms", type=int, default=100)
    p.add_argument("--playout-ms", type=float, default=150)
    p.add_argument("--utilization", type=float, default=0.92)
    p.add_argument("--inference-ms", type=float, default=8,
                   help="Fallback ONLY; replace with measured GPU ms via --metrics-json")
    p.add_argument("--render-ms", type=float, default=8)
    p.add_argument("--decode-ms", type=float, default=1)
    p.add_argument("--metrics-json", help="Receiver-visible JSON periodically updated by GPU viewer")
    p.add_argument("--live", action="store_true", help="Run actual cumulative CUDA viewer during QUIC delivery")
    p.add_argument("--progressive", action="store_true",
                   help="Show P0 first, then render later priority groups during the same playout tick")
    p.add_argument("--show-window", action="store_true",
                   help="Open the browser preview while the GPU renders progressively")
    p.add_argument("--preview-port", type=int, default=8765)
    p.add_argument("--preview-linger-seconds", type=float, default=10)
    p.add_argument("--save-stages", action="store_true",
                   help="Save a PNG for each P0-P3 render instead of viewing only")
    p.add_argument("--tier-delays-ms", default="0,0,0,0",
                   help="Sender release offsets for P0,P1,P2,P3 in milliseconds")
    p.add_argument("--preparation-mode", choices=("none", "cpu", "dense", "async"), default="none",
                   help="GPU playback preparation, only meaningful with --live")
    p.add_argument("--prefetch-frames", type=int, default=4)
    p.add_argument("--scene-base", default=str(Path.home() / "qntc_scenes/flame_steak_official"))
    p.add_argument("--viewer", default=str(Path.home() / "QNTC_Linux_Package/3DGStreamViewer"))
    p.add_argument("--width", type=int, default=1345)
    p.add_argument("--height", type=int, default=1008)
    p.add_argument("--capture-every", type=int, default=0,
                   help="Save every Nth frame for later QoE (0=disabled; adds GL readback cost)")
    p.add_argument("--ready-timeout", type=float, default=120)
    p.add_argument("--port", type=int, default=55443)
    p.add_argument("--timeout", type=float, default=600)
    a = p.parse_args()
    try:
        run(a)
    except (RuntimeError, ValueError, FileNotFoundError, FileExistsError, KeyError,
            OSError, TimeoutError, subprocess.CalledProcessError) as e:
        p.exit(1, f"ERROR: {e}\n")
