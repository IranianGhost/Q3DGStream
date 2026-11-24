"""Stream 299 real PQAT INT4 NTC updates across ONE persistent Quinn connection.

Uses the existing saliency-ranked 40-byte retained-weight records and native
Quinn per-stream priorities. Each frame is completed/ACKed before the next is
started, bounding memory and preventing stale NTCs from accumulating.
This is sequential network transport, not real-time 30-FPS viewer playback.
"""

import argparse
from collections import Counter
from contextlib import nullcontext, redirect_stdout
import csv
from datetime import datetime
import io
import json
import os
from pathlib import Path
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

from block_codec import prepare_ranked, reconstruct_ranked
from net_link import isolated_link
from run_ntc import DEFAULT_WIRE_DIR, pump, within_namespace

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from q3dg_paths import PRIORITIES_DIR as DEFAULT_PRIORITY_DIR
TIMING_RE = re.compile(r"^TIMING (\d+) ([0-3]) (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)$")
COMPLETE_RE = re.compile(r"^FRAME_COMPLETE (\d+) (\d+(?:\.\d+)?)$")


def paths_for(args):
    wires = Path(args.wire_dir).expanduser().resolve()
    orders = Path(args.priority_dir).expanduser().resolve()
    if not wires.is_dir() or not orders.is_dir():
        raise FileNotFoundError(f"Missing wire/priority directory: {wires}, {orders}")
    selection = []
    for index in range(args.start_frame, args.start_frame + args.frames):
        wire = wires / f"NTC_{index:06}.pth"
        order = orders / f"NTC_{index:06}.npz"
        selection.append((index, wire, order))
    missing = [(i, str(w), str(p)) for i, w, p in selection if not w.is_file() or not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} NTC/priority pairs; examples: {missing[:5]}")
    return selection


def prepare_all(selection, out):
    (out / "send").mkdir()
    manifests = {}
    total_bytes = 0
    for step, (index, wire, order) in enumerate(selection, 1):
        frame_dir = out / "send" / f"NTC_{index:06}"
        frame_dir.mkdir()
        # prepare_ranked writes its four streams in a nested 'send' directory.
        with redirect_stdout(io.StringIO()):
            manifest = prepare_ranked(wire, order, frame_dir)
        if manifest["frame"] != index or not manifest["score_sequence_nonincreasing"]:
            raise RuntimeError(f"Invalid sorted saliency metadata for NTC {index}")
        source_dir = frame_dir / "send"
        for p in range(4):
            (source_dir / f"P{p}.bin").replace(frame_dir / f"P{p}.bin")
        source_dir.rmdir()
        manifests[index] = manifest
        total_bytes += manifest["payload_bytes_total"]
        if step % 25 == 0 or step == len(selection):
            print(f"PREPARED {step}/{len(selection)} NTCs ({total_bytes / 1e6:.2f} MB payload)", flush=True)
    (out / "sequence_manifest.json").write_text(json.dumps({
        "frames": len(selection), "start_frame": selection[0][0],
        "payload_bytes_total": total_bytes,
        "source_format": "PQAT INT4 5% retained saliency-ranked 64-value blocks",
        "manifests": {str(k): v for k, v in manifests.items()},
    }, indent=2) + "\n")
    return manifests, total_bytes


def run_transport(args, output, network):
    binary = HERE / "target" / "release" / "quinn-ntc-demo"
    folder = output / args.mode
    folder.mkdir()
    recv_dir = folder / "received"
    recv_dir.mkdir()
    certificate = folder / "cert.der"
    receiver_cmd = within_namespace([
        str(binary), "receiver", "--host", network["host"], "--port", str(args.port),
        "--cert", str(certificate), "--received-dir", str(recv_dir),
        "--start-frame", str(args.start_frame), "--frames", str(args.frames),
    ], network.get("server_ns"))
    sender_cmd = within_namespace([
        str(binary), "sender", "--host", network["host"], "--port", str(args.port),
        "--cert", str(certificate), "--payload-dir", str(output / "send"),
        "--start-frame", str(args.start_frame), "--frames", str(args.frames),
        "--slice-bytes", str(args.slice_bytes), "--fps", str(args.fps),
        *(["--baseline"] if args.mode == "baseline" else []),
    ], network.get("client_ns"))
    messages = queue.Queue()
    receiver = subprocess.Popen(receiver_cmd, cwd=HERE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
    threading.Thread(target=pump, args=("RECEIVER", receiver, messages), daemon=True).start()
    sender = None
    recent = []
    start = time.monotonic()
    ready = False
    timings = {}
    frames_complete = {}
    receiver_finished = False
    sender_finished = False
    try:
        while not ready:
            if receiver.poll() is not None:
                raise RuntimeError(f"Receiver exited early: {receiver.returncode}; {recent[-6:]}")
            if time.monotonic() - start > 20:
                raise TimeoutError(f"Receiver READY timeout: {recent[-6:]}")
            try:
                who, line = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            recent.append(f"{who}: {line}")
            if who == "RECEIVER" and line.startswith("READY ") and certificate.is_file():
                ready = True
        sender = subprocess.Popen(sender_cmd, cwd=HERE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=pump, args=("SENDER", sender, messages), daemon=True).start()
        while not (receiver_finished and sender_finished):
            if time.monotonic() - start > args.timeout:
                raise TimeoutError(f"Sequence timed out after {args.timeout}s; latest={recent[-8:]}")
            try:
                who, line = messages.get(timeout=0.1)
            except queue.Empty:
                who, line = "", ""
            if line and line != "__EXIT__":
                recent.append(f"{who}: {line}")
                recent = recent[-16:]
                if who == "RECEIVER":
                    timing_match = TIMING_RE.fullmatch(line)
                    frame_match = COMPLETE_RE.fullmatch(line)
                    if timing_match:
                        frame, p = int(timing_match[1]), int(timing_match[2])
                        key = (frame, p)
                        if key in timings:
                            raise RuntimeError(f"Duplicate receipt timing {key}")
                        timings[key] = (float(timing_match[3]), float(timing_match[4]))
                    elif frame_match:
                        frame = int(frame_match[1])
                        frames_complete[frame] = float(frame_match[2])
                        completed = len(frames_complete)
                        if completed % 25 == 0 or completed == args.frames:
                            print(f"RECEIVED {completed}/{args.frames}, latest NTC {frame}", flush=True)
                    elif "ERROR" in line:
                        raise RuntimeError(line)
            if receiver.poll() is not None:
                if receiver.returncode != 0:
                    raise RuntimeError(f"Receiver exited ({receiver.returncode}): {recent[-8:]}")
                receiver_finished = True
            if sender.poll() is not None:
                if sender.returncode != 0:
                    raise RuntimeError(f"Sender exited ({sender.returncode}): {recent[-8:]}")
                sender_finished = True
        # Both stdout readers were drained by the time the processes exited,
        # except possibly messages still queued: consume final TIMING lines.
        while not messages.empty():
            who, line = messages.get_nowait()
            match = TIMING_RE.fullmatch(line) if who == "RECEIVER" else None
            if match:
                frame, p = int(match[1]), int(match[2])
                timings[(frame, p)] = (float(match[3]), float(match[4]))
            match = COMPLETE_RE.fullmatch(line) if who == "RECEIVER" else None
            if match:
                frames_complete[int(match[1])] = float(match[2])
    finally:
        for process in (sender, receiver):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
    expected_frames = range(args.start_frame, args.start_frame + args.frames)
    missing_timing = [(i, p) for i in expected_frames for p in range(4) if (i, p) not in timings]
    if missing_timing or len(frames_complete) != args.frames:
        raise RuntimeError(f"Missing timings: {missing_timing[:8]}, complete {len(frames_complete)}/{args.frames}")
    return folder, timings, frames_complete


def run(args):
    if args.frames <= 0 or args.start_frame < 0 or args.fps < 0 or args.timeout <= 0:
        raise ValueError("Invalid frames, start frame, FPS or timeout")
    if args.slice_bytes <= 0 or not 1 <= args.port <= 65535:
        raise ValueError("Invalid slice size or port")
    if args.rate_mbps <= 0 or args.delay_ms < 0 or not 0 <= args.loss_pct < 100:
        raise ValueError("Invalid network parameters")
    selection = paths_for(args)  # fail before creating any output if a frame is missing
    output = (Path(args.out).expanduser().resolve() if args.out else
              HERE / f"results_seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}")
    output.mkdir(parents=True, exist_ok=False)
    manifests, total_bytes = prepare_all(selection, output)
    full_required_mbps = total_bytes * 8 * args.fps / args.frames / 1e6 if args.fps else 0.0
    p0_required_mbps = (sum(m["groups"]["P0"]["payload_bytes"] for m in manifests.values())
                        * 8 * args.fps / args.frames / 1e6 if args.fps else 0.0)
    print(f"INPUT {args.frames} NTCs, total {total_bytes / 1e6:.3f} MB")
    print(f"At {args.fps:g} FPS: full groups need ~{full_required_mbps:.2f} Mbps "
          f"and P0 alone ~{p0_required_mbps:.2f} Mbps, before QUIC/IP overhead.")
    if args.prepare_only:
        print(f"PREPARED {output} (no network transfer)")
        return
    if not shutil.which("cargo"):
        raise RuntimeError("Cargo unavailable; install Rust before network test")
    subprocess.run(["cargo", "build", "--release"], cwd=HERE, check=True)
    network_cm = (isolated_link(args.rate_mbps, args.delay_ms, args.loss_pct) if args.netns
                  else nullcontext({"host": "127.0.0.1"}))
    with network_cm as network:
        folder, timings, complete = run_transport(args, output, network)
    # Reassemble and verify every NTC's original canonical state digest.
    decoded = folder / "reconstructed"
    decoded.mkdir()
    for step, (frame, _, _) in enumerate(selection, 1):
        manifest = manifests[frame]
        count, bytes_received = reconstruct_ranked(
            folder / "received" / f"NTC_{frame:06}", manifest,
            decoded / f"NTC_{frame:06}.pth")
        if count != manifest["retained_blocks"] or bytes_received != manifest["payload_bytes_total"]:
            raise RuntimeError(f"Frame {frame} reconstruction mismatch")
        if step % 25 == 0 or step == args.frames:
            print(f"VERIFIED {step}/{args.frames} exact INT4 tensor states", flush=True)
    rows = []
    previous_full = 0.0
    for frame, _, _ in selection:
        current_full = complete[frame]
        rows.append({
            "frame": frame,
            "mode": args.mode,
            "payload_bytes": manifests[frame]["payload_bytes_total"],
            "p0_complete_ms_connection": f"{timings[(frame, 0)][1]:.3f}",
            "p1_complete_ms_connection": f"{timings[(frame, 1)][1]:.3f}",
            "p2_complete_ms_connection": f"{timings[(frame, 2)][1]:.3f}",
            "p3_complete_ms_connection": f"{timings[(frame, 3)][1]:.3f}",
            "ntc_complete_ms_connection": f"{current_full:.3f}",
            "p0_ms_since_previous_ntc": f"{timings[(frame, 0)][1] - previous_full:.3f}",
            "ntc_ms_since_previous_ntc": f"{current_full - previous_full:.3f}",
            "sha256_state_verified": "yes",
        })
        previous_full = current_full
    csv_path = folder / "per_frame.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    duration_ms = max(complete.values())
    summary = {
        "frames": args.frames, "mode": args.mode, "fps_target_max": args.fps,
        "note": "One persistent QUIC connection; ACK-gated frames (not pipelined live playback)",
        "total_payload_bytes": total_bytes, "total_receiver_time_ms": duration_ms,
        "delivered_complete_ntc_per_second": args.frames * 1000.0 / duration_ms,
        "required_payload_mbps_at_target_fps": full_required_mbps,
        "required_p0_payload_mbps_at_target_fps": p0_required_mbps,
        "network": {"netns": args.netns, "rate_mbps": args.rate_mbps if args.netns else None,
                    "delay_ms_one_way": args.delay_ms if args.netns else None,
                    "loss_pct_each_way": args.loss_pct if args.netns else None},
        "full_ntc_interval_ms_median": statistics.median(
            float(r["ntc_ms_since_previous_ntc"]) for r in rows),
        "reconstructed_states_sha256_verified": args.frames,
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nDONE {args.frames} NTCs, all tensor SHA256=OK, one Quinn connection")
    print(f"Delivered full NTC updates at {summary['delivered_complete_ntc_per_second']:.2f} FPS")
    print(f"Median full-NTC interval: {summary['full_ntc_interval_ms_median']:.2f} ms")
    print(f"RESULTS {folder}\nCSV {csv_path}")
    print("No real-time rendering or automatic stale-frame dropping in this version.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=299)
    parser.add_argument("--wire-dir", default=str(DEFAULT_WIRE_DIR))
    parser.add_argument("--priority-dir", default=str(DEFAULT_PRIORITY_DIR))
    parser.add_argument("--out", help="New output directory, never overwrite")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--mode", choices=["priority", "baseline"], default="priority")
    parser.add_argument("--netns", action="store_true", help="Isolated shaped link, asks for sudo")
    parser.add_argument("--rate-mbps", type=float, default=5)
    parser.add_argument("--delay-ms", type=float, default=10)
    parser.add_argument("--loss-pct", type=float, default=1)
    parser.add_argument("--fps", type=float, default=30, help="Maximum NTC injection rate; not a playback FPS guarantee")
    parser.add_argument("--slice-bytes", type=int, default=1024)
    parser.add_argument("--port", type=int, default=55443)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    try:
        run(args)
    except (RuntimeError, ValueError, FileNotFoundError, FileExistsError, KeyError, TypeError,
            OSError, TimeoutError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
