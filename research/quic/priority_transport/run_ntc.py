"""Send a real compressed NTC wire file through four Quinn streams.

Default four groups are contiguous BYTE QUARTERS for a *transport test* only.
They are NOT saliency groups. An optional --segments-json maps exact byte
ranges to P0..P3; it must cover the input exactly once (no duplication).
"""

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pwd
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

from net_link import isolated_link

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from q3dg_paths import WIRE_DIR as DEFAULT_WIRE_DIR
TIMING_RE = re.compile(r"^TIMING ([0-3]) (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)$")
MAX_GROUP_BYTES = 128 * 1024 * 1024


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


def find_wire(args):
    if args.wire:
        candidate = Path(args.wire).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate
    directory = Path(args.wire_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"NTC wire directory not found: {directory}")
    pattern = re.compile(rf"(?<!\d)0*{args.frame}(?!\d)")
    found = sorted(
        p for p in directory.iterdir()
        if p.is_file() and pattern.search(p.stem)
    )
    if len(found) != 1:
        examples = ", ".join(p.name for p in sorted(directory.iterdir()) if p.is_file())[:350]
        raise RuntimeError(
            f"Expected one matching frame {args.frame} in {directory}, found {len(found)}. "
            f"Examples: {examples}. Use --wire /exact/path/to/file."
        )
    return found[0]


def segments_for(contents, segment_json):
    length = len(contents)
    if length < 4:
        raise ValueError("NTC wire payload must contain at least 4 bytes")
    if segment_json:
        info = json.loads(Path(segment_json).expanduser().read_text())
        raw = info["segments"] if isinstance(info, dict) else info
        if not isinstance(raw, list) or not raw:
            raise ValueError("--segments-json must contain a nonempty segments array")
        segments = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("Each segment must be an object")
            offset, count, group = (item[k] for k in ("offset", "length", "priority"))
            if type(offset) is not int or type(count) is not int or type(group) is not int:
                raise ValueError("Segment offset, length, priority must be integers")
            segments.append({"offset": offset, "length": count, "priority": group})
        grouping = "explicit_byte_ranges"
    else:
        quotient, remainder = divmod(length, 4)
        cursor = 0
        segments = []
        for group in range(4):
            count = quotient + (1 if group < remainder else 0)
            segments.append({"offset": cursor, "length": count, "priority": group})
            cursor += count
        grouping = "contiguous_quarters_TRANSPORT_ONLY"

    cursor = 0
    for segment in sorted(segments, key=lambda s: s["offset"]):
        if segment["length"] <= 0 or segment["offset"] != cursor:
            raise ValueError(f"Priority segments must partition wire bytes exactly: gap/overlap at {cursor}")
        if not 0 <= segment["priority"] <= 3:
            raise ValueError("priority must be 0 (highest) through 3 (lowest)")
        cursor += segment["length"]
    if cursor != length:
        raise ValueError(f"Segment coverage {cursor} differs from source length {length}")
    return segments, grouping


def prepare(wire, segments_json, output):
    source = wire.read_bytes()
    segments, grouping = segments_for(source, segments_json)
    payloads = [bytearray() for _ in range(4)]
    for item in sorted(segments, key=lambda s: s["offset"]):
        a, b, p = item["offset"], item["length"], item["priority"]
        payloads[p].extend(source[a:a + b])
    input_directory = output / "send"
    input_directory.mkdir()
    manifest = {
        "source": str(wire),
        "source_size_bytes": len(source),
        "source_sha256": digest(source),
        "grouping": grouping,
        "priority_semantics": "P0 highest; P3 lowest",
        "segments": segments,
        "groups": {},
    }
    for index, payload in enumerate(payloads):
        if len(payload) > MAX_GROUP_BYTES:
            raise ValueError(f"P{index} too large")
        (input_directory / f"P{index}.bin").write_bytes(payload)
        manifest["groups"][f"P{index}"] = {
            "size_bytes": len(payload), "sha256": digest(payload),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"SOURCE {wire}: {len(source)} bytes, SHA256 {manifest['source_sha256'][:16]}…")
    for index, payload in enumerate(payloads):
        print(f"  P{index}: {len(payload)} bytes")
    if grouping.startswith("contiguous"):
        print("NOTE: Byte quarters demonstrate QUIC transport only; NO 3D saliency is inferred.")
    return manifest


def reconstruct(received_dir, manifest, output_path):
    groups = []
    for index in range(4):
        contents = (received_dir / f"P{index}.bin").read_bytes()
        expected = manifest["groups"][f"P{index}"]
        if len(contents) != expected["size_bytes"] or digest(contents) != expected["sha256"]:
            raise RuntimeError(f"Received P{index} differs from sent P{index}")
        groups.append(contents)
    cursors = [0, 0, 0, 0]
    reconstructed = bytearray(manifest["source_size_bytes"])
    for segment in sorted(manifest["segments"], key=lambda s: s["offset"]):
        offset, count, p = segment["offset"], segment["length"], segment["priority"]
        reconstructed[offset:offset + count] = groups[p][cursors[p]:cursors[p] + count]
        cursors[p] += count
    for p, contents in enumerate(groups):
        if cursors[p] != len(contents):
            raise RuntimeError(f"Unconsumed received P{p} bytes")
    if digest(reconstructed) != manifest["source_sha256"]:
        raise RuntimeError("Full NTC SHA256 mismatch after QUIC reassembly")
    output_path.write_bytes(reconstructed)
    return len(reconstructed)


def pump(name, process, messages):
    for line in process.stdout:
        messages.put((name, line.rstrip("\n")))
    messages.put((name, "__EXIT__"))


def within_namespace(command, namespace):
    if not namespace:
        return command
    # Root is required for ip netns exec, but the actual transport runs as
    # the invoking user, keeping the received files user-owned.
    username = pwd.getpwuid(os.getuid()).pw_name
    return ["sudo", "-n", "ip", "netns", "exec", namespace,
            "sudo", "-n", "-u", username] + command


def trial(binary, mode, folder, manifest, network, args, verify_reconstruction=None):
    received_dir = folder / "received"
    received_dir.mkdir(parents=True)
    certificate = folder / "cert.der"
    receiver_cmd = within_namespace([
        str(binary), "receiver", "--host", network["host"],
        "--port", str(args.port), "--cert", str(certificate),
        "--received-dir", str(received_dir),
    ], network.get("server_ns"))
    sender_cmd = within_namespace([
        str(binary), "sender", "--host", network["host"],
        "--port", str(args.port), "--cert", str(certificate),
        "--payload-dir", str(folder.parent / "send"),
        "--slice-bytes", str(args.slice_bytes),
        *(["--baseline"] if mode == "baseline" else []),
    ], network.get("client_ns"))
    messages = queue.Queue()
    server = subprocess.Popen(receiver_cmd, cwd=HERE, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=1)
    threading.Thread(target=pump, args=("RECEIVER", server, messages), daemon=True).start()
    sender = None
    started = time.monotonic()
    ready = False
    finished = False
    timings = {}
    recent = []
    try:
        while not ready:
            if server.poll() is not None:
                raise RuntimeError(f"Receiver exited early: {server.returncode}. {'; '.join(recent[-6:])}")
            if time.monotonic() - started > 15:
                raise TimeoutError(f"Receiver READY timeout. {'; '.join(recent[-6:])}")
            try:
                name, line = messages.get(timeout=0.05)
            except queue.Empty:
                continue
            recent.append(f"{name} {line}")
            if name == "RECEIVER" and line.startswith("READY ") and certificate.is_file():
                ready = True
        sender = subprocess.Popen(sender_cmd, cwd=HERE, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, bufsize=1)
        threading.Thread(target=pump, args=("SENDER", sender, messages), daemon=True).start()
        while True:
            if time.monotonic() - started > args.timeout:
                raise TimeoutError(f"NTC transfer timed out. {'; '.join(recent[-12:])}")
            try:
                name, line = messages.get(timeout=0.05)
            except queue.Empty:
                line = ""
                name = ""
            if line and line != "__EXIT__":
                recent.append(f"{name} {line}")
                match = TIMING_RE.fullmatch(line) if name == "RECEIVER" else None
                if match:
                    index = int(match[1])
                    if index in timings:
                        raise RuntimeError(f"Duplicate timing P{index}")
                    timings[index] = (float(match[2]), float(match[3]))
                if name == "RECEIVER" and line.startswith("COMPLETE "):
                    finished = True
                if "_ERROR" in line:
                    raise RuntimeError(line)
            if sender.poll() is not None:
                if sender.returncode != 0:
                    raise RuntimeError(f"Sender failed ({sender.returncode}). {'; '.join(recent[-12:])}")
                if finished and len(timings) == 4:
                    break
    finally:
        for process in (sender, server):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)

    for index in range(4):
        first, complete = timings[index]
        if not 0 <= first <= complete:
            raise RuntimeError(f"Bad receiver clock for P{index}")
    if verify_reconstruction is None:
        result_file = folder / "reconstructed.ntc_wire"
        transferred = reconstruct(received_dir, manifest, result_file)
    else:
        transferred = verify_reconstruction(received_dir, folder)
    return timings, transferred


def run(args):
    if args.reps < 1 or not 1 <= args.port <= 65535 or args.timeout <= 0:
        raise ValueError("Invalid --reps, --port, or --timeout")
    if args.slice_bytes <= 0 or args.rate_mbps <= 0 or args.delay_ms < 0 or not 0 <= args.loss_pct < 100:
        raise ValueError("Invalid --slice-bytes or network parameters")
    wire = find_wire(args)
    default_name = f"results_ntc{args.frame}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    output = Path(args.out).expanduser().resolve() if args.out else (HERE / default_name)
    # Never overwrite any previous experiment's output directory.
    output.mkdir(parents=True, exist_ok=False)
    manifest = prepare(wire, args.segments_json, output)
    if args.prepare_only:
        print(f"PREPARED {output}. Add --segments-json for a semantic mapping if one exists.")
        return

    cargo = shutil.which("cargo")
    if not cargo:
        raise RuntimeError("cargo not found. Install Rust with rustup (see README.md).")
    subprocess.run([cargo, "build", "--release"], check=True, cwd=HERE)
    binary = HERE / "target" / "release" / "quinn-ntc-demo"
    if not binary.is_file():
        raise FileNotFoundError(binary)

    context = (isolated_link(args.rate_mbps, args.delay_ms, args.loss_pct)
               if args.netns else nullcontext({"host": "127.0.0.1"}))
    rows = []
    with context as network:
        for repeat in range(1, args.reps + 1):
            modes = ("baseline", "priority") if repeat % 2 else ("priority", "baseline")
            for mode in modes:
                folder = output / f"run{repeat:03d}_{mode}"
                folder.mkdir()
                timings, transferred = trial(binary, mode, folder, manifest, network, args)
                frame_ms = max(entry[1] for entry in timings.values())
                order = " ".join(f"P{i}" for i in sorted(timings, key=lambda i: timings[i][1]))
                print(f"Repeat {repeat:02d} {mode:8s} P0={timings[0][1]:8.2f} ms "
                      f"NTC={frame_ms:8.2f} ms order={order} SHA256=OK", flush=True)
                for index in range(4):
                    first_ms, complete_ms = timings[index]
                    rows.append({
                        "repeat": repeat, "mode": mode, "band": f"P{index}",
                        "payload_bytes": manifest["groups"][f"P{index}"]["size_bytes"],
                        "first_ms": f"{first_ms:.3f}", "complete_ms": f"{complete_ms:.3f}",
                        "full_ntc_ms": f"{frame_ms:.3f}",
                        "ntc_wire_bytes": transferred, "sha256_verified": "yes",
                        "netns": int(args.netns),
                        "rate_mbps": args.rate_mbps if args.netns else "",
                        "one_way_delay_ms": args.delay_ms if args.netns else "",
                        "loss_pct": args.loss_pct if args.netns else "",
                    })
    csv_file = output / "metrics.csv"
    with csv_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nMedian NTC group completion (ms):")
    print("Group     Baseline   Priority   Baseline - Priority")
    for index in range(4):
        baseline = statistics.median(float(row["complete_ms"]) for row in rows
                                     if row["mode"] == "baseline" and row["band"] == f"P{index}")
        priority = statistics.median(float(row["complete_ms"]) for row in rows
                                     if row["mode"] == "priority" and row["band"] == f"P{index}")
        print(f"P{index}       {baseline:10.2f} {priority:10.2f} {baseline - priority:12.2f}")
    print(f"\nResults: {output}\nCSV: {csv_file}")
    print("SHA256 confirms exact wire-byte reconstruction; this is NOT a PSNR measurement.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wire", help="Exact NTC wire file path (preferred when filename is ambiguous)")
    parser.add_argument("--wire-dir", default=str(DEFAULT_WIRE_DIR))
    parser.add_argument("--frame", type=int, default=114)
    parser.add_argument("--segments-json", help="Optional mapping of byte offset/length/priority 0..3")
    parser.add_argument("--out", help="NEW output folder; existing folder is never overwritten")
    parser.add_argument("--prepare-only", action="store_true", help="Inspect and group input, without Rust/QUIC")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--netns", action="store_true", help="Isolated shaped network, requires sudo and iproute2")
    parser.add_argument("--rate-mbps", type=float, default=5)
    parser.add_argument("--delay-ms", type=float, default=10)
    parser.add_argument("--loss-pct", type=float, default=1)
    parser.add_argument("--slice-bytes", type=int, default=1024)
    parser.add_argument("--port", type=int, default=55443)
    parser.add_argument("--timeout", type=float, default=120)
    parameters = parser.parse_args()
    try:
        run(parameters)
    except (RuntimeError, FileNotFoundError, FileExistsError, ValueError, KeyError, TypeError,
            TimeoutError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ERROR: {error}\n")
