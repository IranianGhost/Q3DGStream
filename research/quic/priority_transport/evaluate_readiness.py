#!/usr/bin/env python3
"""Evaluate prospective QUIC transport predictions and tick-gated GPU readiness.

All timestamps are aligned to the RECEIVER connection epoch on the same host.
Three simple predictors are evaluated on identical delivered traffic:
  configured: configured rate ceiling (optimistic baseline),
  cwnd:       min(configured, congestion_window / RTT),
  hybrid:     rate actually estimated by the current Rust sender.

No predictor is trained on the evaluated frame's arrival time. Actual GPU
events are optional, but required for claims about rendered-frame readiness.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


MODELS = ("configured", "cwnd", "hybrid")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(samples, q):
    xs = sorted(samples)
    if not xs:
        return None
    index = (len(xs) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return xs[low]
    return xs[low] * (high - index) + xs[high] * (index - low)


def candidate_bytes(row, tier_count):
    return sum(int(row[f"stage_p{i}_bytes"]) for i in range(tier_count))


def estimated_rate_mbps(row, model):
    initial = float(row["initial_capacity_ceiling_mbps"])
    utilization = float(row["sender_utilization"])
    rtt_ms = max(1.0, float(row["rtt_ms"]))
    cwnd_mbps = float(row["cwnd_bytes"]) * 8.0 / (rtt_ms * 1000.0)
    if model == "configured":
        rate = initial * utilization
    elif model == "cwnd":
        rate = min(initial, max(0.1, cwnd_mbps)) * utilization
    elif model == "hybrid":
        # Rust's DECISION capacity_mbps includes utilization and its
        # congestion/backlog feedback corrections.
        rate = float(row["available_mbps"])
    else:
        raise ValueError(f"Unknown predictor {model}")
    return max(0.1, rate)


def predict_transport_ms(row, tier_count, model):
    rate = estimated_rate_mbps(row, model)
    waiting = int(row["sender_outstanding_bytes"])
    candidate = candidate_bytes(row, tier_count)
    # Bytes * 8 / (Mbps * 1000) gives ms. Queue and candidate may span
    # multiple RTTs. Retransmissions are only reflected indirectly in rate.
    service_ms = (waiting + candidate) * 8.0 / (rate * 1000.0)
    propagation_ms = float(row["rtt_ms"]) / 2.0
    margin_ms = 2.0 * float(row["observed_rtt_jitter_sample_ms"])
    return (float(row["sender_decision_ms"]) +
            float(row["sender_to_receiver_clock_offset_ms"]) +
            service_ms + propagation_ms + margin_ms)


def gated_gpu_prediction_ms(row, predicted_arrival, nominal_ms, tick_ms, include_render):
    # The existing live_client starts work at a playout tick, not immediately
    # after QUIC arrival. If a group misses a tick, it can only be applied on
    # the next tick (provided all cumulative predecessors are ready).
    ticks_late = max(0, math.ceil((predicted_arrival - nominal_ms) / tick_ms))
    gated_start = nominal_ms + ticks_late * tick_ms
    work = float(row["predicted_decode_ms"]) + float(row["predicted_inference_ms"])
    if include_render:
        work += float(row["predicted_render_ms"])
    return gated_start + work


def summarize(samples):
    abs_errors = [abs(x["error_ms"]) for x in samples]
    signed = [x["error_ms"] for x in samples]
    actual_ontime = [x["actual_ontime"] for x in samples]
    predicted_ontime = [x["predicted_ontime"] for x in samples]
    return {
        "samples": len(samples),
        "mae_ms": round(statistics.mean(abs_errors), 3) if samples else "",
        "p90_absolute_error_ms": round(percentile(abs_errors, 0.90), 3) if samples else "",
        "median_signed_error_ms": round(statistics.median(signed), 3) if samples else "",
        "predicted_on_time": sum(predicted_ontime),
        "actual_on_time": sum(actual_ontime),
        "false_on_time_predictions": sum(p and not a for p, a in zip(predicted_ontime, actual_ontime)),
        "false_late_predictions": sum(not p and a for p, a in zip(predicted_ontime, actual_ontime)),
    }


def evaluate(run_dir: Path, out: Path | None = None, fps: float | None = None):
    trial = run_dir / "priority" if (run_dir / "priority").is_dir() else run_dir
    source = trial / "per_frame_deadline.csv"
    rows = read_csv(source)
    if not rows:
        raise ValueError(f"No transport rows in {source}")
    required = ("sender_decision_ms", "sender_outstanding_bytes", "stage_p0_bytes",
                "sender_to_receiver_clock_offset_ms", "initial_capacity_ceiling_mbps",
                "predicted_decode_ms", "predicted_inference_ms", "predicted_render_ms")
    missing = [k for k in required if k not in rows[0]]
    if missing:
        raise ValueError(f"Missing predictor instrumentation {missing}; use this updated ZIP")
    out = out or (trial / "readiness")
    out.mkdir(parents=True, exist_ok=True)
    live_folder = trial / "live"
    events_file = live_folder / "apply_events.csv"
    events = {int(e["ntc"]): e for e in read_csv(events_file)} if events_file.is_file() else {}
    has_gpu = events_file.is_file()
    transport_samples = []
    gpu_inference_samples = []
    gpu_display_samples = []
    # Receiver deadlines contain the frame cadence even if delivery is late.
    # On a single frame, require an explicit FPS or locate a parent run config.
    config_path = next((folder / "config.json" for folder in (run_dir, *run_dir.parents)
                        if (folder / "config.json").is_file()), None)
    config = json.loads(config_path.read_text()) if config_path else {}
    if fps is not None:
        tick_ms = 1000.0 / fps
    elif "fps" in config:
        tick_ms = 1000.0 / float(config["fps"])
    elif len(rows) > 1:
        tick_ms = (float(rows[1]["deadline_ms_receiver"]) -
                   float(rows[0]["deadline_ms_receiver"]))
    else:
        raise ValueError("Single-frame run without config.json requires --fps")
    if not math.isfinite(tick_ms) or tick_ms <= 0:
        raise ValueError("Invalid FPS or receiver deadlines")

    seen_frames = set()
    for row in rows:
        frame = int(row["frame"])
        if frame in seen_frames:
            raise ValueError(f"Duplicate NTC frame {frame}")
        seen_frames.add(frame)
        selected = int(row["selected_tiers"])
        if not 1 <= selected <= 4:
            raise ValueError(f"Invalid selected tier count: {selected} for NTC {frame}")
        if candidate_bytes(row, selected) != int(row["payload_bytes"]):
            raise ValueError(f"NTC {frame} stage sizes disagree with selected payload")
        nominal = float(row["deadline_ms_receiver"])
        for model in MODELS:
            for tier_count in range(1, selected + 1):
                predicted = predict_transport_ms(row, tier_count, model)
                actual = max(float(row[f"p{i}_complete_ms_connection"])
                             for i in range(tier_count))
                transport_samples.append({
                    "frame": frame, "predictor": model, "tier": f"P0..P{tier_count-1}",
                    "candidate_bytes": candidate_bytes(row, tier_count),
                    "estimated_rate_mbps": round(estimated_rate_mbps(row, model), 3),
                    "predicted_arrival_ms": round(predicted, 3),
                    "actual_arrival_ms": round(actual, 3),
                    "error_ms": predicted - actual,
                    "deadline_ms": nominal,
                    "predicted_ontime": predicted <= nominal,
                    "actual_ontime": actual <= nominal,
                })
            if has_gpu and frame in events:
                e = events[frame]
                used = int(e["applied_tiers"])
                if not 1 <= used <= selected:
                    raise ValueError(f"GPU used {used} groups, sender selected {selected}, frame {frame}")
                predicted_arrival = predict_transport_ms(row, used, model)
                inference_predicted = gated_gpu_prediction_ms(
                    row, predicted_arrival, nominal, tick_ms, include_render=False)
                inference_actual = float(e["inference_ready_ms"])
                display_deadline = nominal + tick_ms
                gpu_inference_samples.append({
                    "frame": frame, "predictor": model,
                    "tier_applied": f"P0..P{used-1}",
                    "actual_apply_tick": int(e["tick"]),
                    "predicted_inference_ready_ms": round(inference_predicted, 3),
                    "actual_inference_ready_ms": round(inference_actual, 3),
                    "error_ms": inference_predicted - inference_actual,
                    "frame_end_deadline_ms": round(display_deadline, 3),
                    "predicted_ontime": inference_predicted <= display_deadline,
                    "actual_ontime": inference_actual <= display_deadline,
                })
                if str(e["rendered_this_tick"]) == "1" and e["render_ready_ms"]:
                    render_predicted = gated_gpu_prediction_ms(
                        row, predicted_arrival, nominal, tick_ms, include_render=True)
                    render_actual = float(e["render_ready_ms"])
                    gpu_display_samples.append({
                        "frame": frame, "predictor": model,
                        "tier_applied": f"P0..P{used-1}",
                        "actual_render_tick": int(e["tick"]),
                        "predicted_render_ready_ms": round(render_predicted, 3),
                        "actual_render_ready_ms": round(render_actual, 3),
                        "error_ms": render_predicted - render_actual,
                        "frame_end_deadline_ms": round(display_deadline, 3),
                        "predicted_ontime": render_predicted <= display_deadline,
                        "actual_ontime": render_actual <= display_deadline,
                    })

    write_csv(out / "transport_predictions.csv", transport_samples)
    write_csv(out / "gpu_inference_predictions.csv", gpu_inference_samples)
    write_csv(out / "gpu_display_predictions.csv", gpu_display_samples)
    summaries = []
    for domain, group in (("transport", transport_samples),
                          ("gpu_inference", gpu_inference_samples),
                          ("gpu_display", gpu_display_samples)):
        for model in MODELS:
            candidates = [x for x in group if x["predictor"] == model]
            if domain == "transport":
                groups = [("P0", [x for x in candidates if x["tier"] == "P0..P0"]),
                          ("all_tiers", candidates)]
            else:
                groups = [("applied_ntcs" if domain == "gpu_inference" else "rendered_ntcs",
                           candidates)]
            for subset, values in groups:
                summaries.append({"domain": domain, "predictor": model,
                                  "subset": subset, **summarize(values)})
    write_csv(out / "readiness_summary.csv", summaries)
    report = {
        "transport_frames": len(rows), "selected_payload_bytes": sum(int(r["payload_bytes"]) for r in rows),
        "gpu_instrumented": has_gpu, "gpu_applied_ntcs": len(events),
        "gpu_individually_rendered_ntcs": sum(str(e["rendered_this_tick"]) == "1" for e in events.values()),
        "gpu_ntcs_not_applied_during_playout": len(rows) - len(events) if has_gpu else None,
        "gpu_applied_but_not_individually_rendered": (
            sum(str(e["rendered_this_tick"]) != "1" for e in events.values()) if has_gpu else None),
        "summaries": summaries,
        "limits": [
            "Sender/receiver clock alignment relies on both processes sharing one host wall clock.",
            "Transport MAE concerns selected cumulative tiers; unsent tiers cannot be scored.",
            "GPU work starts at the playout tick; tick-gated predictions approximate this policy.",
            "GPU inference includes every applied NTC; GPU display scores only individually rendered NTCs.",
            "GPU prediction is conditional on the tier actually applied, not a prospective tier-choice policy.",
            "Frames not applied and intermediate catch-up states are excluded from applicable error metrics, but counted separately.",
            "Cumulative backlog, retransmission timing and GPU queue may not be accurately predicted.",
            "Actual display success uses nominal playout tick plus one 33.33ms frame interval.",
        ],
    }
    (out / "readiness_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"READINESS {len(rows)} NTCs, GPU events={len(events)}")
    for record in summaries:
        if record["subset"] in ("P0", "applied_ntcs", "rendered_ntcs"):
            print(f"  {record['domain']:11s} {record['predictor']:10s} {record['subset']:14s} "
                  f"MAE={record['mae_ms']}ms false-on-time={record['false_on_time_predictions']} "
                  f"of {record['samples']}")
    print(f"RESULTS {out}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="One run's rep_XX/MODE output directory (contains priority/)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fps", type=float, help="Required for single-frame runs without a config.json")
    args = parser.parse_args()
    evaluate(args.run_dir, args.out, args.fps)


if __name__ == "__main__":
    main()
