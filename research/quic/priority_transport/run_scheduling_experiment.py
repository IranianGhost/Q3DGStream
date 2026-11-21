"""Cross seven application admission methods with two native Quinn priority policies.

Six original methods plus P0-first are tested under equal and P0>P1>P2>P3
Quinn stream priority, holding P0..P3 source bytes and network conditions fixed.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

from run_ntc import DEFAULT_WIRE_DIR
from run_sequence import DEFAULT_PRIORITY_DIR
from evaluate_readiness import evaluate as evaluate_readiness
from analyze_arrival_order import read_orders, summarize as summarize_arrival_order


HERE = Path(__file__).resolve().parent
MODES = (
    "original", "sequential", "concurrent", "amirali_fifo", "amirali_strict",
    "amirali_weighted", "p0_first",
)
QUINN_POLICIES = ("equal", "tiered")


def summarize(path: Path, mode: str, quinn_priority: str, repetition: int, fps: float,
              tier_selection="full", fixed_tiers=4,
              live_folder: Path | None = None) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty experiment CSV: {path}")
    if any(not 1 <= int(row["selected_tiers"]) <= 4 or
           row["sha256_group_verified"] != "yes" or
           row["scheduling_mode"] != mode or
           row["quinn_priority_policy"] != quinn_priority for row in rows):
        raise RuntimeError(f"Invalid group receipt or mode: {path}")
    expected = 4 if tier_selection == "full" else fixed_tiers if tier_selection == "fixed" else None
    if expected is not None and any(int(row["selected_tiers"]) != expected for row in rows):
        raise RuntimeError(f"Invalid {expected}-group selection in {path}")
    tick = 1000.0 / fps
    first_frame = int(rows[0]["frame"])

    def median(field):
        return statistics.median(float(row[field]) for row in rows)

    def percentile(field, p=0.95):
        samples = sorted(float(row[field]) for row in rows if row.get(field) not in (None, ""))
        if not samples:
            return ""
        return round(samples[int((len(samples) - 1) * p)], 3)

    def optional_median(field):
        values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
        return round(statistics.median(values), 3) if values else ""

    def optional_sum(field):
        return sum(int(row[field]) for row in rows if row.get(field) not in (None, ""))

    hist = {f"P0..P{k}": sum(int(row["selected_tiers"]) == k + 1 for row in rows)
            for k in range(4)}
    result = {
        "repetition": repetition,
        "mode": mode,
        "quinn_priority": quinn_priority,
        "tier_selection": tier_selection,
        "fixed_tiers": fixed_tiers if tier_selection == "fixed" else "",
        "frames": len(rows),
        "selected_tier_histogram": json.dumps(hist, separators=(",", ":")),
        "median_p0_arrival_vs_tick_ms": round(statistics.median(
            float(row["p0_complete_ms_connection"]) -
            (int(row["frame"]) - first_frame) * tick for row in rows), 3),
        "median_p0_lead_before_full_ms": round(median("p0_lead_before_full_ms"), 3),
        "median_p0_lead_before_selected_ms": round(median("p0_lead_before_full_ms"), 3),
        "median_complete_ntc_vs_tick_ms": round(statistics.median(
            float(row["completion_ms"]) -
            (int(row["frame"]) - first_frame) * tick for row in rows), 3),
        "median_p0_sender_admission_lag_ms": round(statistics.median(
            float(row["p0_write_start_ms_sender"]) -
            (int(row["frame"]) - first_frame) * tick for row in rows), 3),
        "median_p0_delivery_after_write_end_ms": optional_median("p0_delivery_after_write_end_ms"),
        "median_p1_delivery_after_write_end_ms": optional_median("p1_delivery_after_write_end_ms"),
        "p0_completed_before_other_groups": sum(
            float(row["p0_complete_ms_connection"]) <= min(
                float(row[f"p{tier}_complete_ms_connection"])
                for tier in range(1, int(row["selected_tiers"]))
            ) if int(row["selected_tiers"]) > 1 else True
            for row in rows
        ),
        "median_concurrent_write_start_spread_ms": round(median("write_start_spread_ms"), 3),
        "median_rtt_ms": optional_median("rtt_ms"),
        "p95_rtt_ms": percentile("rtt_ms"),
        "median_sender_capacity_estimate_mbps": optional_median("available_mbps"),
        "median_client_observed_goodput_mbps": optional_median("client_goodput_mbps"),
        "median_cwnd_bytes": optional_median("cwnd_bytes"),
        "p95_estimated_backlog_ms": percentile("backlog_ms"),
        "p95_rtt_jitter_sample_ms": percentile("observed_rtt_jitter_sample_ms"),
        "p95_arrival_jitter_abs_ms": percentile("arrival_jitter_abs_ms"),
        "lost_packets_delta_sum": optional_sum("loss_delta_packets"),
        "congestion_events_delta_sum": optional_sum("congestion_delta_events"),
        "p0_deadline_misses": sum(row["p0_deadline_missed"].lower() == "true" for row in rows),
        "full_deadline_misses": sum(row["deadline_missed"].lower() == "true" for row in rows),
        "last_full_arrival_ms": round(max(float(row["completion_ms"]) for row in rows), 3),
        "payload_bytes": sum(int(row["payload_bytes"]) for row in rows),
        "mean_payload_delivery_mbps": round(
            sum(int(row["payload_bytes"]) for row in rows) * 8.0 /
            (max(float(row["completion_ms"]) for row in rows) * 1000.0), 3),
        "live_played_current_ticks": "",
        "live_stale_ticks": "",
        "live_tick_overruns": "",
        "live_median_gpu_work_ms": "",
        "live_mean_age_frames": "",
        "live_max_buffer_frames": "",
        "live_preparation_mode": "",
        "live_cpu_cache_hits": "",
        "live_dense_cache_hits": "",
        "live_exact_tier_fallbacks": "",
        "live_early_gpu_dequant_ms": "",
        "live_early_gpu_prep_crossed_tick": "",
        "live_early_gpu_preps_used_at_playout": "",
        "live_early_gpu_preps_not_used": "",
        "live_gpu_prep_guard_blocked_ticks": "",
        "live_async_gpu_prepares_enqueued": "",
        "live_async_gpu_ready_at_tick": "",
        "live_async_gpu_waited_at_tick": "",
        "live_async_gpu_wait_ms_total": "",
        "live_async_gpu_pending_at_next_tick": "",
        "live_median_tick_assembly_ms": "",
        "live_median_tick_dequant_ms": "",
        "live_median_tick_install_ms": "",
        "mean_psnr_vs_fp32": "",
        "mean_ssim_vs_fp32": "",
        "mean_lpips_vs_fp32": "",
        "csv": str(path),
    }
    if live_folder is not None:
        live = json.loads((live_folder / "playback_summary.json").read_text())
        with (live_folder / "playback_frames.csv").open() as handle:
            playback = list(csv.DictReader(handle))
        result.update({
            "live_played_current_ticks": live["ticks_current_ntc"],
            "live_stale_ticks": live["ticks_with_stale_state"],
            "live_tick_overruns": live["tick_overruns"],
            "live_median_gpu_work_ms": live["median_gpu_work_ms"],
            "live_mean_age_frames": round(statistics.mean(
                int(row["age_frames"]) for row in playback), 3),
            "live_max_buffer_frames": max(int(row["buffer_frames"]) for row in playback),
            "live_preparation_mode": live.get("preparation_mode", "none"),
            "live_cpu_cache_hits": live.get("cpu_cache_hits", 0),
            "live_dense_cache_hits": live.get("dense_cache_hits", 0),
            "live_exact_tier_fallbacks": live.get("exact_tier_fallbacks", 0),
            "live_early_gpu_dequant_ms": live.get("early_gpu_dequant_ms_total", 0),
            "live_early_gpu_prep_crossed_tick": live.get("early_gpu_prep_crossed_tick", 0),
            "live_early_gpu_preps_used_at_playout": live.get("early_gpu_preps_used_at_playout", 0),
            "live_early_gpu_preps_not_used": live.get("early_gpu_preps_not_used", 0),
            "live_gpu_prep_guard_blocked_ticks": live.get("gpu_prep_guard_blocked_ticks", 0),
            "live_async_gpu_prepares_enqueued": live.get("async_gpu_prepares_enqueued", 0),
            "live_async_gpu_ready_at_tick": live.get("async_gpu_ready_at_tick", 0),
            "live_async_gpu_waited_at_tick": live.get("async_gpu_waited_at_tick", 0),
            "live_async_gpu_wait_ms_total": live.get("async_gpu_wait_ms_total", 0),
            "live_async_gpu_pending_at_next_tick": live.get("async_gpu_pending_at_next_tick", 0),
            "live_median_tick_assembly_ms": live.get("median_on_tick_assembly_ms", ""),
            "live_median_tick_dequant_ms": live.get("median_on_tick_dequant_ms", ""),
            "live_median_tick_install_ms": live.get("median_on_tick_install_ms", ""),
        })
        quality = live_folder / "live_quality_summary.json"
        if quality.exists():
            scored = json.loads(quality.read_text())
            result.update({
                "mean_psnr_vs_fp32": scored["mean_psnr"],
                "mean_ssim_vs_fp32": scored["mean_ssim"],
                "mean_lpips_vs_fp32": scored["mean_lpips"] if scored["mean_lpips"] is not None else "",
            })
    return result


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("No experiment records")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_interaction_comparison(path: Path, rows: list[dict]):
    """One line per treatment, combining repetitions without hiding failures."""
    by_treatment = {
        (mode, policy): [row for row in rows if row["mode"] == mode and
                         row["quinn_priority"] == policy]
        for mode in MODES for policy in QUINN_POLICIES
    }
    result = []
    for (mode, policy), trials in by_treatment.items():
        if not trials:
            continue
        def med(field):
            vals = [float(row[field]) for row in trials if row.get(field) not in (None, "")]
            return round(statistics.median(vals), 3) if vals else ""
        n = sum(int(row["frames"]) for row in trials)
        p0_misses = sum(int(row["p0_deadline_misses"]) for row in trials)
        full_misses = sum(int(row["full_deadline_misses"]) for row in trials)
        result.append({
            "mode": mode,
            "quinn_priority": policy,
            "repetitions": len(trials),
            "ntc_frames": n,
            "p0_on_time": n - p0_misses,
            "p0_on_time_pct": round(100 * (n - p0_misses) / n, 2),
            "complete_ntcs_on_time": n - full_misses,
            "complete_ntcs_on_time_pct": round(100 * (n - full_misses) / n, 2),
            "median_last_ntc_delivery_ms": med("last_full_arrival_ms"),
            "median_p0_vs_tick_ms": med("median_p0_arrival_vs_tick_ms"),
            "median_full_ntc_vs_tick_ms": med("median_complete_ntc_vs_tick_ms"),
            "median_p0_lead_ms": med("median_p0_lead_before_full_ms"),
            "p0_completed_before_other_groups": sum(
                int(row["p0_completed_before_other_groups"]) for row in trials),
            "median_p0_delivery_after_write_end_ms": med("median_p0_delivery_after_write_end_ms"),
            "median_payload_goodput_mbps": med("mean_payload_delivery_mbps"),
            "median_live_current_ticks": med("live_played_current_ticks"),
            "median_live_stale_ticks": med("live_stale_ticks"),
            "median_gpu_work_ms": med("live_median_gpu_work_ms"),
            "median_psnr_vs_fp32": med("mean_psnr_vs_fp32"),
        })
    if result:
        write_csv(path, result)


def write_native_priority_deltas(path: Path, rows: list[dict]):
    """Within each admission method, compare matched-repetition priority effects."""
    result = []
    for mode in MODES:
        by_policy = {policy: [row for row in rows if row["mode"] == mode and
                      row["quinn_priority"] == policy] for policy in QUINN_POLICIES}
        if not all(by_policy.values()):
            continue
        matched = set(int(row["repetition"]) for row in by_policy["equal"]) & \
                  set(int(row["repetition"]) for row in by_policy["tiered"])
        if not matched:
            continue
        stats = {}
        for policy, trials in by_policy.items():
            matched_trials = [row for row in trials if int(row["repetition"]) in matched]
            n = sum(int(row["frames"]) for row in matched_trials)
            ontime = n - sum(int(row["p0_deadline_misses"]) for row in matched_trials)
            current = sum(int(row["live_played_current_ticks"]) for row in matched_trials
                          if row["live_played_current_ticks"] != "")
            stats[policy] = (n, ontime, current)
        n = stats["equal"][0]
        result.append({
            "mode": mode,
            "matched_repetitions": len(matched),
            "ntcs_per_policy": n,
            "equal_p0_on_time": stats["equal"][1],
            "tiered_p0_on_time": stats["tiered"][1],
            "quinn_priority_gain_p0_percentage_points": round(
                (stats["tiered"][1] - stats["equal"][1]) * 100 / n, 2),
            "equal_live_current_ticks": stats["equal"][2],
            "tiered_live_current_ticks": stats["tiered"][2],
            "quinn_priority_gain_current_ticks": stats["tiered"][2] - stats["equal"][2],
        })
    if result:
        write_csv(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--rate-mbps", type=float, default=5)
    parser.add_argument("--delay-ms", type=float, default=10)
    parser.add_argument("--loss-pct", type=float, default=1)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--playout-ms", type=float, default=150)
    parser.add_argument("--feedback-ms", type=int, default=100)
    parser.add_argument("--delivery-timeout-ms", type=int, default=120000)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--initial-mbps", type=float,
                        help="Capacity estimator ceiling; defaults to rate or max bandwidth profile")
    parser.add_argument("--tier-selection", choices=("full", "fixed", "adaptive"), default="full")
    parser.add_argument("--fixed-tiers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--max-inflight", type=int, choices=range(1, 129), default=16)
    parser.add_argument("--tier-delays-ms", default="0,0,0,0",
                        help="P0,P1,P2,P3 sender release offsets in milliseconds")
    parser.add_argument("--rate-profile", default="",
                        help="netns only: elapsed seconds:Mbps, e.g. 0:20,3:5,6:10")
    parser.add_argument("--live", action="store_true", help="Actually render the client GPU during each run")
    parser.add_argument("--progressive", action="store_true",
                        help="Render P0 immediately and refine with later arriving groups")
    parser.add_argument("--show-window", action="store_true",
                        help="Open a browser preview of progressive rendering")
    parser.add_argument("--preview-port", type=int, default=8765)
    parser.add_argument("--preview-linger-seconds", type=float, default=10)
    parser.add_argument("--save-stages", action="store_true",
                        help="Save per-priority PNGs in progressive mode")
    parser.add_argument("--preparation-mode", choices=("none", "cpu", "dense", "async"), default="none")
    parser.add_argument("--prefetch-frames", type=int, default=4)
    parser.add_argument("--capture-every", type=int, default=0)
    parser.add_argument("--ref-dir", type=Path, help="Exact cam00 FP32 images for quality scoring")
    parser.add_argument("--scene-base", type=Path)
    parser.add_argument("--viewer", type=Path)
    parser.add_argument("--inference-ms", type=float, default=2.59646)
    parser.add_argument("--render-ms", type=float, default=6.29560)
    parser.add_argument("--decode-ms", type=float, default=6.64757)
    parser.add_argument("--modes", default=",".join(MODES),
                        help="Application admission methods; defaults to six original plus P0-first")
    parser.add_argument("--quinn-policies", default=",".join(QUINN_POLICIES),
                        help="Native stream priority policies: equal,tiered")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running the network")
    parser.add_argument("--wire-dir", type=Path, default=DEFAULT_WIRE_DIR)
    parser.add_argument("--priority-dir", type=Path, default=DEFAULT_PRIORITY_DIR)
    parser.add_argument("--out", type=Path,
                        help="Results folder; defaults to a uniquely named directory")
    parser.add_argument("--netns", action="store_true",
                        help="Use isolated Linux network namespaces with tc netem")
    parser.add_argument("--seed", type=int, default=20261120,
                        help="Reproducible per-repetition mode-order randomization")
    args = parser.parse_args()
    if args.frames <= 0 or args.reps <= 0 or args.fps <= 0:
        parser.error("--frames, --reps and --fps must be positive")
    chosen_modes = [mode.strip() for mode in args.modes.split(",")]
    if not chosen_modes or len(set(chosen_modes)) != len(chosen_modes) or any(
        mode not in MODES for mode in chosen_modes):
        parser.error("--modes must be a unique subset of the seven admission methods")
    chosen_policies = [policy.strip() for policy in args.quinn_policies.split(",")]
    if not chosen_policies or len(set(chosen_policies)) != len(chosen_policies) or any(
        policy not in QUINN_POLICIES for policy in chosen_policies):
        parser.error("--quinn-policies must be a unique subset of equal,tiered")
    if args.ref_dir and (not args.live or args.capture_every == 0):
        parser.error("--ref-dir requires --live --capture-every 1 or greater")
    if args.progressive and not args.live:
        parser.error("--progressive requires --live")
    if args.live and args.start_frame != 0:
        parser.error("Live cumulative playback requires --start-frame 0")
    if args.rate_profile and not args.netns:
        parser.error("Bandwidth profiles require --netns")

    output = ((args.out.expanduser().resolve()) if args.out else
              HERE / f"results_priority_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)
    summary_rows = []
    per_frame_rows = []
    run_index = []
    rng = random.Random(args.seed)
    config = {**vars(args), "wire_dir": str(args.wire_dir),
              "priority_dir": str(args.priority_dir), "out": str(output)}
    if not args.dry_run:
        (output / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    print(f"SCHEDULING ABLATION: selection={args.tier_selection} "
          f"fixed_tiers={args.fixed_tiers} live={args.live} "
          f"methods={len(chosen_modes)} native_policies={len(chosen_policies)}\n"
          f"RESULTS {output}", flush=True)
    for rep in range(1, args.reps + 1):
        randomized = [(mode, policy) for mode in chosen_modes for policy in chosen_policies]
        rng.shuffle(randomized)
        for mode, policy in randomized:
            run_out = output / f"rep_{rep:02}" / f"{mode}__quinn_{policy}"
            if not args.dry_run:
                run_out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(HERE / "run_deadline.py"),
                "--scheduling-mode", mode,
                "--quinn-priority", policy,
                "--start-frame", str(args.start_frame),
                "--frames", str(args.frames),
                "--rate-mbps", str(args.rate_mbps),
                "--delay-ms", str(args.delay_ms),
                "--loss-pct", str(args.loss_pct),
                "--fps", str(args.fps),
                "--playout-ms", str(args.playout_ms),
                "--feedback-ms", str(args.feedback_ms),
                "--initial-mbps", str(args.initial_mbps or args.rate_mbps),
                "--tier-selection", args.tier_selection,
                "--fixed-tiers", str(args.fixed_tiers),
                "--max-inflight", str(args.max_inflight),
                "--tier-delays-ms", args.tier_delays_ms,
                "--inference-ms", str(args.inference_ms),
                "--render-ms", str(args.render_ms),
                "--decode-ms", str(args.decode_ms),
                "--delivery-timeout-ms", str(args.delivery_timeout_ms),
                "--timeout", str(args.timeout),
                "--wire-dir", str(args.wire_dir),
                "--priority-dir", str(args.priority_dir),
                "--out", str(run_out),
            ]
            if args.netns:
                cmd.append("--netns")
            if args.rate_profile:
                cmd.extend(["--rate-profile", args.rate_profile])
                if args.initial_mbps is None:
                    # Don't cap a recovering connection at its low initial rate.
                    from net_link import parse_rate_profile
                    capacity = max(rate for _, rate in parse_rate_profile(args.rate_profile))
                    idx = cmd.index("--initial-mbps") + 1
                    cmd[idx] = str(capacity)
            if args.live:
                cmd.extend(["--live", "--capture-every", str(args.capture_every),
                            "--preparation-mode", args.preparation_mode,
                            "--prefetch-frames", str(args.prefetch_frames)])
            if args.progressive:
                cmd.append("--progressive")
            if args.show_window:
                cmd.append("--show-window")
            if args.progressive:
                cmd.extend(["--preview-port", str(args.preview_port),
                            "--preview-linger-seconds", str(args.preview_linger_seconds)])
                if args.save_stages:
                    cmd.append("--save-stages")
            for flag, value in (("--scene-base", args.scene_base), ("--viewer", args.viewer)):
                if value is not None:
                    cmd.extend([flag, str(value)])
            print(f"\nREPETITION {rep}/{args.reps} MODE {mode} QUINN {policy}", flush=True)
            if args.dry_run:
                print(" ".join(cmd), flush=True)
                continue
            subprocess.run(cmd, cwd=HERE, check=True)
            live_folder = run_out / "priority" / "live" if args.live else None
            if args.ref_dir:
                score = [sys.executable, str(HERE / "evaluate_live.py"),
                         "--live-dir", str(live_folder), "--ref-dir", str(args.ref_dir)]
                subprocess.run(score, cwd=HERE, check=True)
            # Evaluate the same three estimators on the delivered traffic in
            # each writing-policy arm. This also scores real CUDA readiness
            # when --live produced apply_events.csv.
            readiness = evaluate_readiness(run_out, fps=args.fps)
            per_frame_csv = run_out / "priority" / "per_frame_deadline.csv"
            item = summarize(per_frame_csv, mode, policy, rep, args.fps,
                             args.tier_selection, args.fixed_tiers, live_folder)
            for record in readiness["summaries"]:
                if record["subset"] not in ("P0", "applied_ntcs", "rendered_ntcs"):
                    continue
                prefix = f"ready_{record['domain']}_{record['predictor']}"
                item[f"{prefix}_mae_ms"] = record["mae_ms"]
                item[f"{prefix}_false_ontime"] = record["false_on_time_predictions"]
                item[f"{prefix}_samples"] = record["samples"]
            # All summary rows must have the same columns even if some GPU
            # runs did not produce individually rendered NTCs.
            summary_rows.append(item)
            with per_frame_csv.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    per_frame_rows.append({"repetition": rep, "mode": mode,
                                           "quinn_priority": policy, **row})
            run_index.append({"repetition": rep, "mode": mode,
                              "quinn_priority": policy, "results": str(run_out)})
            write_csv(output / "summary_scheduling.csv", summary_rows)
            write_interaction_comparison(output / "comparison_priority_matrix.csv", summary_rows)
            write_native_priority_deltas(output / "native_priority_deltas.csv", summary_rows)
            write_csv(output / "per_frame_all.csv", per_frame_rows)
            write_csv(output / "run_index.csv", run_index)
            print(f"P0 lead={item['median_p0_lead_before_full_ms']:.2f}ms "
                  f"P0 deadline misses={item['p0_deadline_misses']}/{args.frames} "
                  f"full misses={item['full_deadline_misses']}/{args.frames}", flush=True)

    if args.dry_run:
        print("DRY RUN: no files or network changes made")
        return
    order_records = read_orders(output / "per_frame_all.csv")
    write_csv(output / "arrival_order_per_frame.csv", order_records)
    write_csv(output / "arrival_order_by_method.csv", summarize_arrival_order(order_records))
    print(f"\nCOMPLETE {len(summary_rows)} runs "
          f"({args.reps} reps x {len(chosen_modes)} methods x {len(chosen_policies)} Quinn policies)")
    print(f"SUMMARY {output / 'summary_scheduling.csv'}")
    print(f"PRIORITY MATRIX {output / 'comparison_priority_matrix.csv'}")
    print(f"NATIVE PRIORITY DELTAS {output / 'native_priority_deltas.csv'}")
    print(f"PER-FRAME {output / 'per_frame_all.csv'}")
    print(f"ARRIVAL ORDER {output / 'arrival_order_by_method.csv'}")
    print(f"ARRIVAL ORDER PER NTC {output / 'arrival_order_per_frame.csv'}")
    print("NOTE: Full/fixed tiers use identical payload selections across strategies; "
          "adaptive tiers may select different payloads under feedback. "
          "Full QoE requires --live, capture, and exact FP32 reference PNGs.")


if __name__ == "__main__":
    main()
