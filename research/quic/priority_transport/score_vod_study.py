#!/usr/bin/env python3
"""Replay the NTC tiers actually applied during a VoD study and score its frames."""

import argparse
import hashlib
import os
import csv
from pathlib import Path
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent


def played_tiers(path, frames):
    with path.open(newline="") as source:
        events = list(csv.DictReader(source))
    if len(events) != frames:
        raise ValueError(f"Expected {frames} apply events in {path}")
    tiers = []
    for index, row in enumerate(events):
        count = int(row["applied_tiers"])
        if (int(row["ntc"]) != index or int(row["tick"]) != index
                or row["rendered_this_tick"] != "1" or not 1 <= count <= 4):
            raise ValueError(f"Invalid applied update {index} in {path}")
        tiers.append(count)
    return tiers


def image_name(index):
    return f"frame_{index + 1:06}.png"


def replay_tiers(payloads, label, output, viewer, scene, wire, manifests, tiers, width, height, source_precision=None):
    from PIL import Image
    from live_client import make_scene_overlay, received_state, scene_renderer

    images = output / "renders"
    images.mkdir(parents=True, exist_ok=False)
    overlay = make_scene_overlay(scene, wire / "NTC_000000.pth", output)
    with scene_renderer(viewer, overlay, width, height) as (renderer, torch, gl, loader, capture):
        renderer.draw(0)
        torch.cuda.synchronize()
        gl.glFinish()
        with torch.no_grad():
            for index, count in enumerate(tiers):
                groups = payloads / f"NTC_{index:06}"
                if source_precision is not None:
                    groups = groups / f"int{source_precision}"
                state, _ = received_state(groups, manifests[str(index)], count)
                dense = (state["model.params"] if "_precision_bits" in state else
                         loader._dequantize_sparse_int4_ntc_state_if_needed(state)["model.params"])
                renderer.NTCs[0].load_state_dict({"model.params": dense}, strict=False)
                renderer.query_NTC(renderer.gaussians.xyz, 0)
                renderer.current_timestep = 0
                renderer.draw(0)
                torch.cuda.synchronize()
                gl.glFinish()
                Image.fromarray(capture(width, height)).save(images / image_name(index), compress_level=1)
                if (index + 1) % 50 == 0 or index + 1 == len(tiers):
                    print(f"REPLAY {label} {index + 1}/{len(tiers)}", flush=True)
    return images


def compare_saved_frames(live, replay, frames):
    import numpy as np
    from PIL import Image

    shared = [image_name(i) for i in range(frames) if (live / image_name(i)).is_file()]
    largest_difference = 0
    identical = 0
    for name in shared:
        with Image.open(live / name) as original, Image.open(replay / name) as regenerated:
            a = np.asarray(original.convert("RGB"), dtype=np.int16)
            b = np.asarray(regenerated.convert("RGB"), dtype=np.int16)
        if a.shape != b.shape:
            raise ValueError(f"Replayed dimensions differ from live capture {live / name}")
        difference = int(np.abs(a - b).max())
        largest_difference = max(difference, largest_difference)
        identical += difference == 0
    return {
        "overlapping_captures": len(shared),
        "identical_captures": identical,
        "max_absolute_rgb_difference_8bit": largest_difference if shared else None,
    }


def summarize_quality(scores, label):
    stats = {"scored_frames": len(scores), "mean_psnr_db": statistics.fmean(
        float("inf") if frame["identical"] else frame["psnr"] for frame in scores)}
    for metric in ("ssim", "lpips", "vmaf"):
        stats[f"mean_{metric}"] = statistics.fmean(frame[metric] for frame in scores)
    print(f"QUALITY {label} {len(scores)} frames, PSNR {stats['mean_psnr_db']:.3f} dB, "
          f"SSIM {stats['mean_ssim']:.5f}, LPIPS {stats['mean_lpips']:.5f}, "
          f"VMAF {stats['mean_vmaf']:.3f}", flush=True)
    return stats


def write_results(path, rows):
    pending = path.with_suffix(".pending")
    with pending.open("w", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def run(args):
    bundle = args.bundle.expanduser().resolve()
    runtime = bundle / "runner" if (bundle / "runner/live_client.py").is_file() else bundle
    if not (runtime / "live_client.py").is_file():
        raise FileNotFoundError(f"Extract QS_3DGStream_VoD_Study.zip or specify --bundle: {bundle}")
    sys.path.insert(0, str(runtime))
    from live_client import render_fp32_reference

    study = args.study.expanduser().resolve()
    config = json.loads((study / "study_config.json").read_text())
    manifests = json.loads((study / "prepared/sequence_manifest.json").read_text())["manifests"]
    manifest_sets = {"prepared/sequence_manifest.json": manifests}
    frames, fps = config["frames"], config["fps"]
    width, height = config["size"]
    viewer = args.viewer.expanduser().resolve() if args.viewer else Path(config["viewer"])
    scene = Path(config["scene"])
    wire = Path(manifests["0"]["source_wire"]).parent
    with (study / "study.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No completed trials in {study / 'study.csv'}")
    trials = []
    for row in rows:
        trial = Path(row["trial"])
        if not trial.is_absolute():
            trial = study / trial
        tiers = played_tiers(trial / "live/apply_events.csv", frames)
        if not (trial / "received").is_dir():
            raise FileNotFoundError(trial / "received")
        from paired_groups import group_precision
        with (trial / "live/apply_events.csv").open(newline="") as stream:
            events = list(csv.DictReader(stream))
        for index, event in enumerate(events):
            actual = (group_precision(trial / "received" / f"NTC_{index:06}", index)
                      if "variants" in manifests[str(index)] else 4)
            # Older flat INT4 logs left this field blank; paired groups always declare it.
            reported = event["precision_bits"] or (4 if "variants" not in manifests[str(index)] else 0)
            if int(reported) != actual:
                raise ValueError(f"Applied precision disagrees with received P0 for update {index}")
        manifest_name = row.get("prepared_manifest") or "prepared/sequence_manifest.json"
        if manifest_name not in manifest_sets:
            manifest_sets[manifest_name] = json.loads((study / manifest_name).read_text())["manifests"]
        trials.append((row, trial, tiers, manifest_sets[manifest_name]))

    reference = args.reference_dir.expanduser().resolve() if args.reference_dir else None
    fp32 = args.fp32_dir.expanduser().resolve() if args.fp32_dir else scene / "NTCs"
    for index in range(frames):
        required = reference / image_name(index) if reference else fp32 / f"NTC_{index:06}.pth"
        if not required.is_file():
            raise FileNotFoundError(f"Missing aligned FP32 reference input: {required}")
    vmaf = args.vmaf or shutil.which("vmaf")
    score_sequence = None
    if not args.render_only:
        ffmpeg = args.ffmpeg
        if not ffmpeg:
            raise ValueError("FFmpeg is required for scoring")
        if not vmaf:
            filters = subprocess.check_output([ffmpeg, "-hide_banner", "-filters"], text=True)
            if " libvmaf " not in filters:
                raise ValueError("Use --vmaf /path/to/vmaf, FFmpeg with libvmaf, or --render-only")
        import lpips
        import skimage.metrics
        from evaluation_support import score_sequence

    output = args.out.expanduser().resolve() if args.out else study / "offline_quality"
    output.mkdir(parents=True, exist_ok=args.resume)
    # Never silently resume quality rows with a different scene, camera or runtime.
    camera_path = Path(os.environ['Q3DG_CAMERA_CONFIG']).resolve() if os.environ.get('Q3DG_CAMERA_CONFIG') else None
    fingerprint = dict(frames=frames, fps=fps, size=[width,height], scene=str(scene), viewer=str(viewer),
        camera_sha256=hashlib.sha256(camera_path.read_bytes()).hexdigest() if camera_path else None,
        runtime_sha256=hashlib.sha256((runtime / 'live_client.py').read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((study / 'prepared/sequence_manifest.json').read_bytes()).hexdigest(),
        reference_dir=str(reference) if reference else None, fp32_dir=str(fp32), vmaf=str(vmaf))
    context_file = output / 'SCORING_CONTEXT.json'
    if context_file.exists() and json.loads(context_file.read_text()) != fingerprint:
        raise ValueError('Scoring scene/camera/runtime changed; use a fresh quality output')
    context_file.write_text(json.dumps(fingerprint,indent=2)+'\n')
    replay_scratch = args.replay_scratch.expanduser().resolve() if args.replay_scratch else output
    if args.replay_scratch:
        replay_scratch.mkdir(parents=True, exist_ok=True)
        print(f"Temporary replay workspace {replay_scratch}", flush=True)
    table = output / "study_with_replay.csv"
    results = {}
    if args.resume and table.is_file():
        with table.open(newline="") as source:
            for previous in csv.DictReader(source):
                if int(previous["scored_frames"]) == frames:
                    results[previous["trial"]] = previous
    print(f"Replay output {output}", flush=True)
    frame_names = {image_name(index) for index in range(frames)}
    if reference is None:
        saved_reference = output / "fp32_reference/renders"
        if args.resume and saved_reference.is_dir():
            if {path.name for path in saved_reference.glob("frame_*.png")} != frame_names:
                old_reference = output / f"fp32_reference-interrupted-{time.time_ns()}"
                saved_reference.parent.rename(old_reference)
                print(f"Preserved incomplete reference in {old_reference}", flush=True)
                reference = render_fp32_reference(viewer, scene, wire, fp32, output / "fp32_reference",
                                                 frames, width, height)
            else:
                reference = saved_reference
        else:
            reference = render_fp32_reference(viewer, scene, wire, fp32, output / "fp32_reference",
                                             frames, width, height)
    for row, trial, tiers, trial_manifest in trials:
        label = trial.parent.name
        if row["trial"] in results:
            previous = results[row["trial"]]
            fields = set(row) - {"mean_psnr_db", "mean_ssim", "mean_lpips", "mean_vmaf"}
            if any(previous.get(field) != row[field] for field in fields):
                raise ValueError(f"Previously scored row disagrees with current study for {label}")
            print(f"ALREADY SCORED {label}", flush=True)
            continue
        folder = output / label
        replayed = folder / "renders"
        temporary_replay = None
        if args.resume and replayed.is_dir():
            if {path.name for path in replayed.glob("frame_*.png")} != frame_names:
                if not args.low_space:
                    raise ValueError(f"Incomplete saved trial replay in {replayed}")
                replayed = None
            else:
                print(f"REUSING {frames} replay frames for {label}", flush=True)
        elif not replayed.is_dir():
            replayed = None
        if replayed is None:
            if args.low_space:
                temporary_replay = tempfile.TemporaryDirectory(prefix="replay-", dir=replay_scratch)
                replay_folder = Path(temporary_replay.name)
            else:
                replay_folder = folder
            replayed = replay_tiers(trial / "received", label, replay_folder, viewer, scene, wire,
                                    trial_manifest, tiers, width, height)
        check = compare_saved_frames(trial / "live/renders", replayed, frames)
        print(f"REPLAY CHECK {trial.parent.name} {json.dumps(check)}", flush=True)
        if check["overlapping_captures"] == 0 or check["identical_captures"] != check["overlapping_captures"]:
            raise ValueError(f"Offline replay does not match the live captures for {label}")
        result = {**row, **check, "quality_source": "offline replay of applied tiers",
                  "scored_frames": 0,
                  "renders": "temporary replay removed after scoring" if temporary_replay else str(replayed),
                  "reference": str(reference)}
        if score_sequence:
            metrics = folder / "metrics"
            if args.resume and metrics.exists():
                metrics = Path(tempfile.mkdtemp(prefix="score-retry-", dir=folder)) / "metrics"
            scores = score_sequence(reference, replayed, metrics, args.ffmpeg, vmaf, fps)["frames"]
            result.update(summarize_quality(scores, label))
        results[row["trial"]] = result
        write_results(table, [results[item["trial"]] for item, _, _, _ in trials if item["trial"] in results])
        if temporary_replay:
            temporary_replay.cleanup()
    if args.full_reference:
        folder = output / "full_pqat_reference"
        images = replay_tiers(study / "prepared/send", folder.name, folder, viewer, scene, wire,
                              manifests, [4] * frames, width, height, source_precision=4)
        payload = sum(group["payload_bytes"] for frame in manifests.values()
                      for group in frame["variants"]["4"]["groups"].values())
        anchor = {"reference": "full PQAT P0–P3 against FP32", "scored_frames": 0,
                  "nominal_application_payload_bytes": payload,
                  "nominal_application_mbps": 8 * payload * fps / (frames * 1e6),
                  "renders": str(images), "fp32_renders": str(reference)}
        if score_sequence:
            scores = score_sequence(reference, images, folder / "metrics", args.ffmpeg, vmaf, fps)["frames"]
            anchor.update(summarize_quality(scores, folder.name))
        write_results(output / "quality_references.csv", [anchor])
    print(f"Results {output / 'study_with_replay.csv'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    default_bundle = HERE
    parser.add_argument("--bundle", type=Path, default=default_bundle)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--viewer", type=Path, help="Use the repository optional viewer")
    parser.add_argument("--camera-config", type=Path, help="Use the same validated camera as live capture")
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument("--reference-dir", type=Path, help="Exact cam00 PNG reference sequence")
    reference_group.add_argument("--fp32-dir", type=Path, help="Original NTCs, defaults to the scene's NTCs directory")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--vmaf", help="Path to the standalone VMAF executable")
    parser.add_argument("--render-only", action="store_true", help="Recover PNGs and verify existing captures")
    parser.add_argument("--resume", action="store_true", help="Reuse complete replay PNGs and completed quality rows")
    parser.add_argument("--low-space", action="store_true", help="Remove only newly rendered temporary frames after scoring")
    parser.add_argument("--replay-scratch", type=Path,
                        help="Write temporary trial replay images under this directory")
    parser.add_argument("--full-reference", action="store_true",
                        help="Also score a complete P0–P3 PQAT replay against the same FP32 reference")
    try:
        args = parser.parse_args()
        if args.resume and (args.render_only or args.full_reference):
            parser.error("--resume requires regular trial scoring")
        if args.low_space and (args.render_only or args.full_reference):
            parser.error("--low-space requires regular trial scoring")
        if args.replay_scratch and not args.low_space:
            parser.error("--replay-scratch requires --low-space")
        if args.camera_config:
            import os
            os.environ["Q3DG_CAMERA_CONFIG"] = str(args.camera_config.resolve())
        run(args)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        parser.exit(1, f"ERROR: {message or exc}\n")
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
