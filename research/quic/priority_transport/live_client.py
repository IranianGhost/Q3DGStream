#!/usr/bin/env python3
"""Real CUDA 3DGS playback from QUIC-delivered, saliency-ranked NTC groups.

One NTC is a *cumulative* Gaussian transformation. Apply each NTC exactly once,
in sequence, with whichever consecutive priority groups arrived by its playout
tick. A missed NTC is held and applied later; cumulative updates are never
silently skipped. No source wire checkpoint is used to decode received blocks.

The initial source NTC is used ONLY to instantiate the same network/config and
normalization bounds as the actual viewer; its weights are replaced before the
first NTC inference. This prototype runs the receiver and renderer on the same
Linux host, in different processes (netns only isolates the QUIC network).
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import queue
import statistics
import sys
import threading
import time
import webbrowser

from stage_codec import assemble_state, read_group
from paired_groups import assemble as assemble_paired
from ntc_preparation import NTCPreparation, AsyncCUDAJobs, gpu_prefetch_guard_ms
from live_preview import LivePreview
from vod_playback import VodClock

HOME = Path.home()
CAM_POS = "0.03796185,-1.32009901,-0.27442019"
CAM_TARGET = "-0.03145422,-1.26137897,0.72143792"
CAM_UP = "0.01708074,-0.99804983,0.06003988"
WIRE = Path(
    "/data/3dgs_storage/QNTC-Stream-PQAT-Chunks/qoe_eval/"
    "lowbit299_v11_20261011T014600_039285Z/cases/int4_keep05/pqat/wire"
)


def atomic_json(path: Path, obj: dict):
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(temporary, path)


def percentile(values, q: float):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] * (high - index) + ordered[high] * (index - low) if low != high else ordered[low]


def highest_ready_tier(folder: Path, manifest: dict):
    """Only tiers P0..Pk available as *complete* atomically renamed files."""
    n = 0
    for t in range(4):
        if not (folder / f"P{t}.bin").is_file():
            break
        n += 1
    if n:
        # The actual sender may choose fewer tiers. Availability is the truth
        # at the tick, and higher tiers are never required to play P0.
        assert n <= 4
    return n


def received_state(folder: Path, frame_manifest: dict, tiers: int):
    """Reconstruct solely from bytes *actually received* over QUIC."""
    if tiers < 1 or tiers > 4:
        raise ValueError(f"Unsupported tier count {tiers}")
    if "variants" in frame_manifest:
        return assemble_paired(folder, frame_manifest, tiers)
    metadata = None
    records = {}
    for tier in range(tiers):
        m, new = read_group(folder, tier, frame_manifest["groups"][f"P{tier}"])
        if tier == 0:
            metadata = m
            if m["source_state_sha256"] != frame_manifest["source_state_sha256"]:
                raise ValueError("Group metadata does not match source tensor manifest")
        if records.keys() & new.keys():
            raise ValueError("Duplicate record across priority groups")
        records.update(new)
    return assemble_state(metadata, records, frame_manifest["retained_blocks"]), len(records)


def monitor_future(received_dir: Path, next_frame: int, end_frame: int, required_tiers=1):
    """Contiguous ready-to-apply NTC count, excluding missing predecessors."""
    count = 0
    for frame in range(next_frame, min(end_frame, next_frame + 128)):
        folder = received_dir / f"NTC_{frame:06}"
        if not all((folder / f"P{tier}.bin").is_file() for tier in range(required_tiers)):
            break
        count += 1
    return count


def image_writer(jobs: queue.Queue, errors: list, fast_png=False):
    try:
        if fast_png:
            from PIL import Image
        else:
            import imageio.v2 as imageio
        while True:
            job = jobs.get()
            if job is None:
                jobs.task_done()
                break
            file, rgb = job
            if fast_png:
                Image.fromarray(rgb).save(file, compress_level=1)
            else:
                imageio.imwrite(str(file), rgb)
            jobs.task_done()
    except BaseException as e:
        errors.append(str(e))


def make_scene_overlay(scene, reference_ntc, output):
    # Reuse one model so scene initialization never loads all 299 checkpoints.
    overlay = output / "viewer_overlay"
    ntc_dir = overlay / "NTCs"
    ntc_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in (
        (scene / "init_3dgs.ply", overlay / "init_3dgs.ply"),
        (scene / "NTCs/config.json", ntc_dir / "config.json"),
        (reference_ntc, ntc_dir / "NTC_000000.pth"),
    ):
        if not src.is_file():
            raise FileNotFoundError(src)
        if not dst.exists():
            dst.symlink_to(src)
    return overlay


@contextmanager
def scene_renderer(viewer, overlay, width, height):
    if not (viewer / "renderer_cuda.py").is_file():
        raise FileNotFoundError(f"Missing GPU renderer: {viewer}")
    os.environ["DUMP_FRAMES_DIR"] = ""
    os.environ["NTC_DEBUG"] = "0"
    os.environ["NTC_DEBUG_DXYZ"] = "0"
    sys.path.insert(0, str(viewer))
    import torch
    import util_gau
    import util_3dgstream
    from OpenGL import GL as gl
    from renderer import (HiddenGLContext, build_renderer, load_dynamic_scene,
                          make_camera, parse_vec3, capture_backbuffer_rgb)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; real GPU playback cannot be verified")
    initial = util_gau.load_ply(str(overlay / "init_3dgs.ply"))
    cam_args = argparse.Namespace(
        cam_pos=CAM_POS, cam_target=CAM_TARGET, cam_up=CAM_UP,
        cam_fov_deg=69.4258952958773, camera_mode="auto_bbox", view_axis="0,0,1",
        cam_distance_mult=2, forward_dir="0,0,1", center_offset="0,0,0",
    )
    camera = make_camera(width, height, initial, cam_args)
    camera.up = parse_vec3(CAM_UP)
    if camera_file := os.environ.get("Q3DG_CAMERA_CONFIG"):
        from camera_config import load_camera, apply_camera
        apply_camera(camera, load_camera(camera_file, overlay / "init_3dgs.ply"))
    camera.is_pose_dirty = True
    with HiddenGLContext(width, height, show_window=False):
        renderer = build_renderer(width, height)
        if not hasattr(renderer, "query_NTC"):
            raise RuntimeError("CUDARenderer required for NTC inference")
        renderer.ntc_rotation_mode = "current"
        renderer.ntc_translation_scale = 1.0
        renderer.use_ntc_mask = False
        renderer.ntc_debug_mask = False
        renderer.ntc_debug_dxyz = False
        load_dynamic_scene(renderer, camera, str(overlay), 2)
        if len(renderer.NTCs) != 1:
            raise RuntimeError("Expected exactly one reusable NTC model")
        yield renderer, torch, gl, util_3dgstream, capture_backbuffer_rgb


def render_fp32_reference(viewer, scene, wire, checkpoints, output, frames, width, height):
    """Render aligned references offline with the same renderer, camera and base geometry."""
    from PIL import Image

    images = output / "renders"
    images.mkdir(parents=True, exist_ok=False)
    overlay = make_scene_overlay(scene, wire / "NTC_000000.pth", output)
    with scene_renderer(viewer, overlay, width, height) as (renderer, torch, gl, loader, capture):
        model = renderer.NTCs[0]
        shape = model.state_dict()["model.params"].shape
        with torch.no_grad():
            for index in range(frames):
                path = checkpoints / f"NTC_{index:06}.pth"
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                params = loader._extract_state_dict(checkpoint).get("model.params")
                if not torch.is_tensor(params) or params.dtype != torch.float32 or params.shape != shape:
                    raise ValueError(f"Expected original FP32 model.params of shape {shape} in {path}")
                model.load_state_dict({"model.params": params}, strict=False)
                renderer.query_NTC(renderer.gaussians.xyz, 0)
                renderer.current_timestep = 0
                renderer.draw(0)
                torch.cuda.synchronize()
                gl.glFinish()
                Image.fromarray(capture(width, height)).save(images / f"frame_{index + 1:06}.png", compress_level=1)
                if (index + 1) % 50 == 0 or index + 1 == frames:
                    print(f"REFERENCE {index + 1}/{frames}", flush=True)
    return images


def run(args):
    if getattr(args, "camera_config", None):
        os.environ["Q3DG_CAMERA_CONFIG"] = str(args.camera_config.resolve())
    if args.start_frame != 0:
        raise ValueError("Live cumulative playback v1 requires --start-frame 0; no history bootstrap yet")
    if args.frames < 1 or args.fps <= 0 or args.playout_ms <= 0:
        raise ValueError("Invalid playback parameters")
    if args.capture_every < 0:
        raise ValueError("--capture-every must be >= 0")
    if args.preparation_mode not in ("none", "cpu", "dense", "async"):
        raise ValueError(f"Unknown preparation mode: {args.preparation_mode}")
    if args.prefetch_frames < 1 or args.prefetch_frames > 32:
        raise ValueError("--prefetch-frames must be 1..32")
    if args.target_tiers not in range(5):
        raise ValueError("--target-tiers must be 0 (adaptive) or 1..4")
    if args.show_window and not args.progressive:
        raise ValueError("--show-window requires --progressive")
    if args.vod_startup_frames and (args.progressive or not args.play_request_file):
        raise ValueError("VoD requires the request clock and one render per update")

    # Load only the real viewer, not copied/reimplemented CUDA kernels.
    viewer = args.viewer.expanduser().resolve()
    scene = args.scene_base.expanduser().resolve()
    reference_ntc = args.wire_dir.expanduser().resolve() / "NTC_000000.pth"

    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = output / "renders"
    if args.capture_every:
        images.mkdir(exist_ok=True)
    progressive_images = output / "progressive"
    if args.save_stages:
        progressive_images.mkdir(exist_ok=True)
    manifest = json.loads(args.manifest.read_text())["manifests"]
    for frame in range(args.frames):
        if str(frame) not in manifest:
            raise ValueError(f"Manifest is missing NTC {frame}")

    overlay = make_scene_overlay(scene, reference_ntc, output)

    ready = output / "ready.json"
    metrics_path = args.metrics_json.expanduser().resolve()
    epoch_file = args.clock_file.expanduser().resolve()
    outputs = []
    apply_events = []
    progressive_events = []
    dec_hist, infer_hist, render_hist = deque(maxlen=30), deque(maxlen=30), deque(maxlen=30)
    assembly_hist, dequant_hist, install_hist = deque(maxlen=30), deque(maxlen=30), deque(maxlen=30)
    actual_dequant_hist = deque(maxlen=30)
    gpu_prefetch_ms = 0.0
    gpu_prefetch_operations = 0
    gpu_preparations = []
    gpu_prep_by_key = {}
    cpu_cache_hits = 0
    dense_cache_hits = 0
    exact_tier_fallbacks = 0
    idle_prefetch_slips = 0
    guard_blocked_ticks = 0
    superseded_partial_checks = 0
    async_ready_hits = 0
    async_waited_at_tick = 0
    async_wait_ms_total = 0.0
    async_crossed_next_tick = 0
    async_enqueues = 0
    writer_errors = []
    captures = queue.Queue(maxsize=16)
    thread = None
    if args.capture_every or args.save_stages:
        thread = threading.Thread(target=image_writer, args=(captures, writer_errors, bool(args.vod_startup_frames)),
                                  daemon=True)
        thread.start()

    preview = LivePreview(args.frames, args.preview_port) if args.show_window else None
    if preview:
        preview.start()
        print(f"[PREVIEW] Open {preview.url} to watch all {args.frames} NTCs", flush=True)
        threading.Thread(target=webbrowser.open, args=(preview.url,), daemon=True).start()

    # Rendering stays on the CUDA/GL context. The browser displays its pixels.
    with scene_renderer(viewer, overlay, args.width, args.height) as (
            renderer, torch, gl, util_3dgstream, capture_backbuffer_rgb):
        # Prewarm the CUDA-OpenGL rasterizer while preserving the initial state.
        renderer.draw(0)
        torch.cuda.synchronize()
        gl.glFinish()
        atomic_json(ready, {"ready": True, "gpu": torch.cuda.get_device_name(0)})
        print("[GPU READY] CUDA/OpenGL scene loaded; awaiting QUIC connection clock", flush=True)

        timeout = time.monotonic() + args.clock_timeout
        while not epoch_file.is_file():
            if time.monotonic() > timeout:
                raise TimeoutError("QUIC receiver connection clock never appeared")
            time.sleep(0.005)
        epoch_unix = float(json.loads(epoch_file.read_text())["unix_ms"]) / 1000.0
        epoch_mono = time.monotonic() - (time.time() - epoch_unix)
        next_apply = args.start_frame
        displayed_tiers = 0
        playback_misses = 0
        capture_dropped = 0
        applied_total = 0
        vod = None
        if args.vod_startup_frames:
            request_time = json.loads(args.play_request_file.read_text())["monotonic_s"]
            vod = VodClock(request_time, args.fps, args.vod_startup_frames, args.frames, args.feedback_ms)

        def ready_frames():
            return monitor_future(args.received_dir, next_apply, args.start_frame + args.frames,
                                  required_tiers=args.target_tiers or 1)

        def publish_vod():
            buffered = ready_frames()
            next_due_unix_ms = (None if vod.next_due is None else
                                (epoch_unix + vod.next_due - epoch_mono) * 1000)
            report = {
                "status": "live", "updated_at_unix_ms": time.time() * 1000,
                "buffer_frames": buffered, "last_applied_ntc": next_apply - 1,
                "playback_deadline_misses": 0,
                "vod": {"phase": vod.phase, "next_ntc": next_apply, "ready_frames": buffered,
                        "next_due_unix_ms": next_due_unix_ms},
            }
            if dec_hist:
                report.update(decode_ms=percentile(dec_hist, .95),
                              ntc_inference_ms=percentile(infer_hist, .95),
                              render_ms=percentile(render_hist, .95))
            atomic_json(metrics_path, report)

        preparation = None
        async_gpu = None
        if args.preparation_mode != "none":
            preparation = NTCPreparation(
                args.received_dir, manifest, args.start_frame,
                args.start_frame + args.frames, highest_ready_tier, received_state,
                ahead=args.prefetch_frames)
            preparation.start()
        if args.preparation_mode == "async":
            async_gpu = AsyncCUDAJobs(
                torch, util_3dgstream._dequantize_sparse_int4_ntc_state_if_needed)

        # In the async arm, synchronizing the whole device would also wait for
        # *future* NTCs on the independent preparation stream, invalidating the
        # overlap measurement. Same phase boundaries, current stream only.
        sync_playout = (torch.cuda.current_stream().synchronize
                        if async_gpu is not None else torch.cuda.synchronize)

        try:
          if vod:
            vod.start(ready_frames, publish_vod)
          for tick in range(args.frames):
            frame = args.start_frame + tick
            deadline = (vod.next_due if vod else
                        epoch_mono + args.playout_ms / 1000.0 + tick / args.fps)
            remaining = deadline - time.monotonic()
            early_gpu_preps_this_tick = 0
            early_gpu_ms_this_tick = 0.0
            guard_blocked_this_tick = False
            if args.preparation_mode == "async":
                # The receiver has a 150ms playout buffer. Preparing *future*
                # NTCs (not only the one due now) makes this useful. Only the
                # CPU-assembled, exact selected tier may be submitted.
                while remaining > 0:
                    preparation.check_error()
                    async_gpu.retire(next_apply)
                    candidate = preparation.pending_dense(
                        min(args.frames - 1, frame + args.prefetch_frames - 1),
                        required_tiers=args.target_tiers,
                        excluded=frozenset(async_gpu.jobs))
                    if candidate is not None and len(async_gpu.jobs) < 2:
                        # Launch CUDA work only with enough wall-clock margin
                        # for the Python submission; completion is event-driven.
                        margin_ms = 4.0
                        if remaining * 1000 > margin_ms:
                            folder = args.received_dir / f"NTC_{candidate.frame:06}"
                            if highest_ready_tier(folder, manifest[str(candidate.frame)]) == candidate.tiers:
                                job = async_gpu.launch(candidate)
                                if job is not None:
                                    async_enqueues += 1
                                    gpu_prefetch_operations += 1
                                    early_gpu_preps_this_tick += 1
                                    record = {
                                        "ntc": candidate.frame, "tiers": candidate.tiers,
                                        "next_tick": frame,
                                        "prepare_start_ms": f"{(job.submitted_at-epoch_mono)*1000:.3f}",
                                        "prepare_finish_ms": "",
                                        "slack_at_start_ms": f"{(deadline-job.submitted_at)*1000:.3f}",
                                        "admission_guard_ms": f"{margin_ms:.3f}",
                                        "gpu_dequant_ms": "", "crossed_next_tick": 0,
                                        "used_at_playout": 0,
                                        "enqueue_ms": f"{job.enqueue_ms:.3f}",
                                        "ready_at_tick": 0, "waited_at_tick": 0,
                                        "wait_at_tick_ms": "", "gpu_device_ms": "",
                                        "job": job,
                                    }
                                    gpu_preparations.append(record)
                                    gpu_prep_by_key[job.key] = record
                            else:
                                superseded_partial_checks += 1
                        else:
                            guard_blocked_this_tick = True
                    elif candidate is not None:
                        guard_blocked_this_tick = True
                    # The host can sleep; completed CUDA events remain queryable.
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(remaining, .001))
                    remaining = deadline - time.monotonic()
                async_gpu.poll()
                for record in gpu_preparations:
                    if (record["next_tick"] == frame and
                            record["job"].completion_observed_at is None):
                        record["crossed_next_tick"] = 1
                        async_crossed_next_tick += 1
            elif args.preparation_mode == "dense":
                # Only this CUDA/GL-owning thread may allocate dense GPU state.
                # Do so when comfortably ahead of the next tick; NEVER install
                # weights or modify the visible Gaussian state before the tick.
                while remaining > 0:
                    preparation.check_error()
                    candidate = preparation.pending_dense(frame)
                    margin_ms = gpu_prefetch_guard_ms(actual_dequant_hist)
                    if candidate is not None and remaining * 1000 > margin_ms:
                        # If P1 has already arrived, avoid spending GPU time
                        # dequantizing an obsolete P0-only CPU snapshot.
                        folder = args.received_dir / f"NTC_{candidate.frame:06}"
                        current_tiers = highest_ready_tier(folder, manifest[str(candidate.frame)])
                        if current_tiers != candidate.tiers:
                            superseded_partial_checks += 1
                            time.sleep(min(remaining, .001))
                            remaining = deadline - time.monotonic()
                            continue
                        began = time.monotonic()
                        p0 = time.perf_counter()
                        candidate.dense = util_3dgstream._dequantize_sparse_int4_ntc_state_if_needed(
                            candidate.quantized)["model.params"]
                        torch.cuda.synchronize()
                        candidate.dequant_ms = (time.perf_counter() - p0) * 1000
                        actual_dequant_hist.append(candidate.dequant_ms)
                        gpu_prefetch_ms += candidate.dequant_ms
                        gpu_prefetch_operations += 1
                        early_gpu_preps_this_tick += 1
                        early_gpu_ms_this_tick += candidate.dequant_ms
                        finished = time.monotonic()
                        crossed = finished > deadline
                        gpu_record = {
                            "ntc": candidate.frame,
                            "tiers": candidate.tiers,
                            "next_tick": frame,
                            "prepare_start_ms": f"{(began-epoch_mono)*1000:.3f}",
                            "prepare_finish_ms": f"{(finished-epoch_mono)*1000:.3f}",
                            "slack_at_start_ms": f"{(deadline-began)*1000:.3f}",
                            "admission_guard_ms": f"{margin_ms:.3f}",
                            "gpu_dequant_ms": f"{candidate.dequant_ms:.3f}",
                            "crossed_next_tick": int(crossed),
                            "used_at_playout": 0,
                            "enqueue_ms": "", "ready_at_tick": 1,
                            "waited_at_tick": 0, "wait_at_tick_ms": "",
                            "gpu_device_ms": "",
                        }
                        gpu_preparations.append(gpu_record)
                        gpu_prep_by_key[(candidate.frame, candidate.tiers)] = gpu_record
                        if crossed:
                            idle_prefetch_slips += 1
                    else:
                        if candidate is not None:
                            guard_blocked_this_tick = True
                        time.sleep(min(remaining, .001))
                    remaining = deadline - time.monotonic()
            elif remaining > 0:
                time.sleep(remaining)
            if preparation is not None:
                preparation.check_error()
            guard_blocked_ticks += int(guard_blocked_this_tick)
            if vod:
                vod.wait_for_update(frame, ready_frames, publish_vod)
                deadline = vod.next_due
            tick_begin = time.monotonic()
            applied_this_tick = 0
            last_tiers = 0
            decode_total = infer_total = 0.0
            assembly_total = dequant_total = install_total = 0.0
            this_tick_events = []
            progressive_base = None
            next_folder = args.received_dir / f"NTC_{next_apply:06}"

            # If a previous NTC was late, catch up IN ORDER. Unseen cumulative
            # updates may not be skipped even if newer groups arrived first.
            while next_apply <= frame and (next_folder / "P0.bin").is_file():
                frame_manifest = manifest[str(next_apply)]
                tiers = highest_ready_tier(next_folder, frame_manifest)
                applied_ntc = next_apply
                applied_start_ms = (time.monotonic() - epoch_mono) * 1000.0
                d0 = time.perf_counter()
                prepared = preparation.get(applied_ntc, tiers) if preparation is not None else None
                if prepared is None:
                    if preparation is not None:
                        exact_tier_fallbacks += 1
                    quantized, _ = received_state(next_folder, frame_manifest, tiers)
                    assembly_ms = (time.perf_counter() - d0) * 1000.0
                else:
                    quantized = prepared.quantized
                    assembly_ms = 0.0  # Already assembled off the critical path.
                    cpu_cache_hits += 1
                q0 = time.perf_counter()
                async_job = async_gpu.get(applied_ntc, tiers) if async_gpu else None
                if async_job is not None:
                    dense, was_ready = async_gpu.consume(async_job)
                    # Current-stream synchronization includes an event wait
                    # only when its result is not already ready at this tick.
                    sync_playout()
                    dequant_ms = (time.perf_counter() - q0) * 1000
                    if was_ready:
                        async_ready_hits += 1
                        dense_cache_hits += 1
                    else:
                        async_waited_at_tick += 1
                        async_wait_ms_total += dequant_ms
                    rec = gpu_prep_by_key[(applied_ntc, tiers)]
                    rec["used_at_playout"] = 1
                    rec["ready_at_tick"] = int(was_ready)
                    rec["waited_at_tick"] = int(not was_ready)
                    rec["wait_at_tick_ms"] = f"{dequant_ms:.3f}"
                    async_gpu.poll()
                elif prepared is not None and prepared.dense is not None:
                    dense = prepared.dense
                    dequant_ms = 0.0  # Already dequantized off the critical path.
                    dense_cache_hits += 1
                    gpu_prep_by_key[(applied_ntc, tiers)]["used_at_playout"] = 1
                else:
                    dense = (quantized["model.params"] if "_precision_bits" in quantized else
                             util_3dgstream._dequantize_sparse_int4_ntc_state_if_needed(quantized)["model.params"])
                    # Separating these two GPU phases requires a sync between
                    # dequantization and installation, in EVERY comparison arm.
                    sync_playout()
                    dequant_ms = (time.perf_counter() - q0) * 1000.0
                    actual_dequant_hist.append(dequant_ms)
                l0 = time.perf_counter()
                renderer.NTCs[0].load_state_dict({"model.params": dense}, strict=False)
                load_call_ms = (time.perf_counter() - l0) * 1000.0
                s0 = time.perf_counter()
                sync_playout()
                load_sync_ms = (time.perf_counter() - s0) * 1000.0
                install_ms = load_call_ms + load_sync_ms
                installed_ms = (time.monotonic() - epoch_mono) * 1000.0
                decode_ms = (time.perf_counter() - d0) * 1000.0
                t0 = time.perf_counter()
                if args.progressive and applied_ntc == frame:
                    # Refinements must start from the same pre-NTC Gaussians.
                    # Applying a cumulative transformation twice would be wrong.
                    progressive_base = (renderer.gaussians.xyz.clone(),
                                        renderer.gaussians.rot.clone())
                renderer.query_NTC(renderer.gaussians.xyz, 0)
                sync_playout()
                inference_ready_ms = (time.monotonic() - epoch_mono) * 1000.0
                inference_ms = (time.perf_counter() - t0) * 1000.0
                event = {
                    "ntc": applied_ntc, "tick": frame, "applied_tiers": tiers,
                    "precision_bits": quantized.get("_precision_bits", 4),
                    "scheduled_ms": f"{(deadline-epoch_mono)*1000:.3f}",
                    "apply_start_ms": f"{applied_start_ms:.3f}",
                    "weights_installed_ms": f"{installed_ms:.3f}",
                    "inference_ready_ms": f"{inference_ready_ms:.3f}",
                    "render_ready_ms": "", "render_display_deadline_ms": "",
                    "rendered_this_tick": 0,
                    "cpu_cache_hit": int(prepared is not None),
                    "dense_cache_hit": int(
                        (async_job is not None and was_ready) or
                        (prepared is not None and prepared.dense is not None)),
                    "async_gpu_waited": int(async_job is not None and not was_ready),
                    "group_assembly_ms": f"{assembly_ms:.3f}",
                    "dequantization_ms": f"{dequant_ms:.3f}",
                    "weight_load_call_ms": f"{load_call_ms:.3f}",
                    "weight_load_sync_ms": f"{load_sync_ms:.3f}",
                    "prepared_cpu_assembly_ms": f"{prepared.assembly_ms:.3f}" if prepared else "",
                    "prepared_gpu_dequant_ms": (f"{async_job.gpu_ms:.3f}" if async_job and async_job.gpu_ms is not None
                                                  else f"{prepared.dequant_ms:.3f}" if prepared and prepared.dense is not None else ""),
                }
                apply_events.append(event)
                this_tick_events.append(event)
                dec_hist.append(decode_ms)
                assembly_hist.append(assembly_ms)
                dequant_hist.append(dequant_ms)
                install_hist.append(install_ms)
                infer_hist.append(inference_ms)
                decode_total += decode_ms
                assembly_total += assembly_ms
                dequant_total += dequant_ms
                install_total += install_ms
                infer_total += inference_ms
                last_tiers = tiers
                next_apply += 1
                if preparation is not None:
                    preparation.advance(next_apply)
                if async_gpu is not None:
                    async_gpu.retire(next_apply)
                applied_total += 1
                applied_this_tick += 1
                next_folder = args.received_dir / f"NTC_{next_apply:06}"

            last_applied = next_apply - 1
            if applied_this_tick:
                displayed_tiers = last_tiers
            if last_applied < frame:
                playback_misses += 1
            t0 = time.perf_counter()
            # Draw the *already-mutated* Gaussian state without applying NTC again.
            renderer.current_timestep = 0
            renderer.draw(0)
            sync_playout()
            gl.glFinish()
            render_ms = (time.perf_counter() - t0) * 1000.0
            render_hist.append(render_ms)
            # Mark completion at GL finish. PNG readback is optional and should
            # never pollute the model's render-ready timestamp.
            render_ready = time.monotonic()
            if vod:
                if applied_this_tick != 1:
                    raise RuntimeError(f"VoD must render update {frame} exactly once")
                vod.advance(render_ready)
                publish_vod()
            capture_ms = 0.0
            if args.capture_every and (tick % args.capture_every == 0):
                c0 = time.perf_counter()
                rgb = capture_backbuffer_rgb(args.width, args.height)
                capture_ms = (time.perf_counter() - c0) * 1000.0
                if not writer_errors:
                    try:
                        captures.put_nowait((images / f"frame_{frame + 1:06}.png", rgb))
                    except queue.Full:
                        capture_dropped += 1
            finish = time.monotonic()
            if this_tick_events:
                # Only the LAST cumulative NTC in a catch-up tick is actually
                # drawn; intermediate states were inferred but not displayed.
                last_event = this_tick_events[-1]
                last_event["rendered_this_tick"] = 1
                last_event["render_ready_ms"] = f"{(render_ready-epoch_mono)*1000:.3f}"
                last_event["render_display_deadline_ms"] = (
                    f"{(deadline-epoch_mono)*1000 + 1000.0 / args.fps:.3f}")
            if progressive_base is not None:
                # P0 is already displayed. Refine the SAME frame as P1-P3
                # arrive, stopping at the next scheduled frame.
                active_tiers = last_tiers
                folder = args.received_dir / f"NTC_{frame:06}"
                frame_manifest = manifest[str(frame)]
                stop_at = deadline + 1.0 / args.fps

                def record_preview(tiers):
                    path = progressive_images / f"NTC_{frame:06}_P{tiers-1}.png"
                    elapsed = (time.monotonic() - epoch_mono) * 1000.0
                    if preview or args.save_stages:
                        rgb = capture_backbuffer_rgb(args.width, args.height)
                        if preview:
                            preview.publish(
                                rgb, frame, tiers - 1,
                                elapsed - frame * 1000.0 / args.fps,
                            )
                        if args.save_stages:
                            captures.put((path, rgb), timeout=10)
                    progressive_events.append({
                        "ntc": frame, "ready_through": f"P{tiers-1}",
                        "display_ms": f"{elapsed:.3f}",
                        "file": str(path) if args.save_stages else "",
                    })
                    if frame % 25 == 0:
                        print(f"[PROGRESSIVE] NTC {frame+1}/{args.frames} "
                              f"P0..P{tiers-1} at {elapsed:.1f} ms", flush=True)

                record_preview(active_tiers)
                while time.monotonic() < stop_at and active_tiers < 4:
                    available = highest_ready_tier(folder, frame_manifest)
                    if available <= active_tiers:
                        time.sleep(min(0.005, max(0, stop_at - time.monotonic())))
                        continue
                    for tiers in range(active_tiers + 1, available + 1):
                        quantized, _ = received_state(folder, frame_manifest, tiers)
                        dense = (quantized["model.params"] if "_precision_bits" in quantized else
                                 util_3dgstream._dequantize_sparse_int4_ntc_state_if_needed(
                                     quantized)["model.params"])
                        renderer.NTCs[0].load_state_dict({"model.params": dense}, strict=False)
                        renderer.gaussians.xyz = progressive_base[0].clone()
                        renderer.gaussians.rot = progressive_base[1].clone()
                        renderer.query_NTC(renderer.gaussians.xyz, 0)
                        sync_playout()
                        renderer.draw(0)
                        sync_playout()
                        gl.glFinish()
                        record_preview(tiers)
                        active_tiers = tiers
                displayed_tiers = active_tiers
            elif preview and last_applied >= 0:
                # Keep browser display ticks moving even when a new P0 is late.
                rgb = capture_backbuffer_rgb(args.width, args.height)
                elapsed = (time.monotonic() - epoch_mono) * 1000.0
                preview.publish(
                    rgb, last_applied, max(0, displayed_tiers - 1),
                    elapsed - last_applied * 1000.0 / args.fps,
                )
            buffer_frames = ready_frames() if vod else monitor_future(
                args.received_dir, next_apply, args.start_frame + args.frames)
            updated = {
                "status": "live", "updated_at_unix_ms": time.time() * 1000.0,
                "decode_ms": percentile(dec_hist, 0.95) or 0,
                "ntc_inference_ms": percentile(infer_hist, 0.95) or 0,
                "render_ms": percentile(render_hist, 0.95) or 0,
                "buffer_frames": buffer_frames,
                "last_applied_ntc": last_applied,
                "playback_deadline_misses": playback_misses,
                "rendered_frames": tick + 1,
            }
            # Feedback gets live p95 timings after initial observed inference;
            # until then the Rust sender uses configured conservative defaults.
            if not vod and dec_hist and (tick == 0 or tick % max(1, round(args.fps * args.feedback_ms / 1000)) == 0):
                atomic_json(metrics_path, updated)
            outputs.append({
                "ntc": frame, "scheduled_ms": f"{(deadline-epoch_mono)*1000:.3f}",
                "tick_start_ms": f"{(tick_begin-epoch_mono)*1000:.3f}",
                "tick_lag_ms": f"{max(0, (tick_begin-deadline)*1000):.3f}",
                "tick_finish_ms": f"{(finish-epoch_mono)*1000:.3f}",
                "applied_up_to_ntc": last_applied, "applied_this_tick": applied_this_tick,
                "applied_tiers_latest": last_tiers, "age_frames": frame-last_applied,
                "played_current_ntc": int(last_applied == frame),
                "decode_total_ms": f"{decode_total:.3f}",
                "assembly_total_ms": f"{assembly_total:.3f}",
                "dequant_total_ms": f"{dequant_total:.3f}",
                "weight_install_total_ms": f"{install_total:.3f}",
                "inference_total_ms": f"{infer_total:.3f}",
                "render_ms": f"{render_ms:.3f}", "capture_ms": f"{capture_ms:.3f}",
                "buffer_frames": buffer_frames,
                "total_gpu_work_ms": f"{decode_total + infer_total + render_ms:.3f}",
                "tick_wall_ms": f"{(finish-tick_begin)*1000:.3f}",
                "tick_overrun": int(finish > deadline + 1 / args.fps),
                "early_gpu_preparations": early_gpu_preps_this_tick,
                "early_gpu_dequant_ms": f"{early_gpu_ms_this_tick:.3f}",
                "gpu_prep_guard_blocked": int(guard_blocked_this_tick),
            })
            if (tick + 1) % 25 == 0 or tick + 1 == args.frames:
                print(f"[LIVE] {tick+1}/{args.frames}, latest applied {last_applied}, "
                      f"held ticks {playback_misses}, buffer {buffer_frames}", flush=True)

        finally:
            if preparation is not None:
                preparation.stop()
            if async_gpu is not None:
                async_gpu.finish_logging()
                for record in gpu_preparations:
                    job = record["job"]
                    if job.completion_observed_at is not None:
                        record["prepare_finish_ms"] = f"{(job.completion_observed_at-epoch_mono)*1000:.3f}"
                        record["gpu_dequant_ms"] = f"{job.gpu_ms:.3f}"
                        record["gpu_device_ms"] = f"{job.gpu_ms:.3f}"
                        gpu_prefetch_ms += job.gpu_ms
                # Strip the private reference before serializing event records.
                for record in gpu_preparations:
                    del record["job"]

        # No further network mutations after the last playout deadline; mark
        # final viewer feedback even if the QUIC connection is already closed.
        if outputs and not vod:
            atomic_json(metrics_path, updated)

    if thread:
        # Finish PNG encoding outside timing-critical playout loop.
        if writer_errors:
            raise RuntimeError(f"Capture writer failed: {writer_errors}")
        try:
            captures.put(None, timeout=10)
        except queue.Full as exc:
            raise RuntimeError("Capture writer stopped draining its queue") from exc
        thread.join(timeout=60)
        if thread.is_alive() or writer_errors:
            raise RuntimeError(f"Capture writer failed: {writer_errors}")
    csv_path = output / "playback_frames.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(outputs[0]))
        writer.writeheader()
        writer.writerows(outputs)
    apply_path = output / "apply_events.csv"
    with apply_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "ntc", "tick", "applied_tiers", "precision_bits", "scheduled_ms", "apply_start_ms",
            "weights_installed_ms", "inference_ready_ms", "render_ready_ms",
            "render_display_deadline_ms", "rendered_this_tick",
            "cpu_cache_hit", "dense_cache_hit", "group_assembly_ms",
            "async_gpu_waited",
            "dequantization_ms", "weight_load_call_ms", "weight_load_sync_ms",
            "prepared_cpu_assembly_ms", "prepared_gpu_dequant_ms"))
        writer.writeheader()
        writer.writerows(apply_events)
    if args.progressive:
        with (output / "progressive_events.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("ntc", "ready_through", "display_ms", "file"))
            writer.writeheader()
            writer.writerows(progressive_events)
    gpu_prep_path = output / "gpu_preparation_events.csv"
    with gpu_prep_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "ntc", "tiers", "next_tick", "prepare_start_ms", "prepare_finish_ms",
            "slack_at_start_ms", "admission_guard_ms", "gpu_dequant_ms",
            "crossed_next_tick", "used_at_playout", "enqueue_ms",
            "ready_at_tick", "waited_at_tick", "wait_at_tick_ms", "gpu_device_ms"))
        writer.writeheader()
        writer.writerows(gpu_preparations)
    summary = {
        "frames": args.frames, "fps_target": args.fps, "playout_ms": args.playout_ms,
        "updates_applied": applied_total, "last_applied_ntc": next_apply-1,
        "instrumented_ntc_apply_events": len(apply_events),
        "ticks_with_stale_state": playback_misses,
        "ticks_current_ntc": args.frames-playback_misses,
        "tick_overruns": sum(int(r["tick_overrun"]) for r in outputs),
        "median_gpu_work_ms": statistics.median(float(r["total_gpu_work_ms"]) for r in outputs),
        "p95_gpu_work_ms": percentile([float(r["total_gpu_work_ms"]) for r in outputs], .95),
        "preparation_mode": args.preparation_mode,
        "cpu_cache_hits": cpu_cache_hits,
        "dense_cache_hits": dense_cache_hits,
        "exact_tier_fallbacks": exact_tier_fallbacks,
        "early_gpu_dequant_ops": gpu_prefetch_operations,
        "early_gpu_dequant_ms_total": round(gpu_prefetch_ms, 3),
        "early_gpu_prep_crossed_tick": idle_prefetch_slips,
        "early_gpu_preps_used_at_playout": sum(event["used_at_playout"] for event in gpu_preparations),
        "early_gpu_preps_not_used": sum(not event["used_at_playout"] for event in gpu_preparations),
        "gpu_prep_guard_blocked_ticks": guard_blocked_ticks,
        "gpu_prep_superseded_partial_checks": superseded_partial_checks,
        "async_gpu_prepares_enqueued": async_enqueues,
        "async_gpu_ready_at_tick": async_ready_hits,
        "async_gpu_waited_at_tick": async_waited_at_tick,
        "async_gpu_wait_ms_total": round(async_wait_ms_total, 3),
        "async_gpu_pending_at_next_tick": async_crossed_next_tick,
        "median_on_tick_assembly_ms": statistics.median(assembly_hist) if assembly_hist else None,
        "median_on_tick_dequant_ms": statistics.median(dequant_hist) if dequant_hist else None,
        "median_on_tick_install_ms": statistics.median(install_hist) if install_hist else None,
        "captured_frames": len(list(images.glob("frame_*.png"))) if args.capture_every else 0,
        "progressive_snapshots": len(progressive_events),
        "capture_dropped": capture_dropped,
        "quality_note": "Cumulative P0/P0..Pk trajectory; quality must be scored separately against exact FP32 reference",
        "timing_note": "Same phase boundaries in all arms. Async preparation uses a separate non-default CUDA stream and completion events; the playout CUDA stream, not the whole device, is synchronized at phase boundaries to avoid waiting for future frames. Event waits on the playout stream are measured on-tick. PNG readback excluded",
    }
    if vod:
        summary.pop("playout_ms")
        summary.update(vod.summary())
        atomic_json(output / "stall_events.json", {"events": vod.stalls})
    atomic_json(output / "playback_summary.json", summary)
    if preview:
        print(f"[PREVIEW] Finished {args.frames} NTCs. Closing in "
              f"{args.preview_linger_seconds:g}s.", flush=True)
        time.sleep(args.preview_linger_seconds)
        preview.close()
    print(f"[LIVE DONE] {json.dumps(summary)}", flush=True)
    print(f"[LIVE CSV] {csv_path}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--received-dir", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--clock-file", required=True, type=Path)
    p.add_argument("--metrics-json", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--scene-base", type=Path, default=Path(__file__).resolve().parents[3] / 'data/scenes/flame_steak')
    p.add_argument("--wire-dir", type=Path, default=WIRE)
    p.add_argument("--viewer", type=Path, default=Path(__file__).resolve().parent / 'viewer')
    p.add_argument("--camera-config", type=Path)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--frames", type=int, default=299)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--playout-ms", type=float, default=150)
    p.add_argument("--vod-startup-frames", type=int, default=0,
                   help="Start on this many contiguous P0 updates and pause on a missing P0")
    p.add_argument("--play-request-file", type=Path)
    p.add_argument("--feedback-ms", type=int, default=100)
    p.add_argument("--clock-timeout", type=float, default=120)
    p.add_argument("--width", type=int, default=1345)
    p.add_argument("--height", type=int, default=1008)
    p.add_argument("--capture-every", type=int, default=0,
                   help="0 disables captures; N saves every Nth tick (adds glReadPixels overhead)")
    p.add_argument("--progressive", action="store_true",
                   help="Capture P0 immediately, then upgrade the same NTC as P1-P3 arrive")
    p.add_argument("--show-window", action="store_true",
                   help="Open the live browser preview (requires a desktop)")
    p.add_argument("--preview-port", type=int, default=8765)
    p.add_argument("--preview-linger-seconds", type=float, default=10)
    p.add_argument("--save-stages", action="store_true",
                   help="Keep a PNG for every displayed P0-P3 stage")
    p.add_argument("--preparation-mode", choices=("none", "cpu", "dense", "async"), default="none",
                   help="none=tick-gated; cpu=early CPU; dense=v2 synchronous CUDA; async=v3 future CUDA event prefetch")
    p.add_argument("--target-tiers", type=int, default=0,
                   help="Exact tier count for GPU prefetch; 0 allows adaptive tiers")
    p.add_argument("--prefetch-frames", type=int, default=4,
                   help="Bounded lookahead of prepared cumulative NTCs")
    return p


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except Exception as e:
        sys.exit(f"LIVE ERROR: {type(e).__name__}: {e}")
