#!/usr/bin/env python3
"""Score captured LIVE cumulative QUIC playback vs the same-view FP32 trajectory.

Only image filenames present in the live capture are evaluated; this is not the
old independent-stage single-frame test. Held/stale frames remain in the score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def score(args):
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity

    images = args.live_dir / "renders"
    files = sorted(images.glob("frame_*.png"))
    if not files:
        raise FileNotFoundError(f"No captured PNG frames in {images}. Use --capture-every 1 or 5")
    lpips_model = None
    device = None
    if not args.no_lpips:
        import torch
        import lpips
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    for live_image in files:
        reference = args.ref_dir / live_image.name
        if not reference.is_file():
            raise FileNotFoundError(f"Missing exact FP32 reference {reference}")
        a = np.asarray(Image.open(reference).convert("RGB"), dtype=np.float32) / 255.0
        b = np.asarray(Image.open(live_image).convert("RGB"), dtype=np.float32) / 255.0
        if a.shape != b.shape:
            raise ValueError(f"Reference/live image dimensions differ: {reference.name}")
        error = float(np.mean((a-b)**2))
        psnr = 10.0 * math.log10(1.0 / error) if error > 0 else float("inf")
        ssim = float(structural_similarity(a, b, channel_axis=2, data_range=1.0))
        perceptual = ""
        if lpips_model:
            import torch
            x = torch.from_numpy(a).permute(2, 0, 1)[None].to(device)*2-1
            y = torch.from_numpy(b).permute(2, 0, 1)[None].to(device)*2-1
            with torch.no_grad():
                perceptual = float(lpips_model(x,y).item())
        rows.append({"frame": int(live_image.stem.removeprefix("frame_")),
                     "psnr_vs_fp32": psnr, "ssim_vs_fp32": ssim,
                     "lpips_vs_fp32": perceptual})
        if len(rows)%25 == 0:
            print(f"[QOE] {len(rows)}/{len(files)}", flush=True)

    dst = args.live_dir / "live_quality.csv"
    with dst.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary={
        "frames_evaluated": len(rows), "mean_psnr": statistics.mean(r["psnr_vs_fp32"] for r in rows),
        "mean_ssim": statistics.mean(r["ssim_vs_fp32"] for r in rows),
        "mean_lpips": statistics.mean(r["lpips_vs_fp32"] for r in rows) if lpips_model else None,
        "reference": str(args.ref_dir),
        "camera_identity": "Not verified by this evaluator; requires external camera provenance",
        "protocol": "Real cumulative QUIC stage playback vs exact indexed, camera-matched FP32 rendered images",
        "caveat": "Captured-sample quality only; glReadPixels readback affects live timing. Compare same camera/protocol.",
    }
    (args.live_dir / "live_quality_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    print(f"CSV {dst}")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live-dir", type=Path, required=True,
                   help="results_deadline_.../priority/live")
    p.add_argument("--ref-dir", type=Path, required=True,
                   help="299-frame camera-matched FP32 reference directory")
    p.add_argument("--no-lpips", action="store_true")
    score(p.parse_args())
