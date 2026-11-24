# Compressed NTC playback: completed GPU1 reproduction

**Completed:** 264 main live-GPU trials, plus six separate tests of the integrated repository runtime. Every case applied all 299 updates. All **2,700 scheduled live captures** matched their reconstructed replay exactly, with zero dropped captures. Thirty hertz is the playback target, not a claim of uninterrupted 30 FPS.

## What changed

Públio’s paired INT2/INT4 runtime is integrated into the existing Quinn experiment. The repository now includes the optional real CUDA/OpenGL viewer, verified asset preparation, explicit network presets, receiver-feedback adaptation, calibrated camera support and rootless server launchers. The old flat INT4 path also passed a real three-update GPU/replay regression. Públio’s selectors and compression method are credited in [UPSTREAM.json](../../UPSTREAM.json); local integration and commands are explained in [ADAPTIVE.md](../../ADAPTIVE.md).

For each update, the sender chooses precision and how many cumulative weight groups to send. Each update has two-/four-bit weight versions. P0 is the approximate playable base; P1–P3 add weights. The decoder zero-fills omitted blocks to build a complete-shaped model tensor. Application submission controls when a stream is admitted. Quinn priority controls selection among pending streams. The receiver reconstructs the model and applies updates in sequence. Priority does not guarantee arrival order or a playback deadline.

## Independent trials and conditions

| Cohort | Main trials | Purpose |
| --- | ---: | --- |
| Precision/tiers | 135 | Nine fixed/adaptive policies × five profiles × three repeats |
| Adaptive QUIC priorities | 30 | Equal versus tiered priorities × five profiles × three repeats |
| Matched-payload ablations | 60 | Four new ordering/priority/saliency arms × five profiles × three repeats |
| Startup/estimator sweep | 39 | Previously validated buffer/deadline and estimator trials |

The ablations reuse 15 fixed INT4 P0–P1 controls from the precision cohort; those controls are **not counted twice**. The 225 newly run cases complete the main matrix alongside the prior 39. The separate repository validation uses campus/jitter/loss × equal/tiered priorities, once each; it validates integration, not a statistical ranking.

All use the actual 299 Flame Steak INT2/INT4 **keep20** checkpoints and saved saliency orders. Models are available ahead of playback (VoD). INT2 P0 is 11.69 MB per clip; INT4 P0–P1 is 38.22 MB; full INT4 is 76.38 MB. MB means 1,000,000 bytes. Initial scene delivery is excluded.

The measured campus trace retains its timestamps and is capped at 70 Mbit/s. Five profiles add: nothing (`campus`); 40 ms delay; normally distributed 40±10 ms delay (`jitter`); 20 ms delay plus 2% random loss; or 20 ms plus Gilbert–Elliott burst loss (1%,25%,80%,0.1%). These preserve the supplied shared-loopback bottleneck and 1,000-packet queue. The startup sweep instead uses campus/controlled rates with 10 ms each direction and zero injected loss.

Feedback is configured every 50 ms. The sender reads the latest feedback and Quinn path state before admitting another update. New choices affect future updates, not streams already in flight. The default playable-media limit is 60 updates; it is separate from the 12-update admission window, the legacy chunk-writer limit, and QUIC’s congestion window.

## New results

Each number below is the median of three independent network runs on GPU1. Stalled time is total freezing per approximately ten-second clip. Quality is update-aligned replay against matching FP32 checkpoint renders.

| Profile / policy | Startup ms | Stalled s | App MB | PSNR dB |
| --- | ---: | ---: | ---: | ---: |
| campus / int4-p1 | 720.7 | 0.00 | 38.22 | 29.84 |
| campus / adaptive | 394.1 | 0.00 | 44.16 | 29.09 |
| loss / int2-p0 | 790.5 | 7.31 | 11.69 | 27.86 |
| loss / adaptive | 711.0 | 7.19 | 12.34 | 27.99 |
| burst / int2-p0 | 719.8 | 5.33 | 11.69 | 27.86 |
| burst / adaptive | 766.5 | 6.67 | 12.11 | 28.00 |

Campus adaptation started 45.3% sooner, but used 15.5% more application bytes and lost 0.74 dB PSNR. This is a trade-off. Early approximate cumulative transforms can affect later states; more later bytes do not guarantee recovery from earlier approximation error.

| Adaptive profile | Stalled s, equal / tiered | P95 P0-ready ms, equal / tiered |
| --- | ---: | ---: |
| campus | 0.00 / 0.00 | 209.5 / 154.6 |
| delay | 0.00 / 0.00 | 540.3 / 217.4 |
| jitter | 0.67 / 0.00 | 579.1 / 207.3 |
| loss | 7.35 / 7.83 | 1061.5 / 1175.6 |
| burst | 4.41 / 6.96 | 916.4 / 1127.2 |

Tiered priority avoided stalls in all six delay/jitter runs. All random/burst-loss runs still stalled. These loss realizations are independent, so the table does not establish a general loss ranking. Adaptive arms can choose different payloads and quality; their full values remain in the CSV. P0-ready means a complete usable group at the receiver, not the first packet.

At a fixed **38,216,575 bytes**, saliency versus ascending retained-block order produced (this changes the selected weight subset; transport-only controls keep that subset fixed):

- PSNR (dB): **28.9864 → 29.8355**.
- SSIM: **0.8518 → 0.8676**.

With equal Quinn priorities and the same fixed payload, P0-first submission changed campus P0-first completion from **16.7% to 85.6%**. These fixed-payload controls isolate submission/saliency effects; adaptive QoE tests measure their interaction with representation selection.

The prior 39-case startup/estimator results remain part of this matrix. They did not establish a general deadline-selector advantage. In the controlled 500 ms estimator comparison, goodput-only reduced paired-median UDP payload by 53.55% and startup by 29.2 ms, with 0.511 dB lower PSNR and zero stalls in either arm. Lower goodput can mean an application-limited sender as well as a slow path.

## Camera and checks

The original hard-coded camera pose is now verified against official **cam00** COLMAP calibration: all four source cohorts match within 4.9e-9. The initial scene file is identical to the official training fixture (Git blob `823792edc9d815a77a6ebb693861c8b1e8385712`). Official video frame zero, after undistortion, matches the fixture at 51.30 dB; it is close, not byte-identical. Export code establishes NTC 0 → video frame 1. All 299 aligned video references were extracted on the server.

The 264 source-protocol cases retain the original projection, which approximates fx=fy (0.384% horizontal focal discrepancy). The six repository cases use [exact cam00 intrinsics](cam00.json). Compression comparisons use the same camera on both sides. Calibrated and original-camera quality values are not mixed as an algorithm improvement.

As a separate model/camera check, the calibrated FP32 replay versus source video scored **26.93 dB PSNR**, **0.8436 SSIM** over 299 frames. This includes model error and omitted dynamic additions; it is not compression-only quality.

Every selected group passed the transport hash checks; applied trajectories were checked again during scoring. Offline quality is computed once per byte-identical applied trajectory; every reused score is disclosed in `quality_reused_from` and server `SCORE_REUSE.json`, after checking all applied group bytes/order and every scheduled live image. Timing/network measurements are always independent. Direct full replay remains available through `parallel_quality.py`.

Validation: **77 Python checks**, the separately enabled real private-netns regression, and **9 Rust tests** passed. The root workflow also passed its 21 CPU tests, decoder demo and compile checks on both Python 3.10 and 3.12. The old flat INT4 GPU path and the chunk-admission scheduler with both paired precisions also passed live/replay checks.

## Scope and evidence

These are single-clip, same-host namespace **emulations** on the GV100. The prepared CutlassMLP/Torch/CUDA/rasterizer environment differs from Públio’s original machine, so absolute scores and startup times are not cross-hardware improvement claims. The original tiny-cuda-nn parameter probe is exact; its backend output probe differed by up to 0.001465. VMAF uses local v3.0.0; all new comparisons use that same build.

Startup measures the first GPU-render-ready update **after QUIC connection establishment**, with the scene/GPU preloaded; handshake, scene download and remote-desktop latency are excluded. Dynamic Gaussian additions are omitted, as in the supplied study. Replay quality does not penalize frozen display time: stalled duration and tick overruns are reported separately. Legacy connection-relative transport-deadline flags are not the buffered player’s actual presentation clock. Priority cannot overcome insufficient P0 capacity or retransmission delay; cumulative updates cannot simply be skipped.

- [Fresh GPU1 CSV](gpu1-trials.csv): 264 main rows plus six clearly labeled calibrated repository cases; raw trial paths and quality reuse are explicit.
- [Supplied result audit](archived-trials.csv): 264 main source cases plus seven historical pilots; these archive rows are distinct from fresh measurements.
- [Implementation and server commands](../../ADAPTIVE.md), [source provenance](../../UPSTREAM.json).

[Source bundle](https://drive.google.com/file/d/1kEgg-dlO702v1pIdvsoiXs4TcehoujDD/view) SHA256: `6415bda495b011581d517a5be8105faddf50b4fce35c22ff3397792de156d546`; 2,610,133,391 bytes and 1,649 verified manifest entries. Models, source archives and detailed trial CSVs/logs stay under `/local/users/abeigi/q3dg-publio-reproduction-20261124`. Additional camera/integration checks and the final summary are under `/local/users/abeigi/q3dg-completion-20261124`. Scene/model files were not downloaded to the Mac.

The previously reported startup trial that overlapped a side quality probe remains quarantined and replaced by its isolated rerun; the 39-case cohort contains only that valid replacement. Every newly timed phase ran separately from GPU scoring.

Portable defaults were centralized after the timed trials; study arguments already supplied explicit paths. A fresh three-update chunk/GPU/replay check passed after this cleanup.
