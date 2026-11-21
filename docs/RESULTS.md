# Results

The current experiments use the Flame Steak sequence with 299 NTC updates. Most progressive reconstruction measurements compare a partially received network against the **complete INT4 keep05 network** for the same frame.

## Checks completed

| Experiment | Result |
| --- | --- |
| Packed NTC stages | All 299 four-stage sets match their compressed wire checkpoints |
| Saved Gaussian states | 300 states reconstructed; all 14 reference checkpoints matched the stored XYZ and quaternion values |
| Saliency ordering | 299 updates compared with five random weight orderings |
| QUIC pilot | 30 NTC updates, 120 stages, and 30 additions files delivered locally |
| HashGrid block mapping | NTC 114 mapping checked against 12 sampled tiny-cuda-nn gradients |

The stage verification report is included at `research/results/wire_stage_verification.json`. A three-frame subset of the reconstruction metrics is in `research/results/metrics_113_115.csv`.

## Reconstruction with partial weights

In the offline 299-update experiment, we evaluated the NTC output after progressively receiving blocks in saliency order. Five independently randomized block orders provided the reference distribution.

At approximately 75% of the retained blocks received, saliency ordering reduced relative RMS translation error by **76.75%** in the red priority group and **78.28%** in the orange group, compared with the mean random-order error. When all four stages were reconstructed, the translation output matched the full compressed network.

These values measure the effect of *transmission order within the compressed model*. The INT4 keep05 source already prunes most of the dense network's parameter blocks, so receiving every transmitted stage does not reconstruct the original FP32 teacher.

For rotation, `qntc_299_priority_eval_rotation_v2.py` handles zero-length predicted quaternion updates using an explicit identity fallback. Rendered rotation comparisons should use the same convention as the live viewer.

## Model size

Each keep05 NTC contains **1,556 retained 64-parameter blocks**. The original dense network has 1,991,680 parameters, or approximately 7.97 MB in FP32.

A representative checkpoint occupies 66,741 bytes when stored in the compressed PyTorch wire representation. The four corresponding `.qblk` stage files occupy:

```text
1,556 blocks × 40 bytes + 4 headers × 14 bytes = 62,296 bytes
```

The difference reflects the file formats. These payload sizes exclude the initial Gaussian model, per-frame additions, connection and QUIC overhead, and any untransmitted metadata.

## Asynchronous NTC preparation

The Quinn receiver was tested with 120 NTC updates per run, three runs per
method. Every run used concurrent-priority submission, fixed P0+P1 weights,
30 FPS, 15 Mbps, 1% loss, about 20 ms RTT, and a 150 ms playout buffer.

| Preparation | GPU tick overruns / 360 ↓ | Median GPU time per tick |
| --- | ---: | ---: |
| On tick | 57 | 19.61–23.54 ms |
| CPU prefetch | 75 | 22.52–23.53 ms |
| Synchronous GPU prefetch | 67 | 19.84–23.87 ms |
| **Async GPU prefetch** | **15** | **10.63–11.14 ms** |

The time ranges are the median from each run. All 360 updates were applied
in every method, with no stale playback ticks. Async prefetch reduced tick
overruns by 74% compared with on-tick preparation. It moved dequantization
ahead of playback; it did not make the CUDA kernels themselves faster.

The [prefetch module](../research/quic/async_preparation/README.md) is kept
separate from the older QUIC viewer.

## Image-quality evaluations

Earlier Flame Steak results used a 50-frame rendering protocol and reported an FP32 reference of **33.938 dB PSNR**, **0.9591 SSIM**, and **0.0776 LPIPS**. A later 299-update experiment used a different renderer protocol and an approximately **30.612 dB FP32 reference**. Those numbers belong to different evaluations and should remain separate in comparisons.

## QUIC playback

The completed local QUIC pilot establishes that the frame additions and priority stages were transmitted and received. It uses application-level ordering of files, rather than a transport-internal QUIC priority scheduler.

Offline partial-NTC error and transport delivery answer different questions. The next live measurements are committed NTC updates per second, displayed image quality, latency under changing bandwidth and RTT, and behavior when stages arrive after their intended playback deadline.

Run instructions and external input requirements are in [Experiments](EXPERIMENTS.md) and [GPU validation](GPU_VALIDATION.md).
