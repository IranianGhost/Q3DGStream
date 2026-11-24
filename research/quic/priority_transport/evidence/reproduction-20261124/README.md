# QUIC streaming experiments

This evaluation separates three controls: **saliency** chooses which weights enter the transmitted prefix; **application submission** starts P0 before other writes; **Quinn priority** favors P0 among queued streams. P0 is the approximate base model; P1–P3 refine it. Missing blocks are zero-filled before inference, and cumulative NTC updates are applied in order.

## Method

- **Content:** 299 real Flame Steak NTC updates, approximately 10 seconds at a **30 Hz playback target**, with INT2/INT4 keep20 representations (20% retained parameter blocks).
- **Coverage:** 264 main trials: 135 precision/representation, 30 adaptive-priority, 60 ordering/saliency controls, and 39 startup/estimator trials. Comparisons use three repetitions; 15 reused controls are counted once. Six additional runs validate the integrated GPU receiver and are excluded from the comparison statistics.
- **Network:** Linux netem on one host, replaying measured campus Wi-Fi bandwidth capped at **70 Mbit/s**, with a 1,000-packet queue.

| Profile | Added impairment |
| --- | --- |
| Campus | Bandwidth variation only |
| Delay | 40 ms packet delay |
| Jitter | Normal delay: mean 40 ms, standard deviation 10 ms |
| Random loss | 20 ms delay and 2% packet loss |
| Burst loss | 20 ms delay; Gilbert–Elliott transitions good→bad 1%, bad→good 25%, with loss 80% in the bad state and 0.1% in the good state |

Precision, priority and ordering runs start after 15 contiguous updates are ready, allow 60 buffered updates, and use 50 ms feedback. The startup sweep varies the buffered-media target over 200/500/1,000 ms and also tests controlled rates of 28→70→16.8→28 Mbit/s at 3/5/7 seconds, with 10 ms delay each direction and no injected loss.

Transport-order controls keep the selected weights and application payload fixed. The saliency control changes the selected weight subset at the same size. Quality compares applied-state replay with matching-camera FP32 renders; higher PSNR/SSIM is better. Stalled duration is measured separately. Every run applied all 299 updates; all 2,700 scheduled live captures matched validated replay.

## Results and interpretation

Values below are GPU1 results. Unless identified as a paired change, numbers are medians of three runs. MB denotes application payload, excluding transport overhead. P95 is the 95th-percentile delay.

| Comparison | Result | Interpretation |
| --- | --- | --- |
| Fixed INT4 P0–P1 → adaptive, campus | Startup **721→394 ms**, zero stalls in both; payload **38.22→44.16 MB**; PSNR **29.84→29.09 dB** | Faster start costs **15.5% more bytes** and **0.74 dB** quality. It is not a win at equal cost/quality. |
| Block-index → saliency prefix | Same **38.22 MB**; PSNR **28.99→29.84 dB**, SSIM **0.852→0.868** | Choosing important weights improves reconstruction at equal size. |
| Concurrent → P0-first submission, equal Quinn priorities | P0 completed first in **16.7%→85.6%** of campus updates | Application write order changes receiver readiness even without different QUIC priorities. It does not guarantee timely playback. |
| Equal → tiered Quinn priorities with adaptive playback | Delay/jitter runs with stalls: **4/6→0/6**. Jitter P95 P0-ready latency: **579→207 ms** | Earlier base readiness helps in these conditions. Adaptive choices can also change bytes and quality; these are joint playback results. |
| Fixed INT2 P0 → adaptive under loss | Random-loss stalls **7.31→7.19 s**; burst-loss stalls **5.33→6.67 s** | No general loss-resilience improvement is established. Smaller representations still stall. |
| Hybrid → goodput-only estimator, controlled profile, 500 ms target | Paired-median changes: startup **−29 ms**, UDP payload **−53.55%**, PSNR **−0.51 dB**; zero stalls in both | A conservative estimate saves bandwidth at a quality cost. Low goodput can also reflect an idle sender. |

The startup sweep did not establish a general advantage for deadline-aware selection. Adaptation chooses feasible future representations; it cannot remove already queued data or guarantee that mandatory P0 data meets its deadline. Retransmission and the dependency on earlier NTC updates can still freeze playback. Early approximation errors can also affect later cumulative scene states.

## Scope and evidence

- **Timing:** startup ends when the first GPU-rendered update is ready, after connection establishment with the scene/GPU preloaded. Handshake, scene download and remote-desktop latency are excluded. Completing 299 updates does not demonstrate sustained 30 FPS.
- **Quality:** cam00 pose/alignment is verified. The main matrix retains the original projection; the six integration runs use exact intrinsics. Replay scores do not penalize freezes, and dynamic Gaussian additions are omitted. Byte-identical trajectories reuse scores only after model-byte/order and sampled-pixel checks; reuse is recorded per row.
- **Comparability:** one scene, one view, three independent loss realizations per arm, on GV100/CutlassMLP. Absolute values are not directly comparable with another GPU backend. VMAF uses v3.0.0.

[Trial CSV](gpu1-trials.csv) contains conditions, arrival metrics, playback, payload and quality for all 270 runs. Use `study` and `implementation` to separate comparisons from integration checks, and `p0_first_multitier_fraction` with `multitier_updates` when comparing adaptive arrival order. [Run instructions](../../ADAPTIVE.md).
