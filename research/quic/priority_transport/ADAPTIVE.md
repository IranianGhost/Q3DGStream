# Paired INT2/INT4 playback over QUIC

This path integrates Públio's supplied startup-study runtime into the existing
Quinn experiment. It uses real compressed NTCs, actual reliable QUIC streams,
and the CUDA/OpenGL viewer in `viewer/`. It does not use MoQT.

## What happens for one update

1. The receiver reports delivered bytes, playable buffer, the next playback
   deadline, and measured decode/inference/render time. The application requests
   feedback every 50 ms; this is our setting, not a QUIC standard interval.
2. Before submitting the next NTC, the sender chooses INT2 or INT4 and one to
   four cumulative groups. It uses the latest receiver feedback, path estimate and deadline or buffer
   selector. A choice is fixed for that NTC: it never combines INT2 and INT4 groups.
3. Application submission (`p0_first`, `concurrent`, or `amirali_strict`) controls
   admission order. Independently, `tiered` assigns Quinn stream priorities
   P0=30, P1=20, P2=10, P3=0; `equal` assigns all zero. Neither guarantees arrival
   order or creates bandwidth. Reliable retransmission can still miss a deadline.
   The deadline selector estimates arrival as now + outstanding-byte service time
   + new-payload service time + RTT/2 + client processing + twice observed RTT
   jitter. The candidate must also fit a one-tick byte budget. If no choice fits,
   it still sends the minimum INT2 P0: mandatory data can therefore still be late.
4. The receiver verifies each selected group's hash and precision. Playback starts
   after the requested contiguous prebuffer, applies cumulative updates in order,
   and stalls if the next required update is unavailable. Late enhancements do not
   alter an already presented state.

The default media limit is 60 buffered updates (2 seconds at the 30 Hz target).
It is distinct from the 12-update admission window and the legacy 16-chunk writer
admission limit used by `amirali_strict`. These are not QUIC's congestion window.

## Run on the prepared university server

From the repository root, with the [supplied reproduction bundle](https://drive.google.com/file/d/1kEgg-dlO702v1pIdvsoiXs4TcehoujDD/view)
extracted on Linux outside Git:

```bash
scripts/run_gpu1_quic.sh python research/quic/priority_transport/adaptive.py prepare \
  --resources /local/users/abeigi/q3dg-publio-reproduction-20261124/inputs/QS_3DGStream_Reproducibility \
  --out /local/users/abeigi/q3dg-adaptive-assets-20261124

scripts/run_gpu1_quic.sh python research/quic/priority_transport/adaptive.py run \
  --assets /local/users/abeigi/q3dg-adaptive-assets-20261124 \
  --out /local/users/abeigi/my-quic-run \
  --profile jitter --selector deadline --precision adaptive --priority tiered

scripts/run_gpu1_quic.sh python research/quic/priority_transport/adaptive.py score \
  --run /local/users/abeigi/my-quic-run \
  --vmaf /local/users/abeigi/q3dg-vmaf-build/tools/vmaf
```

Preparation verifies the pinned resource manifest, model/order inputs, and exact
staged group bytes against the recorded source manifest. Scene/model files stay
outside Git. A shorter smoke test needs a separate `prepare --frames 3` workspace
and `run --frames 3`. Existing run outputs are never overwritten.

`run --dry-run` prints the exact command. Presets are `campus`, `delay` (40 ms),
`jitter` (40±10 ms normal), `loss` (20 ms plus 2% random loss), `burst` (20 ms plus
Gilbert–Elliott 1%,25%,80%,0.1%), and `controlled` (28→70→16.8→28 Mbps at 3/5/7 s;
10 ms each direction). Campus presets retain the supplied shared-loopback
bottleneck and 1,000-packet queue. Rootless user/mount/network namespaces keep
traffic shaping off the host interfaces. No host sudo is needed.

`--selector fixed --precision int2 --tiers 1` selects the fixed INT2 P0 baseline.
`--estimator goodput` compares smoothed receiver delivery with the default hybrid
cwnd/RTT-and-goodput estimator. Low goodput alone can also mean an idle sender.

## Camera and evidence

Default camera constants preserve Públio's original study. Their pose matches
official cam00 calibration within 4.9e-9 in the supplied coordinates. Optional
`--camera-config evidence/reproduction-20261124/cam00.json` uses independently
calibrated horizontal/vertical focal lengths (the original assumes fx=fy).
The JSON is bound to the initial scene's SHA256 and copied into each run.
`camera_reference.py` reproduces calibration and video-frame alignment from the
pinned official 3DGStream training fixture and Neural 3D Video cam00 video.

Each run stores `SESSION.json`, per-update decisions/arrival times, transport and
network samples, playback events, and sampled PNG captures. `score` regenerates
all applied states and checks captured live pixels exactly before reporting
PSNR/SSIM/LPIPS/VMAF against matching FP32 renders. A supplied video reference is
a separate measure of model accuracy, not compression-only error.

Time live trials serially. `parallel_quality.py` only shards **offline** scoring;
never run it alongside timed GPU playback. Retain its plan and logs for resume.
`trajectory_quality.py` optionally avoids duplicate image scoring: it verifies
all applied group hashes/order and every sampled live pixel against a scored
primary trajectory. `SCORE_REUSE.json` and CSV fields disclose every reuse;
network/timing measurements remain independent. Direct `parallel_quality.py`
performs an independent full offline replay for every trial.

The 30 Hz value is a target. Completion alone does not prove sustained 30 FPS.
Startup begins after QUIC connection establishment, with the initial Gaussian
scene/GPU preloaded; handshake and scene download are excluded. Timing ends at GPU-render readiness,
not remote-desktop display. The pinned bundle uses one shared network
architecture and normalization box; all INT2/INT4 and FP32 updates were checked. This reproduction
omits dynamic Gaussian additions. Offline per-update scores do not penalize
frozen display time during a stall. GPU backends and VMAF versions must be stated
when comparing machines. Source provenance and local adaptations: `UPSTREAM.json`.
