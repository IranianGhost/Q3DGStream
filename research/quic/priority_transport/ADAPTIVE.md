# Run the INT2/INT4 QUIC experiments

Use the prepared Linux/CUDA environment, Rust/Cargo and iproute2. The sender and GPU receiver use real QUIC; network shaping runs in isolated namespaces. Models remain outside Git.

From the repository root, set locations for the extracted experiment resources, prepared groups, a new run, and the VMAF executable:

```bash
Q3DG_RESOURCES=/path/to/experiment-resources
Q3DG_ASSETS=/path/to/prepared-groups
Q3DG_RUN=/path/to/new-run
Q3DG_VMAF=/path/to/vmaf

scripts/run_gpu1_quic.sh python research/quic/priority_transport/adaptive.py prepare \
  --resources "$Q3DG_RESOURCES" --out "$Q3DG_ASSETS"

scripts/run_gpu1_quic.sh python research/quic/priority_transport/adaptive.py run \
  --assets "$Q3DG_ASSETS" --out "$Q3DG_RUN" \
  --profile jitter --selector deadline --precision adaptive --priority tiered

scripts/run_gpu1_quic.sh python research/quic/priority_transport/adaptive.py score \
  --run "$Q3DG_RUN" --vmaf "$Q3DG_VMAF"
```

Preparation verifies the supported model inputs and generated groups. Existing run directories are not overwritten. For a three-update smoke test, use a separate `prepare --frames 3` workspace and `run --frames 3`. `run --dry-run` prints the command without starting a trial.

## Controls

| Option | Meaning |
| --- | --- |
| `--selector deadline` / `buffer` | Choose precision and group count using predicted delivery before the deadline or ready-media reserve |
| `--selector fixed --precision int2 --tiers 1` | Fixed INT2 P0 baseline; `--tiers 2` means P0 and P1 |
| `--submission p0_first` / `concurrent` | Start the base write first or admit groups concurrently |
| `--submission amirali_strict` | Priority-based 8 KiB chunk admission, with a default limit of 16 active chunk writers |
| `--priority equal` / `tiered` | Equal Quinn priorities or P0/P1/P2/P3 priorities 30/20/10/0 |
| `--estimator hybrid` / `goodput` | Combine the cwnd/RTT bound with delivery feedback, or use smoothed receiver goodput |

Choices are fixed per NTC; INT2 and INT4 groups are never mixed within an update. The deadline selector estimates queued/new-payload transfer, path delay, jitter and client work. If no representation fits, it still sends minimum P0. Consequently, priority and adaptation cannot guarantee every deadline.

The 60-update media buffer, 12-update admission window and 16-chunk writer limit serve different purposes; none is QUIC's congestion window. Late refinement groups do not change a state already applied to the scene.

Profiles are `campus`, `delay`, `jitter`, `loss`, `burst` and `controlled`; their exact conditions and results are in the [experiment report](evidence/reproduction-20261124/README.md). Add `--camera-config research/quic/priority_transport/evidence/reproduction-20261124/cam00.json` for the calibrated camera; leave it unset to retain the original study projection.

Each run saves decisions, group-ready times, network/QUIC samples, playback events and sampled images. `score` replays the applied models, verifies the saved live pixels and computes PSNR/SSIM/LPIPS/VMAF against matching FP32 renders. Run timed GPU trials serially and score afterwards. `parallel_quality.py` supports independent offline replays; `trajectory_quality.py` can reuse byte-identical trajectories with explicit checks and receipts.
