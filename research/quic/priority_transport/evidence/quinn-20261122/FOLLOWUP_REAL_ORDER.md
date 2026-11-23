# Real-order follow-up — 23 November 2026

The new archive resolves the earlier missing **INT4-5% priority orders and viewer code**. It does not contain the latest precision-switching Quinn code, the measured campus Wi-Fi trace, the 39-run results archive, actual cam00 calibration, or reference video/frames. No campus replay or new image-quality result is claimed here.

## Inputs verified

Archive Google Drive ID `19G_rue5LGwkBvFw2BJIiF5PSWbT6qRvX`, SHA-256 `780fb423a84c944e44bd23aab62183b2aae82314eff9edb05d25282a938d412b`. It is gzip/tar, despite the initial download filename. It contains 717 files: 299 order NPZs, 299 wire checkpoints and 119 viewer files; no CSV or Rust/Cargo source.

All 299 supplied checkpoint SHA-256 hashes match the earlier INT4_keep05 data. Every order matches the retained block IDs and has finite, nonincreasing scores and the correct update index. Originals are preserved. Portable copies change only the machine-specific `source_wire` path; both hashes and the original paths are recorded in `order-audit.json` on the server.

CPU reconstruction on updates 0,114,298 verified all 12 cumulative stages. They contain 389/778/1167/1556 retained blocks and decode to 1,991,680 finite weights. Received weights match the source exactly; unreceived weights are zero. Full reconstruction matches the source state hash. This verifies numerical reconstruction, not image quality.

## Measured network tests

30 runs completed with the real order files, exact group hashes, and equal selected/received unique payload bytes. Each used all 299 updates at a **30 Hz target**. Half-second connection-relative deadline reserve; no graphical playback or actual startup/stall measurement. GPU cost remains the configured 17 ms placeholder.

24 controlled runs: three repeats × two methods (P0-first, strict) × equal/tiered Quinn priority × full/adaptive groups. Rate starts at 28 Mbit/s, then changes to 70 at 3 s, 16.8 at 5 s, and 28 at 7 s. Every run recorded all three in-place queue changes. The estimator ceiling was 70 Mbit/s. Delay is 10 ms on shared private loopback. It affects both directions in one shared queue, not two independent remote links. This reproduces the supplied rate schedule, not the unavailable measured campus trace.

All 24 controlled runs had zero selected-group deadline misses. The current sender changes **tiers of INT4_keep05 only**; it does not switch INT2/INT4 representations. Full staged payload needs 15.358 Mbit/s, below even the 16.8 Mbit/s phase before overhead. Therefore this scenario does not demonstrate precision adaptation or an improvement in quality/startup/stalls.

Six single-repeat controls use the earlier clean,5Mbit/s,and5Mbit/s+20ms+2%loss profiles, P0-first and tiered Quinn priorities:

| Network | Groups | Selected payload MB | Late updates /299 |
| --- | --- | ---: | ---: |
| clean | full | 19.133 | 0 |
| clean | adaptive | 19.133 | 0 |
| bandwidth | full | 19.133 | 299 |
| bandwidth | adaptive | 5.161 | 0 |
| loss | full | 19.133 | 299 |
| loss | adaptive | 5.161 | 156 |

The loss run with only P0 missed 156 deadlines; the earlier address-order control missed 146. Independent random-loss realizations and one repeat do not establish an effect of saliency ordering. Its intended benefit is visual and still needs matched-view rendering.

## Representation sizing and estimator timing

[CSV](followup-results.csv) includes all 30 run summaries and seven model-file sizes. **Serialized checkpoint bytes differ from staged wire bytes**: INT4_keep05 is 66,741 bytes as a `.pth` file (16.018Mbit/s at 30 Hz), but about 63,991 bytes as four staged groups (15.358Mbit/s). Do not use the whole-file sizing table to infer a downgrade in the staged test. INT4_keep20 whole files require about 60.818 Mbit/s; INT2_keep05 requires 10.473 Mbit/s, before transport overhead. The 70 Mbit/s ceiling is a chosen budget, not a universal requirement.

Receiver feedback is configured by the application at 100 ms; Quinn stats are sampled at each sender decision. Actual decision intervals are in the CSV. Existing receiver logs print only every tenth feedback message, so they cannot measure every individual delivery interval. No intrinsic QUIC bandwidth-update period is established. Per-update estimates, RTT/cwnd/backlog and chosen tiers are saved in server `selector-samples.csv` (8,970 rows).

## Camera and remaining work

The viewer archive has a Windows command for manually saving a chosen view, but no saved camera JSON or verified cam00 metadata. Neither the supplied archive nor the complete scene archive supplies the original cam00 video/calibration. Hard-coded client camera identity is now explicitly marked unverified; the evaluator no longer assumes it is cam00. Renderer reconstruction settings must also match between FP32 and compressed references. No new GPU quality result is reported.

Pending inputs: latest precision-and-tier selector implementation; campus trace and exact settings; 39-run archive with estimator/queue/playback logs; actual cam00 pose/intrinsics and aligned FP32/reference frames. These are needed to reproduce the supplied study rather than approximate it.

Verification: 50 focused Python checks passed, including isolated netem regression. Full-repository collection still has the previously documented progressive receiver error. Raw inputs/logs/CSVs stay at `/local/users/abeigi/q3dg-publio-followup-20261123`; no assets or experiments ran on the Mac. This follow-up adds only one report and one summary CSV to the evidence folder.
