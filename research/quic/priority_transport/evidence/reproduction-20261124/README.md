# Reproducing the compressed NTC experiments

Públio's full reproduction bundle is now available on gpu1. It includes the actual INT2/INT4 models, saved saliency orders, campus Wi-Fi trace, adaptation code and completed image-quality results. The earlier report that these inputs and startup quality scores were missing is superseded by this evidence.

## What was tested

| Evidence | Scope |
| --- | --- |
| Supplied results, audited | 264 main trials and 7 earlier pilots. Recorded playback/transport summaries and all 81,029 per-image scores were checked. This is an archive consistency audit. |
| Fresh transport controls | Six 299-update QUIC transfers: fixed INT2 P0, fixed INT4 P0–P1 and adaptive selection, each on campus rate variation and random loss. No GPU playback in these six controls. |
| Fresh live compatibility study | 39 startup/adaptation trials on gpu1, with a real GPU receiver and the original experiment policies. The table below reports completion and quality-validation status. |

The main studies use one Flame Steak clip: 299 sequential NTC updates, approximately 10 seconds at a **30 Hz target**. Each update offers INT2 and INT4 keep20 representations, split into P0–P3. P0 provides the playable base; later groups refine it. The earlier keep05 controls use a different representation and cannot establish an improvement over keep20 by direct comparison.

| Representation | Application payload for 299 updates | Average payload rate at 30 Hz |
| --- | --- | --- |
| INT2 P0 only | 11.69 MB | 9.38 Mbit/s |
| INT2 P0–P3 | 46.60 MB | 37.41 Mbit/s |
| INT4 P0–P1 | 38.22 MB | 30.68 Mbit/s |
| INT4 P0–P3 | 76.38 MB | 61.31 Mbit/s |

These rates exclude the initial scene and transport overhead. The chosen 70 Mbit/s ceiling can accommodate larger representations; the controlled drop to 16.8 Mbit/s forces smaller choices.

The sender selects precision and group count for each update using recent feedback and the chosen policy. Application feedback is configured every **50 ms**; this is an experiment setting, not a mandatory QUIC interval. P0-first submission controls which writes the application starts first. Tiered Quinn priorities control transport scheduling when several streams have queued data. Neither guarantees delivery before a deadline.

The **buffer** selector uses the reserve of ready media to adjust the load. The **deadline** selector estimates how much data can arrive before the next update's playback deadline, allowing for receiver processing. Both can switch precision and group count.

The campus trace retains its original relative timestamps and is capped at 70 Mbit/s. Startup trials also use the controlled 28 → 70 → 16.8 → 28 Mbit/s profile, changing at 3, 5 and 7 seconds, with 10 ms added delay in each direction and no injected loss. The receiver starts after 200, 500 or 1,000 ms of buffered media. Two selectors, two traces, three buffer targets and three repetitions give 36 trials; three additional trials compare bandwidth estimators.

## What the supplied results support

| Comparison | Result | Interpretation |
| --- | --- | --- |
| Campus fixed INT4 P0–P1 versus joint adaptive precision/tiers; three repeats | Median startup **607 → 282 ms**, zero stalls in both. Median application bytes **38.22 → 44.55 MB** and PSNR **27.76 → 27.13 dB**. | Faster startup came with more bytes and lower average PSNR. This is a trade-off, not an improvement at equal cost and quality. |
| Ascending retained-block order versus saved saliency, fixed INT4 P0–P1 | Same **38,216,575 application bytes**: PSNR **27.60 → 27.76 dB**, SSIM **0.841 → 0.862**. | Weight selection/order improves the reconstructed sequence at this fixed payload. The fixed receiver waits for both selected groups. |
| Concurrent versus P0-first application submission, equal Quinn priorities | Median P0-first arrival **19.1% → 79.9%** on campus. Combining P0-first submission with tiered Quinn priorities reached 299/299 in each campus repeat. | Application submission and transport scheduling are separate controls. These are receiver group-ready ranks, not a packet-arrival guarantee under loss. |
| Adaptive playback with equal versus tiered Quinn priorities | Tiered priorities had zero stalls in all six delay/jitter trials; every random/burst-loss trial still stalled. | Earlier base delivery helps in some conditions. Priority does not create capacity or remove retransmission delay. Adaptive payloads can differ between arms. |
| Fixed INT2 P0 versus adaptive, random loss in the precision study; three repeats | Median stalled time **4.65 → 7.43 seconds**, despite median stall counts **37 → 36**. | A lower stall count can still mean longer freezing. Retain stalled duration alongside event count. |
| Startup selection sweep | All **39 archived trials** have quality scores and zero stalls. Only **2/18** tested policy/threshold contrasts meet the median byte/quality/stall tolerances; none of six equal-threshold contrasts do. | No general superiority of deadline-aware selection is established. Comparing different buffer thresholds changes two factors. |
| Hybrid versus smoothed receiver-goodput estimator; controlled trace, 500 ms target, three pairs | Median paired changes: startup **−25.5 ms**, sender UDP bytes **−54.0%**, PSNR **−0.294 dB**; zero stalls in both. | Goodput selected smaller representations. The bandwidth saving has a quality cost; it is not a uniform win across quality metrics. |

The study's declared comparison tolerances are 5% sender UDP bytes, 0.25 dB PSNR and 100 ms stalled time. The two passing contrasts each pass only two of their three individual pairs. Median paired changes are calculated within each repetition before aggregation; they need not equal the difference between arm medians.

The previously cited final campus pair (**580 versus 613 ms**) now has quality results: deadline versus buffer gave **27.10 versus 27.44 dB PSNR** and **45.27 versus 48.60 MB sender UDP payload**. It started sooner using fewer bytes, with lower quality; that single pair exceeds both the 5% byte and 0.25 dB quality tolerances.

## What changed on gpu1

The experiment policies, compressed payloads, saved orders and network topology were retained. All 299 generated group manifests match the recorded payload manifests after source-path rebasing. Network namespaces and veth links run inside private rootless user/mount/network namespaces, so the experiment does not alter the host network.

The available GV100 requires tiny-cuda-nn's CutlassMLP fallback rather than the source machine's FullyFusedMLP. The original tiny-cuda-nn commit was built in an isolated overlay. All 1,991,680 loaded parameters match exactly in a checkpoint probe, but inference differs numerically: maximum absolute difference 0.001465 in a 128-value probe. Other compatibility changes are EGL context selection, accepting the source's disabled rasterizer depth option, and local VMAF 3.0.0/cuDNN library resolution. The installed Torch/CUDA and rasterizer build differ from the supplied environment. Therefore these are **gpu1 compatibility reruns**, not bitwise replication of Públio's hardware.

The reference is a same-view cumulative FP32 render with dynamic additions omitted. It is not calibrated cam00 source-video ground truth. The GPU view was checked to contain the cooking scene rather than black images. Replay quality measures update-aligned rendered content; startup/stalls separately measure waiting. These scores do not measure quality throughout freezes or every intermediate group arrival. Startup is measured after scene/GPU preparation, not from the initial scene download.

## Fresh gpu1 results

All **39 live cases** applied and presented **299/299 updates**; all **11,661 rendered image pairs** were scored. The ten scheduled live captures in each case matched replay exactly (**390/390**), with zero dropped captures and zero buffer stalls. Each case recorded **1–4 playback-tick overruns**. The table gives medians of three repeats on this GPU; these absolute values are not compared directly with the source machine.

| Trace | Buffer target | Startup ms, buffer / deadline | PSNR dB, buffer / deadline | App MB, buffer / deadline |
| --- | --- | --- | --- | --- |
| controlled | 200 ms | 497.2 / 465.3 | 28.894 / 28.770 | 35.24 / 37.91 |
| controlled | 500 ms | 692.0 / 693.3 | 29.139 / 28.861 | 36.38 / 33.95 |
| controlled | 1000 ms | 905.5 / 877.7 | 29.042 / 28.835 | 37.64 / 33.52 |
| campus | 200 ms | 430.6 / 402.6 | 29.547 / 29.427 | 47.11 / 44.77 |
| campus | 500 ms | 566.0 / 556.5 | 29.614 / 29.225 | 46.30 / 43.77 |
| campus | 1000 ms | 700.4 / 699.0 | 29.607 / 29.170 | 47.51 / 44.20 |

The buffer target is **stored media duration**, not a mandatory wall-clock wait: one second of media can arrive in less than one second.

Here **3/18** policy/threshold contrasts meet the declared median tolerances; **one of six equal-threshold contrasts** does. At the 200 ms campus target, deadline selection gave median paired changes of **-33.85 ms startup**, **-4.84% sender UDP bytes** and **-0.078 dB PSNR**. Two of three individual pairs meet the tolerances. This is an isolated positive case, not a general ranking.

For the controlled 500 ms estimator comparison, goodput-only versus hybrid gave median paired changes of **-29.2 ms startup**, **-53.55% sender UDP bytes** and **-0.511 dB PSNR**, with zero stalls in both. P95 bottleneck backlog fell by **634 kB** and P95 sender RTT by **271 ms** (paired medians). Smaller transfers reduced queue pressure while sacrificing quality.

Controlled-profile rows have `all_target_ranges_seen_during_selection=False` because the supplied check requires a 38–<62 Mbit/s configured-rate interval; 28/70/16.8/28 never includes that interval. This flag is not proof of a missing controlled-profile phase. All 18 campus rows satisfy the three-bin check.

One timed trial, `controlled-500ms-rep2-deadline-goodput`, overlapped an accidental side quality probe. Its original files were quarantined outside the results archive and its row was replaced with an isolated rerun. That replacement occurred after the planned trial sequence. The final table contains one valid row per planned case.

The six transport-only controls all verified their selected payloads. Campus controls missed zero synthetic base deadlines. With random loss, base deadline misses were 261/299 for fixed INT2 P0, 299/299 for fixed INT4 P0–P1 and 184/299 for adaptive. Each arm has one loss realization: these counts do not establish a ranking, rendered FPS or live stall behavior. Their connection-relative synthetic deadlines differ from the live startup-buffer clock.

## Evidence and reproduction

- [Archived trial CSV](archived-trials.csv): one row per unique supplied result, with member paths into the five original ZIPs.
- [Fresh gpu1 CSV](gpu1-trials.csv): live reruns and separately labeled network-only controls; byte counts are application or UDP payloads as named, not complete Ethernet/IP wire traffic.
- Source bundle: [Drive download](https://drive.google.com/file/d/1kEgg-dlO702v1pIdvsoiXs4TcehoujDD/view). Size **2,610,133,391 bytes**; SHA-256 `6415bda495b011581d517a5be8105faddf50b4fce35c22ff3397792de156d546`. All **1,649 manifest entries** verified.

Raw data, logs, rootless launch helpers, compatibility patches, original ZIPs and the replacement-trial ledger stay at `/local/users/abeigi/q3dg-publio-reproduction-20261124`. Reproduction command on gpu1: `bash /local/users/abeigi/q3dg-publio-reproduction-20261124/run_startup39_compatibility.sh`. It resumes completed cases. No scene/model assets or runtimes were placed on the Mac.

These results concern one clip and a small set of network profiles. Completion of all 299 updates alone does not demonstrate sustained 30 FPS rendering. The source's full 264-trial factorial was audited, not independently rerun in full.
