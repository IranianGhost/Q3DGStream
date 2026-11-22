# QUIC delivery: review guide

This change connects bounded, prioritized QUIC delivery to the original CUDA viewer. The latest tests use Públio's actual checkpoints, not substitute payloads. Read this guide first; [results.csv](../research/quic/priority_transport/evidence/results.csv) contains all 165 summary rows, including failures. [Reproduction](QUIC_REPRODUCE.md) gives server commands.

## What changed and why

| Part | Behavior | Purpose |
| --- | --- | --- |
| Sender queue | FIFO, strict P0-first, or weighted admission; experiments use 8 KiB chunks and at most 128 outstanding chunks | Leave opportunities for urgent data when a future-frame backlog already exists |
| QUIC transport | A reliable stream per chunk; application acknowledgement returns admission capacity | Bound submitted work; QUIC still owns congestion control and retransmission |
| Receiver | Check chunk identity, offset and hash; assemble by file ID; verify complete-file hash, then rename into the cache | Chunks may arrive in different orders; the viewer never reads a half-written checkpoint |
| Viewer bridge | Load committed scene/model/additions through the original providers | Run the existing NTC inference and Gaussian rasterizer on the GPU |
| Playback clock | Wait for a contiguous startup reserve; hold the current state when the next required update is missing | A later recurrent NTC cannot skip an earlier required update |
| Size controller | Choose a complete, smaller/larger representation for the next update | Reduce offered load when delivery and playback fall behind |
| Read timeout | Configurable receiver budget, default 120 s instead of 30 s | Avoid abandoning a valid but slow reliable transfer prematurely |

Priority is **application admission before QUIC**, not a change to QUIC's internal stream/packet scheduler. Already-admitted lower-priority work can still be transmitted or retransmitted. Sender qlog shows STREAM-frame construction order; receiver completion timestamps show when complete files became usable. Neither is a physical NIC packet capture or a guaranteed arrival order.

## Actual input data

Públio's ZIP contains 299 checkpoints for each of seven representations (2,093 files). All were audited and decoded; their parameter shape is 1,991,680. The supplied FP32 files exactly match the existing Flame Steak sequence, allowing reuse of its initial scene and additions.

| Representation | Bytes per checkpoint |
| --- | ---: |
| FP32 | 7,968,015 |
| FP16 | 3,985,532 |
| INT8_keep20_PQAT | 477,154 |
| INT4_keep20_PQAT | 253,410 |
| INT4_keep10_PQAT | 128,949 |
| INT4_keep05_PQAT | 66,741 |
| INT2_keep05_PQAT | 43,637 |

The archive has block addresses, quantization metadata and one `mean_abs` label, but no numerical saliency scores or P0–P3 priority groups. Omitted sparse blocks decode as zero. Models are decoded only after the complete file is verified; this does not demonstrate rendering from an arbitrary subset of weights.

## 1. Transfer cost: compare identical files first

252/252 transfers passed: seven representations × six profiles × two methods × three repeats. Repeats use source updates 0, 114 and 298. Both methods use QUIC/CUBIC and 1,400-byte maximum datagrams. Whole-file sends a complete file in one stream; chunked uses 8 KiB/128 admissions. Timing starts after the TLS handshake and ends at complete-file receipt, with SHA-256 checked. It excludes scene setup and rendering.

Shaping uses Linux netem on loopback in a private network namespace, affecting traffic in both directions. Profiles: clean/unshaped; 5 Mbit/s; 40 ms delay + 5 Mbit/s; 40±10 ms normal jitter + 5 Mbit/s; 20 ms + random 2% loss + 5 Mbit/s; and 20 ms + burst loss + 5 Mbit/s. Burst model parameters are p=1%, r=25%, bad-state loss=80%, good-state loss=0.1%. Queue limit: 1,000 packets.

Median complete-file receipt time, seconds:

| Profile | FP32 whole | FP32 chunked | INT4-5% whole | INT4-5% chunked |
| --- | ---: | ---: | ---: | ---: |
| clean | 1.538 | 4.929 | 0.017 | 0.022 |
| bandwidth | 14.430 | 15.069 | 0.126 | 0.129 |
| delay | 14.615 | 15.337 | 0.261 | 0.267 |
| jitter | 14.970 | 15.717 | 0.314 | 0.319 |
| loss | 27.126 | 30.030 | 0.207 | 0.210 |
| burst | 25.202 | 33.798 | 0.175 | 0.175 |

**Interpretation:** smaller checkpoints substantially reduce delivery time. This is a representation-size benefit. A single ready file has no priority competition; chunking adds streams, headers, acknowledgements and processing, and is slower for FP32 here. These results do not claim that prioritization makes all transfers faster.

## 2. Priority: urgent files competing with future files

Each trial first releases 32 real INT4-5% future-update files at P3. After 150 ms, eight files each arrive at P0, P1 and P2: 56 files/504 chunks total. The initial 288 P3 chunks exceed the 128-admission window. Priorities represent a constructed frame-urgency workload, not supplied weight saliency. Each policy/profile has three repeats.

P0 time is measured from file release to complete-file receipt. Below is the median across runs of each run's P0 95th-percentile time, seconds:

| Profile | FIFO | Strict | Weighted | Fully completed trials, FIFO / Strict / Weighted |
| --- | ---: | ---: | ---: | --- |
| clean | 1.640 | 1.075 | 1.259 | 3/3 / 3/3 / 3/3 |
| bandwidth | 6.110 | 3.946 | 4.251 | 3/3 / 3/3 / 3/3 |
| delay | 6.175 | 4.129 | 4.463 | 3/3 / 3/3 / 3/3 |
| jitter | 6.218 | 4.278 | 4.690 | 3/3 / 3/3 / 3/3 |
| loss | 13.302 | 7.812 | 8.636 | 3/3 / 3/3 / 3/3 |
| burst | 12.482 | 7.429 | 7.630 | 3/3 / 3/3 / 2/3 |

**Interpretation:** urgent files finish sooner when admission is prioritized. On the 5 Mbit/s profile, P0's median first-STREAM rank improves from 324.5 (FIFO) to 164.5 (strict). Already-admitted P3 chunks explain why P0 is not first. Strict may starve lower priorities under continuous urgent arrivals; weighted allocates opportunities to lower classes.

The weighted burst-loss table uses only two successful runs: the third failed after 37.32 s with a receiver timeout. It remains recorded as a failure. After the timeout change, three independent weighted/burst follow-ups completed in 12.43–13.13 s; these are not replays of the original loss trace.

## 3. Original graphical viewer, actual checkpoint data

Seven GPU sanity checks each applied three updates with exact decoded parameters and finite, nonblack output from the original rasterizer. Fixed playback trials use eight updates/nine scene states; the 3% loss tests use six/seven. A scene state is the initial scene or the result of one NTC update, not a file, packet or display refresh. Scene/metadata adds 29.20 MB before updates.

CUBIC, 8 KiB/128 admissions; requested playback 30 updates/s; startup/refill reserve 0.1 s (three contiguous updates). Native profiles differ from the transfer tests: jitter is 60 Mbit/s, 15±5 ms; loss profiles are 60 Mbit/s, 10 ms, random 1% or 3% loss; queue limit 4,096 packets. Each fixed case has one trial.

| Profile | Model | Startup s | Buffer waiting s | Presented states | Complete |
| --- | --- | ---: | ---: | --- | --- |
| clean | FP32 | 14.55 | 10.49 | 9/9 | yes |
| clean | INT4_keep05_PQAT | 7.54 | 0.00 | 9/9 | yes |
| jitter | FP32 | 26.29 | 19.59 | 9/9 | yes |
| jitter | INT4_keep05_PQAT | 11.83 | 3.28 | 9/9 | yes |
| loss1 | FP32 | 71.77 | 56.77 | 9/9 | yes |
| loss1 | INT4_keep05_PQAT | 42.59 | 0.67 | 9/9 | yes |
| loss3 | FP32 | 133.08 | 36.18 | 4/7 | no: timeout |
| loss3 | INT4_keep05_PQAT | 71.09 | 0.92 | 7/7 | yes |

**Interpretation:** smaller real models also reduce startup and waiting in the native player. All 20 native trials are retained in the CSV: 19 completed. The incomplete FP32/3% loss case has censored waiting time and cannot be compared as a finished run. Even clean FP32 falls behind; smaller models reduce the load but do not establish real-time 30 Hz. Clean INT4-5% achieved 8.36 update transitions/s; zero buffer stalls alone does not prove the requested rate.

## 4. Adaptation: select real model sizes during changing conditions

48 updates at a deliberately paced 2 Hz, 2 s startup reserve, two repeats per policy. After metadata: 24 Mbit/s + 10 ms; at 6 s switch to 1.5 Mbit/s + 40±10 ms + 2% loss; at 12 s recover. Phase changes use wall time, independent of update completion. The ladder runs from INT2-5% to INT8-20%; every selected representation is for the same source update. There is no synthetic optional payload in these trials.

Feedback includes contiguous playable seconds, its timestamp, and the last completed active transfer's payload bytes / submit-to-application-ACK time. Feedback older than 1 s selects the smallest model. The controller reserves 20% of the delivery budget and subtracts additions bytes. Low playable reserve steps down; sufficient reserve and a safe budget permit one step up. Pacing idle time is excluded from active transfer rate; a held last-transfer estimate can still be stale, so its age is reported separately.

| Repeat | Policy | Startup s | Waiting s | NTC MB | All update MB |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | adaptive | 13.94 | 2.94 | 18.84 | 24.07 |
| 1 | fixed-high | 13.66 | 4.45 | 22.90 | 28.13 |
| 1 | fixed-low | 13.49 | 1.53 | 2.09 | 7.32 |
| 2 | adaptive | 13.72 | 3.08 | 18.93 | 24.16 |
| 2 | fixed-high | 13.68 | 3.74 | 22.90 | 28.13 |
| 2 | fixed-low | 13.63 | 2.05 | 2.09 | 7.32 |

All six runs presented 49/49 states. Adaptive used about 17–18% fewer NTC bytes and 14% fewer total update bytes than fixed-high, with less waiting in both repeats. Fixed-low waited less still, but uses a different model. Image quality equivalence is unproven, so this is a load/playback comparison, not an optimal-quality claim.

Example metric response, adaptive repeat 1 (phase medians):

| Phase | Playable s | RTT ms | Observed ACK goodput Mbit/s | Oldest due update age ms |
| --- | ---: | ---: | ---: | ---: |
| good | 2.5 | 75 | 8.26 | 173 |
| impaired | 1.0 | 482 | 1.17 | 3009 |
| recovered | 6.5 | 55 | 10.49 | 1237 |

Playable reserve and delivery rate guide adaptation; RTT and due-update backlog help explain a slowdown. Queue growth alone cannot distinguish network congestion, excessive source production or a slow receiver. Observed ACK goodput is a wall-time measurement, not the held last-transfer estimate. Receiver commit time excludes NTC decoding/inference and rendering.

## 5. Timeout fix and verification

A deterministic real-QUIC test sends a header, waits 35 s, then sends an actual 66,741-byte checkpoint. A 30 s receiver budget fails at 30.01 s; a 120 s budget commits at 35.06 s with the exact hash. The change prevents premature read abandonment; it does not fix congestion-control throughput or reproduce the original random-loss trace.

The focused transport/player suite previously passed 76 tests with one skipped because an interactive player owned its port. After cleanup, the focused transport/player suite passed 60 tests with one skipped (occupied player port); shell syntax and reviewer-document links also passed. Full-repository collection has a separate existing error in `progressive_299_v2/test_receiver_state_machine_v2.py`; a passing full-repository suite is not claimed.

Limits: loopback plus simulated networks, small independent random-loss samples, same-host file feedback rather than a deployed receiver-feedback protocol, no TCP comparison in this suite, no QUIC-internal priority patch, no proof of partial-weight rendering, equal visual quality or sustained 30 Hz.

Raw logs, qlog, hashes, screenshots and original reports remain on the GPU server at `/local/users/abeigi/q3dg-publio-eval-20261119`. Historical PR reports are preserved in `/local/users/abeigi/q3dg-pr-review-archive-20261122/previous-pr-evidence-and-notes.tar.gz`. They are removed from the PR to keep review focused; no scene or experiment data was moved to the Mac. Results above describe the Nov19 runs, not new runs of the cleaned documentation commit.
