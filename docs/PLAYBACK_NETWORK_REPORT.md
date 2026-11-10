# QUIC playback buffering and loss evaluation — 10 November 2026

The sender now supports a 128-chunk admission ceiling (1 MiB payload), configurable
congestion control and datagram size, and a 120-second chunk recovery budget.
The actual priority adapter and original-scene sender expose the tuning options.
A media-time simulator implements contiguous prebuffer, playback, depletion and
refill. A buffer-aware controller selects future load using playable seconds and
acknowledged bytes/sec. Queued VOD work can be resized before transmission.

## What the repository actually contains

README.md and docs/FORMATS.md document original FP32 NTCs at 7,968,015 bytes,
keep05 packed checkpoints around 66,741 bytes, and four indexed stages totaling
62,296 bytes. The latter is 1,556 records ×40 bytes plus four 14-byte headers.
Only the original FP32 checkpoints exist locally. The compressed experiment is
an external input, as stated in docs/EXPERIMENTS.md. Remote main has not changed;
PR #1 remains open. The earlier review/data follow-up email has no new reply.

These scales are formats/representations, not a measured interchangeable quality
ladder. Experiments use byte-matched synthetic loads and separate real FP32 file
replays. They do not prune or truncate model files or claim equivalent images.

## Loss diagnosis and tuning

Two independent repetitions transmitted the same two original checkpoints under
3% random loss, 10 ms delay per direction and 12 Mbit/s shaping. At 1,200-byte
QUIC datagrams, median full transfer durations were:

|Admission ceiling / controller|Median seconds|
|---|---:|
|16 / Reno|53.93|
|128 / Reno|54.39|
|128 / Cubic|48.91|

The larger application window alone did not help. Samples show median congestion
windows around 8.2–9.4 KiB, with bytes in flight at the congestion window and RTT
around 24 ms. The application can admit 1 MiB, but QUIC's congestion controller
only permits a small fraction onto the path. Receiver processing was sub-ms in
previous diagnostics; it cannot explain the tens of seconds of runtime.

Additional single trials with 1,400-byte datagrams completed in 42.99 s (Reno)
and 44.60 s (Cubic). Increasing useful bytes per packet and choosing congestion
control are practical tuning candidates, not a universal repair. Trials have
independent random loss traces; the MTU comparison has only one trial per setting.
Do not present these observations as statistically proven improvements.

A burst-loss reproduction with a 15 s chunk timeout failed at 15.22 s. A 120 s
budget allowed a separate stochastic run to finish at 57.89 s. This fixes an
application policy that abandons slow recovery prematurely; it does not fix the
underlying loss or prove identical-trace recovery. Playback deadlines stay separate
from the reliable-transfer recovery budget. Default 1400-byte datagrams assume a
suitable path MTU; 1200 remains configurable for conservative compatibility.

## Playback semantics and adaptation

Each complete job represents a media interval. The experiments assume the initial
scene and decoder state are already available. Buffer occupancy is contiguous
media seconds beyond the playback position; later intervals cannot bridge a gap.
Playback starts when that occupancy reaches the target (or the complete shorter
video is buffered). Playback pauses on depletion and resumes after refill.
Receiver commit callbacks drive both online feedback and offline replay. Missing
or unproduced intervals censor playback at observation end; no extra unobserved
stall time is manufactured. Complete downloads may finish playback locally.

The buffer controller decreases future size when playable reserve is below half
the target (minimum guard 0.5 s), using delivery-rate/fps ×0.75 as a conservative
size budget. It probes upward after eight eligible samples when reserve is at
least the target. Current in-flight data is preserved. The simulation tracks
startup, stalls, unplayed duration and lateness against the initial media schedule.
VOD enqueue-to-commit latency is explicitly not used as a playback deadline.

## VOD comparison and initial-buffer sensitivity

Two retained VOD repetitions per policy use matching 1,400-byte datagrams and
Cubic, 8 s of media at 2 jobs/sec, and 128 chunk admissions. The path is good
24 Mbit/s /10 ms per direction, impaired at 6 s (1.5 Mbit/s, 40±10 ms and 2% loss),
then recovered at 12 s. The superseded 1,200-byte first baseline is retained
locally but excluded from comparisons.

With a 2-second initial media buffer, fixed-large startup was about 19.27 s with
about 33.06 s of rebuffering. Buffer-aware startup was about 3.25 s with no stalls.
This is a volume/prefetch trade-off: buffer-aware transferred mostly ~62 KiB jobs
and finished download before the impaired phase, while fixed-large sent ~7.97 MB
per job and necessarily encountered impairment. It is not evidence of equal image
quality or a controlled comparison of reaction to impairment.

Replay of the same fixed-large receiver commits shows the startup/stall trade-off:

|Initial playable media|Median startup s|Median rebuffer s|
|---|---:|---:|
|1 s|7.84|43.48|
|2 s|19.27|33.06|
|4 s|32.11|21.72|

Only the 2 s setting drives online control; 1/4 s are offline sensitivity replays,
not independently controlled adaptation runs. Larger initial buffers shift the
initial playback schedule but cannot create network capacity.

## Continuous-generation control

To expose the adaptive policy to all three network phases, two further runs
produce 18 s of media over time with the same 6/12 s phase changes. Both start
playback at approximately 3.24 s and complete all 18 s without a stall. Actual
transmitted levels move 2→0→1→0→1 (with different switch positions per trial).
This demonstrates both decrease and increase of future workload during transfer.
It does not validate rendered NTC quality. The largest offered size at 2 Hz is
about 127.5 Mbit/s, beyond the good path's 24 Mbit/s; returning to that level is
therefore not a sensible steady-state expectation.

## Selected metrics and limits

Use playable buffer seconds and observed delivery rate for load control; record
oldest-work age, congestion window/in-flight occupancy, RTT, declared loss and
receiver processing for diagnosis. Queue age under VOD prefetch describes deliberate
queued future content, so applying the old live queue threshold blindly is not
appropriate. QUIC-declared loss can include recovery/reordering effects and is not
the same as netem's actual drop counter.

18 retained trials: 17 successful transfers plus the expected 15 s burst failure;
19 physical runs including the superseded VOD baseline. Two repetitions are
exploratory, kernel loss traces differ, and some tests ran in separate isolated
containers concurrently. No significance or Internet-wide generalization claim.
Actual compressed NTC transfers, same-frame quality alternatives, dependency-aware
real model selection and GPU playback remain pending external inputs/validation.
Random-loss throughput still takes tens of seconds: tuning and timeout robustness
are implemented, but the loss-induced congestion response is not eliminated.

## Reproduction and evidence

See research/quic/priority_transport/README.md. Native demonstration:
`sh research/quic/priority_transport/demo_playback.sh` after activating the QUIC
virtualenv. Full isolated netem: `run_playback_tests.sh`. Analyze both output roots
with `python -m research.quic.priority_transport.playback_analyze ... --output ...`.
Compact raw samples, frame outcomes, source hashes, netem data, CSV and analysis
are under research/quic/priority_transport/evidence/playback-network. Full qlogs
and chunk events remain local under data/outputs/playback-final.

Scoped tests: 47 passed, including contiguous start, missing intervals, censored
stalls, interrupted production, receiver callback delivery and existing transport
and player-bridge tests. The previously documented full-repository progressive
receiver collection error is outside this change and remains unresolved.

The early main-matrix summary profile/seed labels retained CLI defaults; actual
conditions and repeated trials are identified by run names and netem/schedule
evidence. The runner now passes the profile label explicitly. No raw labels were
rewritten to hide this metadata issue.
