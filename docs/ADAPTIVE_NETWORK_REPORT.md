# Adaptive offered-load evaluation over QUIC

## Observed results and selected signals

All 27 matrix cases ran: 26 completed successfully; the real-checkpoint burst-loss
case hit the 15 s per-chunk application-ACK timeout after about 16.7 s, with neither
checkpoint committed. That failure is retained and excluded from successful-run
performance aggregates. It is not a successful transfer or a validated recovery.

Median outcomes over three dynamic trials (72 generated jobs, 1 s deadline):

|Policy|On-time jobs|On-time MiB|Dropped jobs|p95 latency (ms)|Peak backlog MiB|
|---|---:|---:|---:|---:|---:|
|Fixed low|61|1.91|0|1584|0.25|
|Fixed high|22|11.00|21|7082|4.50|
|Queue age|49|13.28|5|2786|1.25|
|RTT only|39|15.62|20|6911|3.41|
|Composite|48|13.38|4|2606|1.03|

Relative to fixed-high, composite increased median on-time jobs from 22 to 48,
reduced p95 latency by approximately 63%, and reduced dropped jobs from 21 to 4.
On-time payload grew by about 22%. This is a load-control result, not a video
quality gain. Fixed-low delivered more jobs on time than composite, but transferred
much less payload. RTT-only favored more large on-time bytes but left substantial
backlog. No policy met every deadline; composite p95 was still 2.61 s.

Recommended signals for the next iteration:

1. **Oldest job age for action:** in held-out fixed-high samples, age >500 ms
   predicted whether a job generated in the next second would miss its deadline
   with F1 0.95 (precision 0.91, recall 1.00). It is a pressure signal, not a network
   fault diagnosis: its false-positive rate against the configured impaired phase
   was 0.76, partly because backlog persists after the network recovers.
2. **RTT relative to baseline for network-change evidence:** the RTT alarm had
   precision 1.00, recall 0.87 and no false-positive samples against the two held-out
   dynamic phase labels. It first fired about 1.01 s after degradation. This does
   not generalize to all impairment types: the same alarm did not fire in the
   static high-delay, random-loss or burst-loss cases. RTT alone is insufficient.
3. **Acknowledged payload rate versus offered load, plus receiver processing
   time:** delivery deficit had next-second deadline F1 0.79 on fixed-high traces.
   Use it as supporting evidence, and use receiver processing measurements to
   distinguish a slow receiving application from path delay.

Do not select raw buffer occupancy or its growth alone. The 64 KiB QUIC-retained
buffer alarm had a 0.83 false-positive rate against held-out dynamic impairment
labels. The bounded application-ACK window is often occupied on a healthy path.
Growth can disappear once a queue is full even while delivery remains bad.
Queue growth's deadline-prediction F1 was only 0.46 in these runs.

The negative controls support that distinction. With injected receiver delay,
queue-age alarms occurred in about 55% of samples and application-ACK-delay alarms
in 100%, but RTT alarms in 0%. On a clean path with excess offered load, queue-age
alarms occurred in about 39% of samples and RTT alarms in 0%. These are exploratory
closed-loop controls, not accuracy estimates; delivery-deficit alarm fractions are
omitted for them because the policy changes the offered rate.

The composite policy is a reasonable experimental default for balancing timely
jobs, backlog and bytes, while queue-only is a simpler contender with similar
results. The current evidence does not establish statistically significant
superiority between those two. Keep the other metrics in logs for diagnosis;
validate thresholds on longer, different traces before a production decision.

### Actual NTC files

Each checkpoint was 7,968,015 bytes. Both distinct source hashes are recorded.
Under the good profile, both files arrived intact within the 8 s deadline (total
run about 9.29 s). Under 3% random loss, both arrived intact but neither on time
(total run about 53.64 s). Under burst loss, both remained incomplete at timeout.
These runs validate generic transport behavior on real weights, not adaptive
compression or rendering. A production NTC size/quality ladder remains dependent
on valid representations from Publio. Dropping synthetic jobs does not establish
decodability of a recurrent NTC sequence: real playback also needs dependency-aware
selection. The feedback loop is implemented in the real-QUIC experiment harness;
it does not yet turn the legacy sender/player into an adaptive-quality pipeline.

### Evidence and verification

The reproducible compact evidence is in
`research/quic/priority_transport/evidence/adaptive-network/`: source manifests,
per-frame outcomes, sampled metrics, netem counters, analysis and hashes of raw
qlog/chunk artifacts. The full raw output remains at
`data/outputs/adaptive-run-20261109-222015-71356/` locally.

Targeted verification: **42 tests and 755 subtests passed** across `tests`, the
priority transport and FP32 reference suites. The full-repository collection
failure is documented below. Startup cleanup and matrix nonzero exit propagation
were hardened after the main run; successful transfer/control behavior was not
changed. The failed burst run remains failed in the retained evidence.

## Research question and implemented behavior

Can sender-side observations detect delivery pressure early enough to reduce
future data volume, and then increase volume when the path recovers?

This implementation extends `codex/quic-priority-transport` in Q3DGStream. The
upstream PR was still open and draft on 9 November 2026. Publio was emailed for a
review/merge update and for his exact compressed, saliency-ranked NTC inputs.
The experiments do not depend on that response.

A producer generates one job every 250 ms. Each job is a complete file. A separate
worker sends it through the existing real aioquic QUIC transport. The application
queue contains at most eight waiting jobs plus one active job; overflowing jobs
are counted as dropped before admission. The worker retains the existing limit
of 16 chunks awaiting application ACKs. These two bounds are different.

Every 250 ms a feedback policy can change the size of *future* jobs. It cannot
recall bytes already admitted to reliable QUIC streams. Pending jobs keep their
original sizes. A receiver publishes each file only after complete SHA-256
validation. Full-file arrival time, ACK turnaround, drops and unfinished jobs
are recorded separately. No transport priority algorithm was changed; workload
files use P0 to isolate load adaptation from scheduling priority.

## Data and what the size ladder means

The generic workload uses deterministic 32, 128 and 512 KiB files. At four files
per second these offer approximately 1.05, 4.19 and 16.78 Mbit/s of payload.
They simulate different model sizes without interpreting their contents. They
are **not trained compressed NTCs**, and no image-quality claim follows from
choosing a smaller level.

For real-data integrity tests, two complete original Flame Steak FP32 checkpoints
are sent cyclically under three network profiles. Each received file must match
its original SHA-256 exactly. The checkpoint bytes are not sliced, pruned or
recompressed. Exact Publio INT4/saliency stage assets are still unavailable;
therefore this does not establish performance for those specific assets.

`--payload-dir` accepts arbitrary files. With `--sequence` it cycles through
whole files; without that option it treats size-sorted files as selectable
alternatives. Only a caller with genuinely interchangeable representations should
use the latter for actual NTC adaptation. A checkpoint from a different time is
not automatically a lower-quality representation of the current time.

## Monitored quantities

|Quantity|Meaning and limitation|
|---|---|
|Outstanding payload bytes|Queued logical work plus the active file's unacknowledged remainder; not allocated RAM|
|Oldest job age|Time since generation of the oldest pending/active job; captures delivery pressure|
|Backlog growth|Change in outstanding payload per second; bursty production can cause spikes|
|Application unacknowledged bytes|Admitted payload waiting for receiver application ACK; bounded by 16 chunks|
|QUIC retained stream bytes|aioquic stream sender buffers, including bytes retained for retransmission; not OS socket memory|
|Payload goodput|Bytes acknowledged by the receiving application per sampling interval; idle periods do not reveal link capacity|
|Smoothed/minimum RTT, RTT variance|QUIC recovery observations; baseline RTT and sampling affect interpretation|
|Congestion window, bytes in flight|QUIC congestion control state; includes transport effects rather than just application bytes|
|Packets sent and declared lost|QUIC recovery events per interval, retained alongside complete qlog|
|Receiver processing time|Receive/validation/callback time before application ACK, including injected receiver delay|
|Application ACK turnaround|Chunk submission to application acknowledgement; includes processing and both directions|
|Full-file latency and deadline misses|Generation to receiver's validated file commit, not rendering or GPU playback|

Transport diagnostics access pinned aioquic 1.7.2 internals. Netem statistics are
also retained: configured loss and QUIC-declared loss are not the same quantity.

## Policies and predeclared thresholds

- **Fixed-low/high:** always the smallest/largest payload.
- **Queue:** step down when oldest outstanding job age exceeds 500 ms.
- **RTT:** step down when smoothed RTT exceeds `max(100 ms, 1.5 × minimum RTT)`.
- **Composite:** step down on either alarm, or over 1 s with no application ACK
  while work remains outstanding.

Each adaptive policy probes one size level higher after eight consecutive healthy
samples (about 2 s). Downward steps occur at most once per sample. This is a
simple experimental controller with hysteresis, not a trained optimizer or an
implementation of a standardized adaptive streaming protocol. Minimum-RTT-based
alarms can miss a path that was already slow during connection establishment.

## Experiment protocol

All network experiments run locally in an isolated Linux Docker container using
`tc netem`; the host network is unchanged. The dynamic path is 24 Mbit/s with
10 ms per-direction delay for 6 s, then 1.5 Mbit/s with 40 ± 10 ms normal delay
and 2% random loss for 6 s, then returns to the initial path for 6 s. Phase changes
use `qdisc change` and preserve the queue; actual transition timestamps and
counters are logged. Rate shaping applies to traffic on container loopback,
including both data and return traffic, not to two independently provisioned links.

Five policies × three repetitions give 15 dynamic trials. Seven further trials
cover clean, bandwidth-limited, delayed, jittered, random-loss, burst-loss and
reordering conditions. Receiver slowdown (150 ms per chunk before application
ACK) and clean-path offered-load overload are negative controls. Three additional
runs transmit the original checkpoints. Total planned matrix: 27 trials.

The receiver deadline is 1 s for synthetic jobs and 8 s for the larger real
checkpoints. Generation stops after the specified duration; reliable in-flight
and queued jobs may drain for up to 60 s. Draining is included in elapsed runtime,
and jobs completing after their deadlines do not count as on time.

All thresholds were fixed before the main matrix. Signal analysis uses only
fixed-high traces to avoid feedback changing the predictor's own workload.
Repeat 1 is designated calibration inspection; repeats 2–3 are held out, with
no threshold fitting. Network labels come from configured phases. A separate
label asks whether any job generated in the next second will miss its deadline.
These are correlated samples, not independent observations; three stochastic
repetitions do not establish statistical significance. Netem loss traces are
not paired between policies. Static and negative-control runs are exploratory.

## Verification and known limits

The existing transport tests and new controller/integration tests cover real
QUIC file delivery, bounded queues, event byte accounting, receiver delay,
hysteresis, no-ACK pressure, idle behavior, full-file sequence replay and failure
reporting. GPU rendering and media-player adaptation are outside these tests.

The repository-wide `pytest -q` currently fails during collection in
`research/quic/progressive_299_v2/test_receiver_state_machine_v2.py` with
`TypeError: object.__init__() takes exactly one argument`, in the pre-existing
legacy receiver path. This is recorded rather than presented as a green full
repository suite.

Reproduction and metric definitions are in
`research/quic/priority_transport/README.md`. Run the matrix with
`sh research/quic/priority_transport/run_adaptive_tests.sh`, then analyze its
output directory using `python -m research.quic.priority_transport.adaptive_analyze`.
