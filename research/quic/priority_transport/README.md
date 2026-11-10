# Bounded priority transport for Q3DGStream

This opt-in path extends `../dynamic_priority_adapter/send_dynamic_priority.py`.
It uses real aioquic QUIC streams and preserves the bytes of upstream QNP01 stage
files. It does not change saliency ranking, NTC compression, model inference or
QUIC's internal packet scheduler. No MoQT is involved.

## What changed

| Aspect | Upstream code at f89cab37 | Local extension |
|---|---|---|
| Priority assignment | Existing saliency-ranked P0–P3 files | Uses those same priorities |
| Transfer calls | Await `send_file` for each complete stage in order | Split files into 8 KiB chunks and schedule admissions |
| Scheduling policy | Fixed stage order in the visible caller | FIFO, strict P0-first, weighted 8:4:2:1 |
| Queue control | Internal behavior unknown because `quic_transport` is absent | At most 16 chunks awaiting application receipt ACKs by default |
| Evidence | Existing project-specific pilot reports | Per-chunk selection, submission, receive, ACK, stream IDs, qlog, netem counters |
| Integrity | Legacy transport implementation unavailable for audit | SHA-256 per chunk and per reconstructed file, atomic file publication |
| TLS | Existing caller sets `CERT_NONE` for loopback | New path verifies the explicitly trusted localhost certificate |
| Dependencies | Metadata, additions, then stages in frame order | Explicit commit barriers preserve those dependencies |
| Receiver | Legacy live/GPU receiver partly absent from this clone | Standalone receiver plus native-player LivePriorityState bridge; GPU validation outstanding |

These are implementation differences, not a measured speedup over the upstream
binary. The network comparison uses FIFO versus strict versus weighted **within
the new transport**, holding chunk size and admission window constant.

## Local execution

Use Python 3.12+ for the pinned optional environment. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/pip install -r research/quic/priority_transport/requirements-player.txt
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 7 -subj '/CN=localhost' -addext 'subjectAltName=DNS:localhost'
.venv/bin/python -m pytest research/quic/priority_transport -q
.venv/bin/python -m research.quic.priority_transport.benchmark --scenario pipeline --frames 2 --cert cert.pem --key key.pem --output data/outputs/priority-pipeline
```

Use a fresh output directory for every benchmark. Certificates/keys and generated
outputs are ignored by Git. Local runs use one process with both endpoints, so
monotonic clocks are comparable. The standalone receiver can use another process
on this machine. Do not compare its monotonic timestamps across different hosts.

Run all network profiles with Docker Desktop active:

```sh
sh research/quic/priority_transport/run_network_tests.sh
```

This builds an isolated Linux container with `tc netem`, copies source files in
(no Documents bind mount), generates a fresh TLS certificate, runs tests and
copies artifacts to a new `data/outputs/priority-run-*` directory. Only the
container loopback qdisc changes. Each of six profiles runs three repeats of each
policy: clean, 5 Mbit/s, 40 ms delay + 5 Mbit/s, normal delay jitter, 2% random loss,
and Gilbert–Elliott burst loss. Delay applies in both directions. Netem seeds are
not fixed, so loss traces differ between runs.

## Using actual scene files

The existing adapter retains all its source validation. Supply its scene, wire,
and stage directory arguments as before, plus the new transport options:

```sh
# Terminal 1: new headless receiver, with a NEW/EMPTY dedicated cache
.venv/bin/python -m research.quic.priority_transport.receiver --cache data/cache/priority-test --cert cert.pem --key key.pem --port 5003

# Terminal 2: existing sender, using its configured data directories
.venv/bin/python research/quic/dynamic_priority_adapter/send_dynamic_priority.py --transport priority --policy strict --max-inflight 16 --cert cert.pem --port 5003 --transport-log data/outputs/priority-events.json
```

The full scene command still requires the original scene assets, compressed
checkpoint and PyTorch needed to extract model bounds. These assets are not in
Git and were **not available for the initial synthetic experiment**. That initial experiment used
synthetic INT4 values, not trained NTC weights. Original trained FP32 weights have
since been obtained; see `docs/PRIORITY_PLAYER_REPORT.md`. Empty Gaussian/metadata fixtures
exercise the adapter's ordering only, not a valid rendered scene.

The new ALPN is `q3dg-priority-chunks/1`. The legacy receiver does not understand
this chunk framing. Complete files retain their original QNP01 format, and a new `LivePriorityState` now connects this framing to the existing player
(see `docs/PRIORITY_PLAYER_REPORT.md`). GPU validation remains outstanding. Legacy mode remains
the default. `--stage-delay` applies only to legacy mode.

## Experiment and interpretation

The stress workload contains 32 frame IDs, four stages per ID, 389 records per
stage, 1,556 unique block IDs per frame. Each 15,574-byte stage becomes two chunks.
All P3 files become available at 0 ms, and P0/P1/P2 at 150 ms. This intentionally
creates late urgent arrivals. Cross-frame availability is artificial and does not
model the recurrent renderer. Separately, a two-frame integration test runs under all six profiles through the
actual adapter helper and asserts additions/stage/frame commit order.

Every received QNP01 file must match its source hash and parse using upstream
`q3dgstream.parse_qblk`. Upstream `decode_records_into` must produce the same dense
INT4 parameter array before and after transmission. None of these checks establish
visual quality or quality relative to original FP32 NTCs.

`analyze.py` audits actual admission choices, receipt timestamps, successful file
validation, sender STREAM-frame qlogs, and positive netem drop counters in every
loss run. The sender first-send rank is the order of first STREAM-frame appearance
in qlog. This is stronger evidence than application submission alone, but is not
a hardware packet capture or a guarantee of arrival order. It is a chunk-order
metric, not a byte-share metric.

Strict scheduling can starve lower priorities under sustained urgent load.
Weighted scheduling shares chunk admissions, not exact bandwidth or receipt times.
Already admitted data cannot be recalled. Reliable streams retry lost data, and
there is no deadline cancellation. The application ACK window can limit throughput
on long-RTT paths. An earlier P3 chunk may still arrive before a later P0 chunk.
Three stochastic repeats provide exploratory evidence, not statistical significance.

The production adapter's strict per-frame barriers restrict scheduling freedom.
With all four stages already ready, strict selection largely follows upstream's
P0–P3 order. The largest demonstrated benefit concerns late urgent arrivals behind
a lower-priority backlog. Do not claim the stress result as an NTC playback gain.

## Recorded local results

The run stored in `data/outputs/priority-matrix/REPORT.txt` completed 54 stress
transfers (6,912 QNP01 files, 13,824 chunks) plus the two-frame adapter test under
all six profiles. All files passed hashes, QNP01 parsing and dense decode parity.
For complete P0 stages, median p95 latency over three repeats was:

| Profile | FIFO ms | Strict ms | Weighted ms |
|---|---:|---:|---:|
| Clean | 66.2 | 59.6 | 100.8 |
| 5 Mbit/s | 1686.1 | 993.7 | 1678.7 |
| Delay | 1982.4 | 1218.5 | 2019.3 |
| Jitter | 2094.9 | 1295.9 | 2113.2 |
| Random loss | 6034.1 | 2208.4 | 5362.3 |
| Burst loss | 4244.2 | 3078.5 | 3572.4 |

Median P0 first-chunk-send rank was 96.5 with FIFO and 48.5 with strict for all
impaired profiles. Clean-loopback ranks were equal (96.5), so its timing difference
is not evidence of bypassing queued lower-priority data. Rankings are per chunk.
The production-shaped batch contains eight chunks and fits in the 16-slot window.
That pipeline therefore does not demonstrate a priority latency improvement.

## Adaptive offered-load experiments (November 2026)

The new `experiment` module runs a periodic producer independently of transfer
completion. This is essential: a sender that waits before producing its next
frame cannot reveal upstream backlog growth. Payload-size feedback changes future
jobs; existing QUIC chunks remain reliable and are not cancelled. The producer
queue is capped at eight waiting files (plus one active file), and drops are
explicitly counted. The existing transport still admits at most 16 chunks.

```sh
# Full isolated Linux netem matrix; Docker Desktop must be running.
sh research/quic/priority_transport/run_adaptive_tests.sh
python -m research.quic.priority_transport.adaptive_analyze data/outputs/adaptive-run-<timestamp>

# Local smoke test; generate localhost certificate as above.
python -m research.quic.priority_transport.experiment --cert cert.pem --key key.pem \
  --output data/outputs/adaptive-smoke --duration 3 --policy composite

# Replay arbitrary WHOLE files (including original NTC checkpoints) cyclically.
python -m research.quic.priority_transport.experiment --cert cert.pem --key key.pem \
  --output data/outputs/replay --payload-dir /path/to/payloads --sequence --fps 0.2 --duration 10
```

Without `--sequence`, supplied files sorted by size become selectable load levels.
This is valid only when the caller intentionally supplies alternative payloads:
it is not a general way to substitute one temporal NTC checkpoint for another.
Synthetic levels are 32/128/512 KiB at 4 Hz (1.05/4.19/16.78 Mbit/s payload).
They are transport load surrogates, **not decodable NTC quality representations**.
Default `composite` is an experimental policy, not a production recommendation.

Every run records samples, frame outcomes, chunk submit/ACK events, complete qlog,
source hashes, and netem counters. `queue_bytes` includes queued payload plus the
unacknowledged remainder of the active file; it is logical work outstanding, not
allocated RAM or the kernel socket buffer. `quic_retained_bytes` is the sum of
stream sender buffers retained for transmission/recovery; `app_unacked_bytes`
counts admitted payload waiting for application ACK. These differ from QUIC
`bytes_in_flight`. Payload goodput excludes framing, encryption and retransmits.

The controller samples every 250 ms, steps down on queue age >500 ms and/or RTT
>max(100 ms, 1.5 × minimum RTT), and probes one level higher after eight healthy
samples. Composite also responds to >1 s without ACK with pending data. All
thresholds are experimental and frozen before the main evaluation. Flags are
pressure alarms, not proof the network itself failed. Compare receiver slowdown
and clean-path overload controls before attributing a growing queue to the link.

Matrix: 15 good→bad→good trials (five policies × three repetitions), seven static
profiles, two negative controls, and three original-checkpoint runs. Phase changes
use `tc qdisc change` so queued packets are not deliberately flushed. Kernel
randomness is not paired across policies. The `seed` field identifies the repeat;
it is not a fixed netem RNG seed. Netem's actual seeds and counters are retained.
The 1 s deadline is measured at full receiver file commit on the same host;
ACK turnaround is recorded separately. No GPU/playback quality is measured.

Analysis scores signal thresholds only on fixed-high traces, with repeat 1 for
calibration inspection and repeats 2–3 held out. No thresholds are fitted on these
runs. Samples are correlated and the sample confusion counts do not establish
statistical significance. Static cases and negative controls are diagnostics,
not randomized comparative trials.

## Playback buffering and loss diagnostics (10 November)

Current sender admission default is 128 chunks (1 MiB payload), configurable down
to 1. It is a ceiling, not a throughput guarantee. The priority adapter and scene
sender expose `--cc reno|cubic` and `--datagram-size` (local test default 1400).
Use 1200 for conservative path compatibility; 1400 assumes a suitable path MTU.
Chunk ACK recovery budget is 120 s, independently of playback deadlines.

`experiment --policy buffer --prebuffer 2 --playback-buffers 1 2 4` implements
contiguous media-time playback, stalls and refills. `--vod` enables finite-video
prefetch with backpressure; otherwise media jobs are generated over time.
The buffer controller uses receiver buffer seconds plus acknowledged payload
rate to choose future sizes; queued VOD jobs can be resized before admission.
The simulation assumes initial scene/decoder state is already available. It does
not run GPU decoding or model visual dependencies beyond contiguous intervals.

Documented byte scales are 62,296 / 66,741 / 7,968,015, corresponding to indexed
keep05 stages, packed keep05 and original FP32. These are different representations
and formats, not a measured quality ladder. Only FP32 asset bytes are available.
Use `--sizes` to choose synthetic loads and `--payload-dir --sequence` to preserve
real files. Never substitute a different time's checkpoint as a quality level.

Run `sh research/quic/priority_transport/demo_playback.sh` for a native clean-path
demo. For netem comparisons use `run_playback_tests.sh` (Docker required). The
additional `playback_followup` module runs continuous-generation feedback and MTU
diagnostics inside the same isolated Docker image. Analyze output with
`python -m research.quic.priority_transport.playback_analyze OUTPUT --output REPORT`.
Playback CSV/JSON and English results are committed under `evidence/playback-network`.
