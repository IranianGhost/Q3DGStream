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
