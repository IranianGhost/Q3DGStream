# Prioritized QUIC delivery integrated with Q3DGStream

Prepared for project review — 5 November 2026.

## Main result

The experimental transport now feeds the existing Q3DGStream player through its live checkpoint and additions providers. The native player is retained; a new QUIC receive mode replaces its TCP receive path when explicitly selected. This is raw QUIC using aioquic, not MoQT.

Original trained Flame Steak checkpoints were transferred and loaded through those providers on the local machine. GPU inference and graphical playback remain unverified: this Mac cannot run the player's NVIDIA CUDA/tiny-cuda-nn stack. The CPU smoke test uses a CPU model holder and renderer stub while exercising the original lazy checkpoint/additions providers; it does not establish image quality or playback frame rate.

## What changed from upstream

| Area | Existing project path | Added implementation |
|---|---|---|
| Sending | P0–P3 whole-file order through a legacy transport module absent from this checkout | Explicit FIFO, strict and weighted application admission policies; 8 KiB chunks; at most 16 unacknowledged chunks |
| QUIC scheduling | Legacy implementation unavailable for inspection | aioquic's existing stream scheduling; no new native QUIC scheduler |
| Reception | Existing native player consumes complete checkpoints through LiveTCPState | LivePriorityState receives QUIC files and reuses the same lazy NTC/additions providers |
| Integrity | Existing binary format parser | Chunk/file hashes, atomic file publication, TLS verification and application commit acknowledgments |
| Compressed stages | QNP01 records and upstream INT4 decoder | All four validated stages reconstruct one complete checkpoint before the player is notified |
| Missing updates | Existing sparse/hold-last options | Contiguous update gate, additions required by default; no silent substitution of another NTC |
| Evidence | Source-level and format checks | Actual QUIC exchange, sender qlogs, receiver commit times, netem impairments, trained parameter equality |

The separate receiver mentioned in the earlier report was functional as a file-transfer test, but was not connected to playback. This change adds that connection. It does not replace the renderer or claim that incomplete NTC weights are renderable.

## Data provenance

The [original project download script](https://github.com/publioelon/Q3DGStream/blob/f89cab37d83d3599e7ea8c85451fe1dcc40269e8/scripts/download_flame_steak.sh) links to Google Drive file `1AXDqSzSaT_uNu_DhKeSmZmrBAfuOhWYY`. The downloaded Flame Steak archive contains 299 trained NTC checkpoints, an initial Gaussian scene, configuration and per-update additions. The archive was fully extracted with ZIP CRC validation. Archive SHA-256: `5822aab8ccf5af4c18ca8689edf116c06ead4b5c6e8f7e8a354980f3c62a8880`.

The network comparison uses the original `NTCs/NTC_000000.pth` unchanged: **7,968,015 bytes**, **1,991,680 FP32 model parameters**, plus the original spatial bounds. SHA-256:

`7465eb688a3524da736fc925ee7e152be558315887d6a4b4e66991afe1f8beb1`

Publio's compressed INT4 checkpoint and saliency priorities were not publicly available from the supplied sources. We have not invented importance labels for trained weights. Synthetic QNP01 tests and original FP32 tests are reported separately. Dataset files and generated caches are excluded from Git.

## How prioritization works

1. Every available file has an explicit application priority and release time.
2. The application selects the next eligible chunk using FIFO, strict priority, or weighted admission (8:4:2:1).
3. A chunk is placed on a reliable QUIC stream. Only 16 chunks may be outstanding; a new admission waits for an application acknowledgment.
4. The receiver validates chunks and publishes a complete file atomically. Compressed NTC publication additionally waits for all four stages and complete, nonoverlapping block coverage.
5. The player advances only when the next contiguous update and required additions are available.

Application priority chooses what enters QUIC next. Native QUIC stream scheduling decides how already-admitted stream bytes share packets and retransmissions. This implementation changes the former, not the latter. Sender qlogs confirm actual first STREAM-frame ordering, beyond merely recording queue decisions. They do not guarantee receiver arrival order or constitute a physical packet capture.

## Local correctness evidence

- 32 tests passed, including additional source and format subtests, covering existing source/format behavior and the new transport/bridge behavior.
- Real network-to-player smoke: initial scene, configuration, three original checkpoints and additions; 6,518 chunks; playback gate advanced to display frame 3.
- Received parameters and spatial bounds were exactly equal to the originals. Future missing NTCs were not substituted with older ones.
- Both whole-file and chunked transfers exercised real QUIC connections and receiver callbacks.
- A review found a false startup error when metadata preceded checkpoint bounds; model initialization now waits for checkpoint/bounds availability, with a regression test.

## Original-checkpoint network comparison

The comparator is a **reconstructed whole-file-per-stream reference using the same aioquic connection and receiver**, not the absent original legacy QUIC binary. Both methods wait for a validated application commit. Timing starts after handshake and includes transfer preparation; scene startup, GPU inference and rendering are excluded.

Each run transfers the same complete trained checkpoint. It is one ready P0 job, so there is no priority competition. This experiment evaluates framing, concurrency and acknowledgment cost. It cannot demonstrate a saliency-priority or playback-quality improvement.

Six Linux `tc netem` profiles run inside a Docker container on the local Mac: clean, 5 Mbit/s, 40 ms delay, 40±10 ms normal jitter, 20 ms plus 2% random loss, and 20 ms plus Gilbert–Elliott burst loss. All impaired profiles use 5 Mbit/s. Loopback packets traverse the qdisc in both directions, so configured delay is not a measured RTT. There are three repeats per method/profile, with alternating method order. Three repeats are exploratory, not statistical significance.

**35 of 36 transfers completed** and passed byte-hash and parameter/bounds equality checks. One chunked burst-loss run failed; it is retained in the results. Packet drops were observed in every loss/burst run. Median complete-checkpoint commit times:

| Profile | Whole-file ms | Chunked ms | Chunked / whole |
|---|---:|---:|---:|
| clean | 453.9 | 861.1 | 1.90× |
| bandwidth | 14101.3 | 14819.3 | 1.05× |
| delay | 14261.3 | 14997.1 | 1.05× |
| jitter | 14329.6 | 15222.8 | 1.06× |
| loss | 42716.5 | 44663.0 | 1.05× |
| burst | 36209.8 | 40917.0 | not compared (2/3 completed) |

A ratio above 1 means the chunked method was slower. No median speedup was observed in the five profiles with all repeats completed. The burst chunked median describes only its two successful attempts and is not a complete comparison. The earlier priority-stress improvement does **not** transfer automatically to this scenario.

Compact per-run evidence is committed under `research/quic/priority_transport/evidence/real-checkpoints/`; full events and qlogs are retained locally under `data/outputs/real-run-20261105-164931-82388`.

The failed run returned an empty receiver error after approximately 32.9 seconds (qlog duration). This is consistent with the configured 30-second receiver read deadline, but the original message does not conclusively identify the exception. A separate diagnostic retry using the same netem seed completed in 36.7 seconds; it does not replace the failed attempt. Error reporting now preserves exception types, and the benchmark records failure metadata. The intermittent transfer failure remains a review item; no fix to its network cause is claimed.

## Separate synthetic priority stress evidence

The earlier QNP01 experiment created 32 frame IDs and deliberately released urgent stages after a lower-priority backlog. Across 54 runs, 6,912 stage files and 13,824 chunks passed hashes, parsing and dense decode equality. Median p95 latency to complete P0 stages was:

| Profile | FIFO ms | Strict ms | Weighted ms |
|---|---:|---:|---:|
| Clean | 66.2 | 59.6 | 100.8 |
| 5 Mbit/s | 1686.1 | 993.7 | 1678.7 |
| Delay | 1982.4 | 1218.5 | 2019.3 |
| Jitter | 2094.9 | 1295.9 | 2113.2 |
| Random loss | 6034.1 | 2208.4 | 5362.3 |
| Burst loss | 4244.2 | 3078.5 | 3572.4 |

Median P0 first-chunk STREAM-frame rank improved from 96.5 to 48.5 with strict admission in impaired profiles. Clean-loopback ranks were equal. The production-shaped four-stage batch has only eight chunks and fits the entire 16-slot window; no priority-latency improvement was demonstrated for that batch. These are synthetic transport results, not trained-model rendering improvements.

## Playback policy and limitations

The integration intentionally waits for a complete NTC update. Applying an incomplete transformation repeatedly could change the recurrent scene state incorrectly. Earlier delivery of selected parameters does not yet yield an earlier renderable frame.

Strict admission can starve low-priority work. Weighted admission allocates chunk opportunities, not guaranteed bandwidth. Already-admitted data cannot be recalled. Reliable retransmissions remain enabled; there is no expiry or deadline cancellation. The 16-chunk application-ACK window can substantially limit throughput at high RTT. Network reordering and loss can invert arrival order.

The receiver is currently local-only and requires a fresh empty cache. GUI execution needs an NVIDIA machine. INT4 reconstruction is tested with synthetic valid stages; compatibility and visual quality with Publio's actual compressed assets still require those assets and GPU validation.

## Reproduction

From the repository root, use the project's existing CUDA environment for graphical playback. Install the transport dependencies from `research/quic/priority_transport/requirements.txt`; CPU smoke additionally needs compatible PyTorch and plyfile (`requirements-player.txt` records the local test version).

Generate a local test certificate (do not commit its private key):

```sh
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem \
  -days 7 -subj /CN=localhost -addext subjectAltName=DNS:localhost
```

On an NVIDIA/CUDA machine, launch the original player in its new receive mode:

```sh
python main.py --priority_listen 5003 --priority_cache data/cache/priority-demo-new \
  --priority_cert cert.pem --priority_key key.pem --frames 4 --autoplay
```

Send three original updates from a second terminal:

```sh
python -m research.quic.priority_transport.send_scene \
  --scene data/scenes/flame_steak --cert cert.pem --port 5003 --updates 3
```

CPU verification on the local Mac (no graphical rendering):

```sh
python -m pytest tests research/quic/priority_transport -q
python -m research.quic.priority_transport.player_smoke \
  --scene data/scenes/flame_steak --cert cert.pem --key key.pem \
  --output data/outputs/player-smoke-new
```

The network matrix needs Docker Desktop and the original first checkpoint at the scene path:

```sh
sh research/quic/priority_transport/run_real_tests.sh
python -m research.quic.priority_transport.real_analyze data/outputs/real-run-TIMESTAMP-PID
```

Each run records summary JSON, transmission events, qlog and netem counters. The launcher copies files into the container to avoid Mac bind-mount permissions, and removes the container after copying results out. It does not change the host network.

## Review and next decisions

Review the transport admission policy independently from the partial-model rendering question. The useful next validation is on an NVIDIA system with Publio's exact compressed checkpoints and saliency outputs. Measure time to a usable frame and quality over time before claiming a visual benefit. A larger/adaptive admission window or transport-level scheduler may be needed if throughput under delay is a primary objective.
