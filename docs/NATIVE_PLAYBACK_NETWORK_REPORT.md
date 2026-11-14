# QUIC under network impairment in the native CUDA player

14 November 2026 · gpu1 · Quadro RTX 8000

## What ran

The original GLFW viewer presented Flame Steak through its native CUDA NTC and
rasterization path while a local QUIC sender transferred real FP32 checkpoints
and matching additions. The experiment uses a separate authenticated X display.
Linux tc netem runs inside a private network namespace on its loopback device.
The sender and receiver share that namespace and its qdisc. The network link
between different physical hosts and the Mac remote desktop are outside this test.

The opt-in viewer clock waits for contiguous media, presents updates sequentially,
and refills after an underrun. Startup is measured from before the connection
handshake to the first GLFW swap. A network-buffer stall starts when the next
update is due but unavailable and ends at presentation after refill. The
readiness deadline uses the initial playback schedule. Long model loads or
render intervals can delay presentation independently, so stall duration should
not be read as all elapsed playback delay. The latest harness also records exact
NTC applications and reports completion only after the intended extent of play.

## Original-data baseline

This uses 30 video fps, 0.1 s startup and refill targets, and the original files
with no optional payload. Rates and delays below are qdisc settings.

| Run | Committed updates | Presented frames | Startup (s) | Stall count | Stall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean, Cubic | 8/8 | 9/9 | 13.52 | 2 | 9.60 |
| Clean repeat, Cubic | 8/8 | 9/9 | 12.36 | 2 | 8.73 |
| 60 Mbit/s, 15 ± 5 ms jitter, Cubic | 8/8 | 9/9 | 24.99 | 2 | 18.60 |
| 60 Mbit/s, 10 ms, 1% loss, Cubic | 8/8 | 9/9 | 71.11 | 2 | 56.66 |
| 60 Mbit/s, 10 ms, 3% loss, chunked Cubic | 5/6 | 4/7 | 128.75 | 1 | 40.56 through cutoff |
| Same 3% profile, chunked Reno | 4/6 | 4/7 | 146.60 | 1 | 22.66 through cutoff |
| Same 3% profile, sequential whole files, Cubic | 6/6 | 7/7 | 115.64 | 1 | 53.42 |

Both chunked 3% runs hit the 170 s limit. The whole-file trial finished in 169.89 s
with six exact NTC applications and zero missing applications. Its margin was
0.11 s. Loss realizations were independent and unpaired, with one trial per
3% configuration. These results do not establish a general winner or sustained
loss recovery. Partial-run stall durations end at the cutoff and cannot be ranked
against completed playback using the raw stall total. Even clean-loopback FP32
delivery did not keep up with the 30 fps presentation schedule for this workload.

## Native playback feedback with optional traffic

The viewer reports contiguous playable media and a two-second committed-payload
rate estimate to the sender through an atomic same-host JSON file. The sender
chooses optional bytes per future update using fresh feedback, a buffer minimum,
and a rate budget with 20% headroom. Required FP32 checkpoints and additions have
priority 0. Optional synthetic bytes have priority 3 and are stored as experiment
files. This loads the transport without changing the decoded scene.

All runs use nine original FP32 updates, 0.25 fps, and a four-second startup/refill
target to leave room for the feedback experiment. The qdisc changes after every
three update submissions: clean, 60 Mbit/s with 10 ms and 1% random loss, then
clean. Phase changes use qdisc replacement at the update boundary, after the
prior update has committed. Durations vary by policy. Fixed uses 4 MiB of optional
bytes per update. The required-only baseline sends zero optional bytes.

| Policy | Presented frames | Startup (s) | Stall count | Stall time (s) | Optional MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed | 10/10 | 10.56 | 4 | 31.04 | 37.75 |
| Adaptive | 10/10 | 9.65 | 3 | 21.08 | 12.40 |
| Fixed repeat | 10/10 | 8.56 | 4 | 30.18 | 37.75 |
| Adaptive repeat | 10/10 | 8.29 | 2 | 17.19 | 8.38 |
| Required only | 10/10 | 8.56 | 2 | 12.96 | 0 |

Adaptive admitted 67–78% fewer optional bytes and spent about 10–13 fewer seconds
waiting for the buffer than fixed in these runs. The required-only run had the
fewest stall seconds. Adaptation avoided part of the cost of optional load while
still delivering some of it. It did not reliably increase optional volume again
after recovery in these short runs, and the design currently waits for an
update's optional transfer before submitting the next update. A quality selector
that switches between validated NTC representations and sustained playback under
loss remain separate evaluation steps.

Reproduce on the prepared server with a new output directory for each run:

```bash
bash scripts/run_gpu1_network_trial.sh \
  --scene /local/users/abeigi/Q3DGStream/data/scenes/flame_steak \
  --output /local/users/abeigi/q3dg-trial-loss1 \
  --profile loss1 --updates 8 --timeout 150

bash scripts/run_gpu1_network_trial.sh \
  --scene /local/users/abeigi/Q3DGStream/data/scenes/flame_steak \
  --output /local/users/abeigi/q3dg-trial-adaptive \
  --profile dynamic --policy adaptive --updates 9 --fps .25 --prebuffer 4
```

Compact JSON reports and per-update sender feedback are in
`research/quic/priority_transport/evidence/native-playback-network`. Verbose
logs, viewer caches, and session keys remain on gpu1. Earlier trials recorded
display readiness and swaps. The repeat trials also recorded exact NTC
applications. The scoped suite on gpu1 passed 54 tests, with one occupied-port
test skipped while the existing demo held its port.
