# Quinn NTC transport tests

This experiment checks whether the order in which we write P0–P3 to Quinn
helps or gets in the way of Quinn's own stream priorities. The sender is in
`src/main.rs`; `run.py` selects the experiment and collects its results.

Run the commands below from `research/quic/priority_transport`.

You'll need Python 3.10+, NumPy, Rust/Cargo, and the compressed INT4 NTCs
with their P0–P3 ordering files. The default data paths point to our Ubuntu
test machine. On another machine, supply `--wire-dir` and `--priority-dir`.
The files should be named `wire/NTC_000000.pth` and
`orders/NTC_000000.npz`, continuing through `NTC_000298`.
For the shaped network tests, install `iproute2` and allow `sudo` to create
network namespaces. Add `--no-netns` for a quick loopback check.

For example, with the files unpacked under `/data/ntcs`:

```bash
python3 run.py p0-first --frames 5 \
  --wire-dir /data/ntcs/wire --priority-dir /data/ntcs/orders
```

Before running the experiments:

```bash
python3 -m unittest discover -p 'test_*.py' -q
cargo test
python3 run.py p0-first --frames 3 --dry-run
```

## Compare writing methods

```bash
python3 run.py all --frames 30 --reps 3
```

This tests seven application methods against equal and tiered Quinn priorities.
It uses a 30 Mbps, 10 ms one-way delay, zero-loss network namespace by default.
The same NTC data is sent in every run.

Methods: `original`, `sequential`, `concurrent`, `p0-first`, `fifo`,
`strict`, `weighted`. Pick one instead of `all`, and optionally set
`--quinn equal` or `--quinn tiered`.

The result directory contains `summary_scheduling.csv`,
`arrival_order_by_method.csv`, and `per_frame_all.csv`. Use these to compare
arrival order, P0 delivery, and deadline misses. A completed QUIC transfer
does not necessarily mean the NTC arrived before its playback deadline.

## Watch P0, then P1–P3

```bash
python3 run.py p0-first --progressive --show-window
```

This plays all 299 NTCs at 1 FPS (about five minutes). P0 is released
immediately; P1, P2 and P3 follow after 250, 500 and 750 ms. Playback
starts with P0 and refines the current frame as the other groups arrive.
The rendered frames appear in your browser at `http://127.0.0.1:8765`.
Open that address manually if the tab does not appear.

Arrival and display times are saved to `progressive_events.csv` inside the
run's `priority/live` folder. Add `--save-stages` to keep every intermediate
PNG; otherwise the browser preview avoids storing hundreds of images.

Change the delays with `--tier-delays-ms 0,100,200,300`. These are **sender
write offsets**, not added network latency. A late tier cannot refine a frame
that has already advanced, so use slow playback to inspect individual tiers.

The CUDA renderer also needs the 3DGStreamViewer, PyTorch/CUDA, Pillow,
PyOpenGL, and the initial Gaussian scene. The browser displays the renderer's
output; it is not a free-camera viewer. Change paths with `--scene-base` and
`--viewer`. For a short check, add `--frames 5`.

## Watch at 30 FPS

```bash
python3 run.py p0-first --progressive --show-window \
  --frames 299 --fps 30 --tier-delays-ms 0,0,0,0
```

This requests 30 NTCs per second and removes the artificial P1–P3 delays.
The browser reports the observed NTC update rate. The GPU may render fewer
unique updates if decoding and inference cannot keep up.

At 30 FPS, frames are only 33.3 ms apart, so there is not enough time to
show four updates separated by 250 ms each. The receiver displays whichever
consecutive groups are ready for the current frame; late enhancements to an
older frame are not replayed. Use the slow preview above to see P0–P3
individually.

The slow-motion preview is for observing arrival and refinement. The 30 FPS
command targets live playback; whether it sustains 30 distinct NTC updates
must be checked using the observed rate, not the requested `--fps` value.

Latest GPU-server [transport controls and pending quality inputs](evidence/quinn-20261122/SUMMARY.md).
