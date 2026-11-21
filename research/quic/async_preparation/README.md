# Async NTC preparation

This module contains the receiver-side preparation logic from the Quinn
prefetch experiment. It is kept separate from the older QUIC viewer until
the live receiver and CUDA hooks are ported and tested together.

Received P0–P3 stages contain packed INT4 weights. The CPU worker assembles
the available stages for upcoming NTCs. A dedicated CUDA stream can then
dequantize an exact `(frame, tier count)` state ahead of playback. At the
scheduled tick, the playback stream waits for its CUDA event if needed,
installs the weights, and runs inference.

There are two important rules:

- Only the playback thread owns CUDA preparation and changes the scene.
- A partial NTC must not be replaced with a different tier count silently.

The GPU staging queue holds at most two future states. If preparation is not
ready, the caller must use its measured on-tick fallback. This module does
not open a QUIC connection or render frames on its own.

Run the CPU-only tests from the repository root:

```bash
python3 -m unittest discover -s tests -p test_async_preparation.py -v
```

The measured preparation results and test conditions are in
[Results](../../../docs/RESULTS.md#asynchronous-ntc-preparation).
