# NTC prefetch

This is the preparation code from the Quinn playback tests. It has two parts:

- `NTCPreparation` assembles received NTC stages in a CPU worker.
- `AsyncCUDAJobs` dequantizes upcoming weights on a second CUDA stream. The
  playback stream waits for the corresponding event before using them.

The CUDA queue holds two prepared states at most. Only the playback thread
may submit CUDA work or apply an NTC to the scene. If the requested tier count
is not ready, the caller has to decode it at the playback tick.

This module does not run Quinn or the viewer. The older receiver in this
repository has not been wired to it yet.

From the repository root, run:

```bash
python3 -m unittest discover -s tests -p test_async_preparation.py -v
```

See [receiver measurements](../../../docs/RESULTS.md#asynchronous-ntc-preparation)
for the November 20 comparison.
