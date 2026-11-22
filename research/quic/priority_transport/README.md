# Bounded QUIC priority transport

Start with the [review guide](../../../docs/QUIC_REVIEW.md): what changed, test conditions, results and limits.
Use the [server commands](../../../docs/QUIC_REPRODUCE.md) to repeat the experiments.
All measured cases, including failures, are in [results.csv](evidence/results.csv).

Code map:
- `core.py`: FIFO, strict and weighted admission of ready chunks.
- `transport.py`: reliable QUIC streams, bounded outstanding work and hash-checked file assembly.
- `player_bridge.py` and root `live_priority.py`: publish complete files to the original viewer.
- `live_playback.py`: startup buffer, media clock and playback feedback.
- `checkpoints.py`: decode the supplied sparse INT8/INT4/INT2 checkpoints.
- `publio_matrix.py`: matched transfer and constructed priority tests.
- `publio_native.py`: original CUDA viewer and real model-size adaptation.

Priority is applied before admitting a chunk to QUIC. This does not replace the QUIC library's internal packet scheduler.
