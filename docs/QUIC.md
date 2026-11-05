# Progressive NTC transmission over QUIC

The QUIC code lives under `research/quic/`. It includes the original FP32 transmission experiment, staged INT4 delivery, and an adapter that sends saliency-ranked blocks for the 299-update Flame Steak sequence.

The experimental sender transmits Gaussian additions and four `.qblk` files per NTC, in the order P0, P1, P2, P3. Each stage carries network parameter blocks. Their priority is chosen before sending, and QUIC is used as the reliable transport.

## Implementations

| Directory | Experiment |
| --- | --- |
| `fp32_reference/` | FP32 QUIC sender and reference cache checks |
| `progressive_pilot_v1/` | Initial receiver and progressive stage sender |
| `progressive_299_v2/` | Bounded CPU decoding and per-NTC stage updates |
| `dynamic_priority_adapter/` | Per-frame saliency-ranked stages and Gaussian additions |

The sender and evaluation code are available in Git. The most recent CUDA viewer and receiver modifications are still being integrated from the working local installation. The source checks alone do not establish that a fresh clone can render progressive QUIC updates.

The [GPU validation guide](GPU_VALIDATION.md) covers a read-only source preview, safe integration into a separate checkout, CUDA inference tests, and live receiver tests.

## Receiver behavior

A `.qblk` file contains quantized NTC weights, **not rendered geometry**. On reception, the client reconstructs the available parameter blocks and evaluates the NTC to obtain a candidate Gaussian update.

Successive priority stages may refine the result for the same animation step. Each revision must be computed from the **same pre-step Gaussian state**. Applying the P1 displacement to a position that already received P0 would accumulate motion incorrectly.

The frame index also matters: the dynamic NTC representation is recurrent, so the state for frame *t* depends on the preceding update. Falling behind in animation time and receiving only part of the current NTC are two separate situations.

## Local 30-update experiment

After the receiver has been integrated and the preflight checks pass, start the viewer and sender in separate terminals:

**Receiver**

```bash
cd research/quic/dynamic_priority_adapter
bash run_receiver.sh
```

**Sender**

```bash
cd research/quic/dynamic_priority_adapter
bash preflight.sh
QNTC_STAGE_DELAY=0 bash run_sender.sh
```

The default sender range covers updates 0 through 29, with four priority stages and one additions file per update. To transmit all 299 updates, restart the receiver and use `QNTC_END=298` on the sender.

The `QNTC_STAGE_DELAY` setting introduces a pause after each stage for inspecting partial reconstruction. A nonzero value intentionally lowers the rate of complete NTC updates.

## Scope of the prototype

The measured local QUIC experiment confirms delivery of 30 updates, 120 priority stages, and 30 additions files. It uses **application-level stage ordering**, not QUIC's internal stream-priority mechanism. Claims about adaptive bandwidth control, end-to-end 30-update-per-second playback, or saliency-weighted rendered QoE require the corresponding network and GPU measurements.

See [Results](RESULTS.md) for completed tests and [Experiments](EXPERIMENTS.md) for reconstruction evaluation.

## Optional bounded chunk transport

The local extension in [priority_transport](../research/quic/priority_transport/README.md)
adds FIFO, strict and weighted chunk admission to the existing dynamic sender via
`--transport priority`. It has its own headless receiver and ALPN, preserves stage
file bytes and per-frame dependency barriers, and includes synthetic QNP01 tests
under Linux netem. This does not modify QUIC's internal scheduler or establish
compatibility with the legacy GPU receiver. See that directory for reproducible
commands and a comparison with the original stage-ordering implementation.
