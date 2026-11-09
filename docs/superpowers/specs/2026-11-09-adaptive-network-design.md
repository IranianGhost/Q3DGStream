# Adaptive load experiments over QUIC

User-authorized continuation on codex/quic-priority-transport, regardless of pending PR #1.
Use existing aioquic sender/receiver, TLS and SHA-256, not MoQT. Keep admission priority independent from load adaptation. No changes to NTC decoding or claims of visual quality.

Compare sender application queue age/bytes, transport outstanding bytes, QUIC RTT/variance, congestion window/bytes in flight, detected packet loss, application ACK delivery rate, receiver processing cost and delivery deadline misses. Separate the bounded 16-chunk admission window from an upstream producer queue; its saturation alone cannot establish congestion. CPU/receiver slowdown and overload on a clean path are negative controls for diagnosing network impairment.

Use real original Flame Steak checkpoints unchanged for transport integrity tests. Exact Publio INT4/saliency stages are not available; requested by email. Variable-size deterministic synthetic payloads (32/128/512 KiB at 4 Hz) are load surrogates, never playable NTCs. A supplied payload directory can replace synthetic data; choose only whole valid files, never truncate models in this mode.

Approaches: (1) application sleeps only (rejected: do not exercise QUIC recovery); (2) Linux tc netem in isolated Docker (preferred existing setup); (3) seeded userspace UDP impairment proxy if Docker cannot start (explicit approximation, measure counters and avoid claiming kernel equivalence).

Controller compares fixed-low, fixed-high, queue-only, RTT-only and composite feedback. Use 250 ms sampling, conservative hysteresis, bounded producer queue and explicit dropped-before-admission records. No cancellation of admitted reliable data. Choose future workload size only; no claims that lowering size preserves a complete NTC. Production labels unavailable: test against configured impairment intervals and separately against deadline pressure. Report false alarms, missed detections, reaction/recovery delay, useful on-time bytes and switch count. Compare same workload cadence and profile schedule across policies, repeat random trials, separate calibration/holdout results; report failures rather than suppressing them.
