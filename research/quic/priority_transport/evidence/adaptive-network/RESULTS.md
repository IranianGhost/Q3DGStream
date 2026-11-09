# Adaptive QUIC load experiment results

Three dynamic repetitions per policy; medians below. Deadline: 1 second.

|Policy|On-time frames|On-time MiB|Dropped|p95 latency (ms)|Max backlog MiB|Switches|
|---|---:|---:|---:|---:|---:|---:|
|fixed-low|61|1.91|0|1584|0.25|0|
|fixed-high|22|11.00|21|7082|4.50|0|
|queue|49|13.28|5|2786|1.25|6|
|rtt|39|15.62|20|6911|3.41|4|
|composite|48|13.38|4|2606|1.03|4|

## Signal discrimination (fixed-high, holdout repeats 2 and 3)

|Signal|Network precision|Network recall|False-positive rate|Next-second deadline F1|
|---|---:|---:|---:|---:|
|queue_age|0.39|1.00|0.76|0.95|
|queue_growth|0.24|0.30|0.46|0.46|
|rtt|1.00|0.87|0.00|0.57|
|rtt_variance|0.37|1.00|0.81|0.76|
|packet_loss|0.96|0.50|0.01|0.38|
|quic_buffer|0.30|0.76|0.83|0.77|
|app_unacked|0.31|0.89|0.96|0.84|
|ack_delay|0.76|0.54|0.08|0.49|
|delivery_deficit|0.60|1.00|0.32|0.79|

## Real checkpoint transfers

|Run|Complete|Generated|On-time|Valid|
|---|---:|---:|---:|---|
|real-good|2|2|2|True|
|real-loss|2|2|0|True|

## Failures

[{"run": "real-burst", "returncode": 1, "error": "TimeoutError()"}]

## Interpretation limits

- Detection/clear times are first matching samples after transition, not sustained state changes; an alarm may predate impairment.
- Latency p95 uses the lower empirical order statistic at floor((n-1)*0.95); tiny real-file samples are not tail estimates.
- Failed runs remain listed but are excluded from performance/predictor aggregates; coverage is explicit.
- Delivery-deficit alarm omitted for adaptive negative controls because offered rate changes with policy.
- Only original FP32 checkpoints available; exact Publio INT4/saliency assets requested.
- Synthetic size adaptation does not imply valid NTC representations or measured visual quality.
- Signal thresholds frozen before main runs; repeat 1 calibration, repeats 2/3 holdout. No statistical significance claim.
- Netem modifies only container loopback, affecting both directions. Kernel random loss traces differ across runs.
- Transition uses qdisc change preserving queue; no host qdisc changes.
- Receiver deadline computed from complete file receipt; playback and GPU inference are not measured.
- Queue pressure can mean application overload or slow receiver, not uniquely network impairment.
- Congestion/RTT/buffer diagnostics access aioquic 1.7.2 internals. Loss is QUIC-declared loss, not exact wire drops.
- All regular workload files priority P0: this experiment isolates load adaptation, not prioritization gains.
