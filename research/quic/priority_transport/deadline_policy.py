"""Deterministic Python reference of the Quinn Rust tier decision rule.

Used for unit tests and research sweeps. Live sender independently implements
the same equation and logs its measured inputs per NTC.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    tiers: int
    payload_bytes: int
    deadline_feasible: bool
    predicted_arrival_ms: float
    steady_state_budget_bytes: float


def select_tiers(group_sizes, now_ms, deadline_ms, effective_mbps, fps,
                 rtt_ms, backlog_bytes, client_compute_ms, rtt_jitter_ms=0):
    if len(group_sizes) != 4 or any(n <= 0 for n in group_sizes):
        raise ValueError("Need four non-empty cumulative saliency groups")
    if effective_mbps <= 0 or fps <= 0 or rtt_ms < 0 or backlog_bytes < 0 or client_compute_ms < 0:
        raise ValueError("Invalid bandwidth or latency inputs")
    budget = effective_mbps * 1e6 / (8 * fps)
    backlog_ms = backlog_bytes * 8 / (effective_mbps * 1000)
    compute_feasible = client_compute_ms <= 1000 / fps
    chosen, chosen_size, arrival = 0, 0, 0.0
    running_size = 0
    for tier, length in enumerate(group_sizes, start=1):
        running_size += length
        predicted = (now_ms + backlog_ms +
                     running_size * 8 / (effective_mbps * 1000) +
                     rtt_ms / 2 + client_compute_ms + 2 * rtt_jitter_ms)
        if running_size <= budget and predicted <= deadline_ms and (compute_feasible or tier == 1):
            chosen, chosen_size, arrival = tier, running_size, predicted
        else:
            break
    if not chosen:
        chosen, chosen_size = 1, group_sizes[0]
        arrival = (now_ms + backlog_ms + chosen_size * 8 / (effective_mbps * 1000) +
                   rtt_ms / 2 + client_compute_ms + 2 * rtt_jitter_ms)
    return Decision(chosen, chosen_size,
                    compute_feasible and arrival <= deadline_ms and chosen_size <= budget,
                    arrival, budget)
