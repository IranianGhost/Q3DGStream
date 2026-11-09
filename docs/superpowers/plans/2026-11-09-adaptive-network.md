# Adaptive Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test real payload transport and quantify signals usable for adapting offered load under impairments.
**Architecture:** Instrument existing transport; independent periodic producer and bounded queue; feedback controller selects next complete payload; real QUIC through isolated impairment harness; JSON traces and reproducible analysis.
**Tech Stack:** Python, aioquic 1.7.2, pytest, Linux tc netem/Docker where available.
**Spec:** docs/superpowers/specs/2026-11-09-adaptive-network-design.md

## Global Constraints
- Continue existing feature branch; preserve byte integrity and existing priority semantics.
- Never imply synthetic size levels are valid compressed NTC quality levels.
- Never mutate host network qdiscs; credentials/generated data stay untracked.
- Report all failed runs and unavailable exact Publio assets.

## Review Focus
- No ACKs/stalled path must trigger pressure, not false healthy zero throughput.
- Empty/idle queues must not manufacture capacity estimates.
- Receiver slowdown must not be described as proven network congestion.
- Reliable delivery may continue after workload end: drain cost counted separately.
- Controller must recover without flapping and obey queue limits.

### Task 1: Instrumentation and controller
- [x] Add failing tests in test_adaptive.py for overload, recovery hysteresis, idle/no-ACK behavior and queue limits.
- [x] Add telemetry callback to send_files; add adaptive.py controller with documented thresholds and reasons.
- [x] Run targeted and existing transport tests.

### Task 2: Reproducible workload/harness
- [x] Add experiment.py with independent producer, sampler and sender, exact-payload mode and synthetic size ladder.
- [x] Use netem if available, otherwise UDP proxy with seeded loss/jitter/rate queues; record actual applied profiles/counters.
- [x] Verify clean byte integrity and loss recovery plus queue bounds in integration tests.

### Task 3: Experiment matrix and selection
- [x] Run baseline/feedback comparisons across steady and changing conditions, plus receiver-delay control and original NTC files.
- [x] Analyze signals, deadline misses, on-time bytes, backlog and decisions; include failure cases.
- [x] Write English report and reproduction instructions, review branch, commit and push authorized review branch.

## Ledger
- PR #1 open/draft, unmerged at start. Follow-up and request for exact payloads sent to Publio.
- Ruling: prior user delegation plus explicit implement request authorizes design choices without another approval round.

- Main matrix: 27 attempted, 26 valid; real-burst failed with a 15 s chunk ACK timeout. Retained failure evidence.
- Verification: 42 tests and 755 subtests pass in scoped suites; repository-wide collection fails in pre-existing progressive receiver test.
- Review fixes: preserve qdisc across transitions, cycle both real files, distinguish receiver commit/ACK time, bound readiness and guarantee cleanup, exclude failed runs from aggregates, omit invalid adaptive-control deficit comparisons.
- Ruling: use existing feature branch as explicitly requested; no new worktree needed. Exact Publio compressed inputs remain external pending email.
