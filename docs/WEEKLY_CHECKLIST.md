# Weekly streaming experiments — 10 November 2026

- [x] Check upstream PR and continue existing branch; follow-up sent to Publio.
- [x] Audit repository sizes: original 7,968,015 bytes; keep05 packed 66,741; stages 62,296.
- [x] Verify compressed assets are external and unavailable locally; do not claim their bytes were tested.
- [x] Make experimental admission window configurable, default 128 chunks; compare 16 and 128.
- [x] Diagnose random-loss slowdown with original checkpoint, window and congestion-controller ablation.
- [x] Validate chosen transport configuration and burst recovery budget with before/after evidence.
- [x] Implement contiguous media-time prebuffer, playback progression and rebuffer accounting.
- [x] Replay deadlines using 1/2/4 seconds of initial playable media.
- [x] Implement buffer/rate-based future-size selection and finite-video prefetch.
- [x] Finish fixed/adaptive netem comparisons at repository-documented byte scales.
- [x] Select metrics and quantify startup/stall/volume trade-offs; preserve failures.
- [x] Update English report, reproducible demo, charts and presentation.
- [x] Verify tests, review and push branch.

Exact compressed NTC transfer and decoded-quality comparisons remain dependent on
external assets and same-frame alternative representations from Publio. Size-matched
payload tests exercise the network only. They are not interchangeable NTC models.

Execution plan: audit → window/CC ablation → playback and feedback evaluation →
evidence/report/demo → tests/review/push. Defaults: 128 chunks, 120-second per-chunk
recovery budget, cubic as a candidate pending ablation, prebuffer 2 s with 1/2/4 s
sensitivity analysis. No host network changes; isolated Docker netem only.

## External dependency / remaining limits

- [ ] Transfer Publio’s actual compressed variants and measure rendered quality: assets are external and have not been supplied.
- [ ] Establish sustained throughput recovery under random loss across broader paths: current tuning improves observed runtime, but random-loss congestion response remains.

18 retained experiments (19 physical including a superseded baseline), 17 completed
transfers and one deliberate timeout reproduction. All results/failures retained.
Playback is media-time simulation with initial scene assumed available, not GPU validation.
