# QUIC priority integration plan

> Execute inline using superpowers:executing-plans. User authorized autonomous decisions and implementation in Publio's repository.

Goal: integrate bounded chunk scheduling into the existing dynamic sender and test actual QUIC transport on local network profiles.
Architecture: opt-in chunk transport preserves upstream QNP01 file bytes and stage priorities. A separate headless receiver atomically reconstructs files. Keep frame/additions dependencies as barriers. Legacy mode remains default because its missing quic_transport wire protocol cannot be inferred safely.
Tech: Python 3.10+, aioquic 1.7.2, pytest, upstream QNP01 codec, Linux tc netem.
Spec: user request in this conversation, preserving current renderer semantics and clearly separating synthetic fixtures from trained NTCs.

Constraints: no MoQT; no GPU or weights committed; local-only receiver; TLS verification; no push. Existing clean clone on codex/quic-priority-transport is the work location explicitly requested by user.
Review focus: malformed chunks/path traversal; mismatched digests; partial files on failure; dependency barriers; priority arrival under congestion.

1. Tests first: scheduler fairness/late urgency, file reassembly/out-of-order/integrity/path guards. Implement priority_transport/core.py and transport.py.
2. Integrate opt-in transport into dynamic_priority_adapter/send_dynamic_priority.py. Metadata/additions commit before stages, stages complete before next frame. Test invocation/dependency ordering without GPU.
3. Benchmark shared transport using valid synthetic QNP01 stages, validate with upstream decoder. Compare sequential file reference, FIFO, strict, weighted under six netem profiles, three repeats. Exact legacy transport unavailable; do not claim measured speedup over upstream binary.
4. Document differences and limitations, run existing CPU tests and compileall. Fresh review. Complete English presentation with separately labelled earlier lab results and integrated results.

Ledger: baseline 16 CPU tests passed, 3 numpy tests skipped in system Python. No trained weights or .qblk files found in workspace. Missing upstream quic_transport/live_quic modules confirmed. Native QUIC scheduler remains unchanged.

Review: distinguish chunk latency from completed-stage latency. Added stage aggregation and regression test. Removed performance-improvement assertions so neutral/negative results remain reportable. Production fixture has eight chunks per frame versus a 16-slot window: no demonstrated priority latency gain in that pipeline. Explicit in documentation and presentation. Eight implementation tests passed before network run; analysis regression added afterward. All 16 existing CPU tests passed with numpy installed.

Completed: 54 stress transfers passed (6,912 QNP01 files, 13,824 chunks). Real adapter helper passed on all six network profiles. Final checks: 9 transport/analysis tests, 16 existing CPU tests, compileall and git diff --check passed. Review reporting issues resolved. Nine English slides saved to Spaces page_31886e21518c81918c2a044e5e17682e, with editable comparisons/results and explicit scope. No trained NTC or GPU result claimed. Local branch only, no push.
