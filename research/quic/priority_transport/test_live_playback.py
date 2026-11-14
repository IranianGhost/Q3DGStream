from research.quic.priority_transport.live_playback import LivePlayback, choose_optional_bytes


def test_startup_and_stall_wait_for_contiguous_updates():
    clock = LivePlayback(updates=3, fps=2, prebuffer=0.5)
    assert clock.select(10.0, True, 0) is None
    assert clock.select(10.2, True, 1) == 0
    clock.present(10.3, 0)
    assert clock.select(10.7, True, 1) == 0
    assert clock.select(10.8, True, 1) == 1
    clock.present(10.8, 1)
    assert clock.select(11.4, True, 1) == 1
    assert clock.select(11.6, True, 2) == 2
    clock.present(11.7, 2)
    assert clock.stalls[0][0] == 11.3
    assert clock.stalls[0][1] == 11.7
    assert clock.select(12.3, True, 3) == 3
    assert not clock.finished
    clock.present(12.4, 3)
    assert clock.finished
    assert len(clock.stalls) == 1


def test_short_clip_caps_prebuffer_at_duration():
    clock = LivePlayback(updates=1, fps=30, prebuffer=2)
    assert clock.select(5, False, 1) is None
    assert clock.select(6, True, 1) == 0


def test_optional_load_requires_fresh_feedback_and_budget():
    snapshot = dict(at=10.0, buffer_seconds=2.0, committed_Bps=3000)
    assert choose_optional_bytes(snapshot, 10.1, 1000, 1, 1, 1000) == 1000
    assert choose_optional_bytes(snapshot, 12.0, 1000, 1, 1, 1000) == 0
    assert choose_optional_bytes(dict(snapshot, buffer_seconds=0), 10.1, 1000, 1, 1, 1000) == 0
    assert choose_optional_bytes(snapshot, 10.1, 2500, 1, 1, 1000) == 0
