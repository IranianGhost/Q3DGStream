from research.quic.priority_transport.analyze import stage_latencies


def test_stage_latency_uses_last_chunk_including_out_of_order_arrival():
    rows = [
        dict(path="a", priority=0, release_ms=10, received_ms=40),
        dict(path="b", priority=0, release_ms=10, received_ms=20),
        dict(path="a", priority=0, release_ms=10, received_ms=25),
    ]
    assert stage_latencies(rows, 0) == [10, 30]
