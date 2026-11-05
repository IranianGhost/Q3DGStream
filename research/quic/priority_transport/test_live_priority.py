from unittest.mock import patch
from live_priority import LivePriorityState
from live_tcp import LiveTCPState


def test_metadata_defers_cuda_initialization_until_checkpoint(tmp_path):
    state = LivePriorityState(str(tmp_path), total_frames_hint=4)
    try:
        with patch.object(LiveTCPState, "_ensure_ntc_single") as initialize:
            state._ensure_ntc_single(None)
            initialize.assert_not_called()
            assert state.err == ""
            state.ntc_paths[0] = tmp_path / "NTC_000000.pth"
            state._ensure_ntc_single(None)
            initialize.assert_called_once_with(None)
    finally:
        state.request_stop()
