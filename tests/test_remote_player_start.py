import os
from pathlib import Path
import socket
import subprocess
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_gpu1_live_player.sh'


def run_start(tmp_path):
    return subprocess.run(['bash', str(SCRIPT)], env=dict(os.environ, Q3DGSTREAM_GPU_ROOT=str(tmp_path)), capture_output=True, text=True, timeout=10)


def test_reconnect_keeps_active_session_without_reinitializing_gpu(tmp_path):
    desktop = tmp_path / 'q3dg-desktop'
    desktop.mkdir()
    (desktop / 'player.pid').write_text(str(os.getpid()))
    (desktop / 'latest-live').write_text('existing-session')
    result = run_start(tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'already running' in result.stdout
    assert (desktop / 'latest-live').read_text() == 'existing-session'


def test_occupied_quic_port_preserves_existing_metadata(tmp_path):
    desktop = tmp_path / 'q3dg-desktop'
    desktop.mkdir()
    (desktop / 'latest-live').write_text('existing-session')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener:
        try:
            listener.bind(('127.0.0.1', 5003))
        except OSError:
            pytest.skip('port 5003 already in use')
        result = run_start(tmp_path)
    assert result.returncode != 0
    assert 'port 5003' in result.stderr
    assert (desktop / 'latest-live').read_text() == 'existing-session'
