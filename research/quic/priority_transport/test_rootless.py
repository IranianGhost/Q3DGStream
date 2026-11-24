from pathlib import Path
import os,subprocess

SCRIPTS=Path(__file__).resolve().parents[3]/'scripts'

def test_sudo_shim_rejects_host_namespace_even_with_ready_flag():
    env=dict(os.environ,QS_REPRO_ROOTLESS_READY='1')
    result=subprocess.run([str(SCRIPTS/'rootless-bin/sudo'),'-v'],env=env)
    assert result.returncode==65
