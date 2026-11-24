import argparse,hashlib,json
from pathlib import Path
import pytest
from adaptive import build_command,trace_profile


def assets(tmp_path):
    root=tmp_path/'assets';(root/'prepared').mkdir(parents=True)
    (root/'prepared/sequence_manifest.json').write_text(json.dumps(dict(frames=299)))
    trace=tmp_path/'wifi.txt';trace.write_text('0 60.2\n1 80\n2 27.4\n')
    (root/'assets.json').write_text(json.dumps(dict(schema='q3dg_paired_assets_v1',frames=299,
      scene=str(tmp_path/'scene'),bootstrap_dir=str(tmp_path/'wire'),trace=str(trace),
      trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())))
    return root


def args(root,**changes):
    values=dict(assets=root,out=root.parent/'run',profile='campus',selector='deadline',precision='adaptive',
                tiers=4,frames=299,startup_ms=500.,max_buffer_frames=60,priority='tiered',submission='p0_first',
                estimator='hybrid',capture_every=30,camera_config=None)
    values.update(changes);return argparse.Namespace(**values)


def test_campus_preset_keeps_precision_selection_and_quic_priority_separate(tmp_path):
    cmd=build_command(args(assets(tmp_path)))
    assert cmd[cmd.index('--precision-policy')+1]=='adaptive'
    assert cmd[cmd.index('--tier-selection')+1]=='adaptive'
    assert cmd[cmd.index('--quinn-priority')+1]=='tiered'
    assert cmd[cmd.index('--scheduling-mode')+1]=='p0_first'
    assert cmd[cmd.index('--wifi-impairment')+1]=='campus'
    assert cmd[cmd.index('--vod-startup-frames')+1]=='15'
    assert cmd[cmd.index('--feedback-ms')+1]=='50'
    assert '--live' in cmd and '--netns' in cmd


def test_controlled_profile_has_two_independent_delay_directions(tmp_path):
    cmd=build_command(args(assets(tmp_path),profile='controlled',selector='buffer'))
    assert cmd[cmd.index('--rate-profile')+1]=='0:28,3:70,5:16.8,7:28'
    assert cmd[cmd.index('--delay-ms')+1]=='10'
    assert '--wifi-impairment' not in cmd
    assert cmd[cmd.index('--tier-selection')+1]=='buffer'


def test_fixed_int2_base_and_custom_camera_are_explicit(tmp_path):
    cmd=build_command(args(assets(tmp_path),precision='int2',selector='fixed',tiers=1,camera_config=tmp_path/'camera.json'))
    assert cmd[cmd.index('--precision-policy')+1]=='fixed2'
    assert cmd[cmd.index('--fixed-tiers')+1]=='1'
    assert cmd[cmd.index('--camera-config')+1]==str(tmp_path/'camera.json')


@pytest.mark.parametrize('change',[dict(startup_ms=-1),dict(startup_ms=float('nan')),dict(frames=300),dict(selector='fixed')])
def test_invalid_or_ambiguous_presets_fail_before_starting_network(tmp_path,change):
    with pytest.raises(ValueError):build_command(args(assets(tmp_path),**change))


def test_trace_is_capped_and_must_have_strict_monotonic_time(tmp_path):
    f=tmp_path/'trace';f.write_text('0 60.2\n1 80\n2 27.4\n')
    assert trace_profile(f)[0]=='0:60.2,1:70,2:27.4'
    f.write_text('0 30\n0 40\n')
    with pytest.raises(ValueError):trace_profile(f)


def test_partial_sequence_requires_separate_verified_workspace(tmp_path):
    with pytest.raises(ValueError, match='Frame count must match'):
        build_command(args(assets(tmp_path), frames=3))


def test_existing_run_is_never_overwritten(tmp_path):
    from adaptive import run
    a=args(assets(tmp_path));a.dry_run=False;a.out.mkdir()
    with pytest.raises(ValueError, match='already exists'):run(a)


def test_public_run_command_uses_only_supported_runner_flags(tmp_path):
    import subprocess,sys
    command=build_command(args(assets(tmp_path)))
    result=subprocess.run(command+['--help'],capture_output=True,text=True)
    assert result.returncode==0
    # argparse --help exits early, so check generated option tokens against help.
    for value in command[3:]:
        if value.startswith('--'):assert value in result.stdout
