import csv,hashlib,json
from pathlib import Path
import pytest
from PIL import Image
from trajectory_quality import trajectory,compare_captures

def fixture(tmp_path):
    trial=tmp_path/'case/priority';group=trial/'received/NTC_000000';group.mkdir(parents=True)
    (group/'P0.bin').write_bytes(b'group payload')
    live=trial/'live';live.mkdir()
    with (live/'apply_events.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['ntc','tick','rendered_this_tick','applied_tiers','precision_bits']);w.writeheader()
        w.writerow(dict(ntc=0,tick=0,rendered_this_tick=1,applied_tiers=1,precision_bits=2))
    prepared=tmp_path/'prepared';prepared.mkdir()
    (prepared/'sequence_manifest.json').write_text(json.dumps(dict(manifests={'0':{'variants':{'2':{'groups':{'P0':{'sha256':hashlib.sha256(b'group payload').hexdigest()}}}}}})))
    return trial,group

def test_applied_byte_change_invalidates_quality_cache(tmp_path):
    trial,group=fixture(tmp_path);row=dict(trial=str(trial))
    original=trajectory(row,tmp_path,1)
    assert len(original)==64
    (group/'P0.bin').write_bytes(b'changed payload')
    with pytest.raises(ValueError,match='checksum'):trajectory(row,tmp_path,1)

def test_missing_applied_update_is_rejected(tmp_path):
    trial,group=fixture(tmp_path)
    with pytest.raises(ValueError,match='Incomplete'):trajectory(dict(trial=str(trial)),tmp_path,2)

def test_identical_live_samples_are_required_for_reuse(tmp_path):
    for name in ['a','b']:
        p=tmp_path/name/'live/renders';p.mkdir(parents=True)
        Image.new('RGB',(8,8),(20,50,80)).save(p/'frame_000001.png')
    assert compare_captures(tmp_path/'a',tmp_path/'b',1,1)==1
    Image.new('RGB',(8,8),(21,50,80)).save(tmp_path/'b/live/renders/frame_000001.png')
    with pytest.raises(ValueError,match='differ'):compare_captures(tmp_path/'a',tmp_path/'b',1,1)
