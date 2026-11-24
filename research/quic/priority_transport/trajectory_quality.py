"""Reuse offline scores only for byte-identical applied trajectories.

Network trials remain independent. Every applied group is checked against its
manifest and every sampled live PNG must match the independently scored primary.
Run parallel_quality.py directly instead if independent replay of every
identical trajectory is desired. This cache never reuses timing measurements.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def trajectory(row,study,frames):
    trial=Path(row['trial'])
    if not trial.is_absolute():trial=study/trial
    manifest_path=study/(row.get('prepared_manifest') or 'prepared/sequence_manifest.json')
    manifests=json.loads(manifest_path.read_text())['manifests']
    with (trial/'live/apply_events.csv').open(newline='') as f:events=list(csv.DictReader(f))
    if len(events)!=frames:raise ValueError('Incomplete applied trajectory')
    parts=[]
    for i,e in enumerate(events):
        count=int(e['applied_tiers']);bits=e['precision_bits']
        if int(e['ntc'])!=i or int(e['tick'])!=i or e['rendered_this_tick']!='1' or not 1<=count<=4 or bits not in ('2','4'):
            raise ValueError('Invalid applied trajectory')
        hashes=[]
        for tier in range(count):
            expected=manifests[str(i)]['variants'][bits]['groups'][f'P{tier}']['sha256']
            actual=sha(trial/'received'/f'NTC_{i:06}'/f'P{tier}.bin')
            if actual!=expected:raise ValueError('Applied group checksum differs')
            hashes.append(actual)
        parts.append([i,bits,count,hashes])
    return hashlib.sha256(json.dumps(parts,separators=(',',':')).encode()).hexdigest()


def compare_captures(primary,other,frames,capture_every):
    from PIL import Image
    import numpy as np
    expected={f'frame_{i+1:06}.png' for i in range(0,frames,capture_every)}
    a=Path(primary)/'live/renders';b=Path(other)/'live/renders'
    if not expected or any({p.name for p in d.glob('frame_*.png')}!=expected for d in [a,b]):
        raise ValueError('Missing expected live captures')
    for name in expected:
        # Primary PNGs were decoded by the scorer; identical encoded bytes imply identical pixels.
        if sha(a/name)==sha(b/name):continue
        with Image.open(a/name) as left,Image.open(b/name) as right:
            if not np.array_equal(np.asarray(left.convert('RGB')),np.asarray(right.convert('RGB'))):
                raise ValueError('Live captures differ for purported identical trajectory')
    return len(expected)


def main(a):
    study=a.study.resolve();out=a.out.resolve();work=a.work.resolve()
    work.mkdir(parents=True,exist_ok=True)
    config=json.loads((study/'study_config.json').read_text());frames=config['frames'];capture_every=config.get('capture_every',30)
    with (study/'study.csv').open(newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    groups={};keys={}
    for row in rows:
        key=trajectory(row,study,frames);groups.setdefault(key,[]).append(row);keys[row['trial']]=key
    if len(keys)!=len(rows):raise ValueError('Duplicate trial rows')
    reduced=work/'unique-study';reduced.mkdir(exist_ok=True)
    for p in study.iterdir():
        if p.is_dir() and p.name.startswith('prepared'):
            target=reduced/p.name
            if not target.exists():target.symlink_to(p,target_is_directory=True)
    shutil.copyfile(study/'study_config.json',reduced/'study_config.json')
    primaries=[g[0] for g in groups.values()]
    table=reduced/'study.csv'
    if table.exists():
        with table.open(newline='') as f:assert list(csv.DictReader(f))==primaries
    else:
        with table.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(primaries)
    reference_hashes={p.name:sha(p) for p in sorted(a.reference.glob('frame_*.png'))}
    if len(reference_hashes)!=frames:raise ValueError('Reference length differs')
    viewer=Path(config['viewer']);runtime=a.scorer.parent/'runner'
    if not runtime.is_dir():runtime=a.scorer.parent
    context=dict(reference_sha256=hashlib.sha256(json.dumps(reference_hashes,sort_keys=True).encode()).hexdigest(),
       scene_sha256=sha(Path(config['scene'])/'init_3dgs.ply'),
       viewer_sha256={str(p.relative_to(viewer)):sha(p) for p in sorted(viewer.rglob('*')) if p.is_file() and p.suffix in ('.py','.glsl')},
       runtime_sha256={p.name:sha(p) for p in sorted(runtime.glob('*.py'))},
       camera_sha256=sha(a.camera_config) if a.camera_config else None)
    plan=dict(context=context,study_sha256=sha(study/'study.csv'),config_sha256=sha(study/'study_config.json'),
              trajectories={key:[r['trial'] for r in group] for key,group in groups.items()},
              reuse='same applied group bytes and order; all live samples must match primary; timings never reused')
    planpath=work/'TRAJECTORIES.json'
    if planpath.exists():assert json.loads(planpath.read_text())==plan
    else:planpath.write_text(json.dumps(plan,indent=2)+'\n')
    if a.seed_work:
        completed={};seed_columns=None
        for table in a.seed_work.glob('worker*/quality/study_with_replay.csv'):
            with table.open(newline='') as stream:
                reader=csv.DictReader(stream);seed_columns=reader.fieldnames
                for record in reader:completed[record['trial']]=(record,table.parent/Path(record['trial']).parent.name)
        seeded={}
        for i,primary in enumerate(primaries):
            if primary['trial'] not in completed:continue
            record,source=completed[primary['trial']]
            if int(record['scored_frames'])!=frames:raise ValueError('Incomplete seed score')
            destination=work/'shards'/f'worker{i%min(a.workers,len(primaries))}'/'quality'
            if (destination/'study_with_replay.csv').exists():continue
            destination.mkdir(parents=True,exist_ok=True)
            shutil.copytree(source,destination/source.name,dirs_exist_ok=True)
            seeded.setdefault(destination,[]).append(record)
        for destination,records in seeded.items():
            with (destination/'study_with_replay.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=seed_columns);writer.writeheader();writer.writerows(records)
        print('REUSED_COMPLETED_INDEPENDENT_REPLAYS',sum(map(len,seeded.values())),flush=True)
    print('TRAJECTORIES',len(groups),'independent replays for',len(rows),'independent network trials',flush=True)
    cmd=[sys.executable,'-u',str(Path(__file__).with_name('parallel_quality.py')),'--study',str(reduced),
         '--out',str(work/'unique-quality'),'--work',str(work/'shards'),'--reference',str(a.reference),
         '--scorer',str(a.scorer),'--vmaf',str(a.vmaf),'--workers',str(a.workers)]
    for flag in ['bundle','viewer','camera_config']:
        value=getattr(a,flag)
        if value:cmd+=['--'+flag.replace('_','-'),str(value)]
    subprocess.run(cmd,check=True)
    with (work/'unique-quality/study_with_replay.csv').open(newline='') as f:
        reader=csv.DictReader(f);columns=reader.fieldnames;scored={r['trial']:r for r in reader}
    outputs=[];out.mkdir(parents=True,exist_ok=True)
    for row in rows:
        key=keys[row['trial']];primary=groups[key][0];score=scored[primary['trial']]
        count=compare_captures(primary['trial'],row['trial'],frames,capture_every)
        if count!=int(score['identical_captures']):raise ValueError('Primary offline replay captures were incomplete')
        summary=json.loads((Path(row['trial'])/'live/playback_summary.json').read_text())
        if summary['capture_dropped']!=0:raise ValueError('Dropped live capture')
        result={**row,**{k:v for k,v in score.items() if k not in fields},
                **{k:score[k] for k in ['mean_psnr_db','mean_ssim','mean_lpips','mean_vmaf']},
                'quality_trajectory_sha256':key,'quality_reused_from':'' if row is primary else primary['trial']}
        if row is not primary:result['quality_source']='cached byte-identical applied trajectory; every sampled live pixel checked'
        source=work/'unique-quality'/Path(primary['trial']).parent.name
        target=out/Path(row['trial']).parent.name;shutil.copytree(source,target,dirs_exist_ok=True)
        if row is not primary:
            (target/'SCORE_REUSE.json').write_text(json.dumps(dict(source_trial=primary['trial'],trajectory_sha256=key,
                verified_applied_updates=frames,identical_live_samples=count,maximum_live_rgb_difference=0,
                note='Quality reused only; startup/stall/arrival/bytes remain this trial original measurements'),indent=2)+'\n')
        outputs.append(result)
    with (out/'study_with_replay.pending').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(outputs[0]));w.writeheader();w.writerows(outputs)
    (out/'study_with_replay.pending').replace(out/'study_with_replay.csv')
    (out/'SCORING.json').write_text(json.dumps({**plan,'independent_network_trials':len(rows),
      'independently_replayed_trajectories':len(groups),'represented_image_pairs':frames*len(rows),
      'independently_scored_image_pairs':frames*len(groups),'identical_live_samples':len(rows)*len(range(0,frames,capture_every))},indent=2)+'\n')
    print('VALIDATED',len(rows),'trials;',len(groups),'independent image replays;',len(rows)-len(groups),'byte-identical quality reuses',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['study','out','work','reference','scorer','vmaf']:p.add_argument('--'+flag,type=Path,required=True)
    for flag in ['bundle','viewer','camera-config']:p.add_argument('--'+flag,type=Path)
    p.add_argument('--seed-work',type=Path,help='Preserved shards from the identical scorer/reference; primary rows only')
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if not 1<=a.workers<=16:p.error('workers must be between 1 and 16')
    main(a)
