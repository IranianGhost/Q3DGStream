"""Shard offline replay/scoring only; never parallelize timed playback trials.

Each shard invokes the existing scorer. Completed rows and captures are verified
before an atomic merge. A fixed plan makes interrupted invocations resumable.
"""
import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


def read_csv(path):
    with path.open(newline='') as stream:
        r = csv.DictReader(stream)
        return r.fieldnames, list(r)


def write_csv(path, fields, rows):
    pending = path.with_suffix('.pending')
    with pending.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    pending.replace(path)


def main(a):
    study, out, work = (p.resolve() for p in (a.study, a.out, a.work))
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fields, rows = read_csv(study / 'study.csv')
    config = json.loads((study / 'study_config.json').read_text())
    frames = config['frames']; capture_every = config.get('capture_every', 30)
    assert rows and len({r['trial'] for r in rows}) == len(rows)
    expected = {f'frame_{i+1:06}.png' for i in range(frames)}
    reference = a.reference.resolve()
    assert {p.name for p in reference.glob('frame_*.png')} == expected
    scorer = a.scorer.resolve()
    plan = dict(study=str(study), study_sha256=hashlib.sha256((study/'study.csv').read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256((study/'study_config.json').read_bytes()).hexdigest(),
                scorer=str(scorer), scorer_sha256=hashlib.sha256(scorer.read_bytes()).hexdigest(),
                reference=str(reference), workers=min(a.workers,len(rows)), vmaf=str(a.vmaf.resolve()),
                cases=[r['trial'] for r in rows])
    saved = work / 'plan.json'
    if saved.exists():
        assert json.loads(saved.read_text()) == plan, 'Scoring plan changed; use a fresh workspace'
    else:
        saved.write_text(json.dumps(plan,indent=2)+'\n')
    supported_flags=subprocess.check_output([sys.executable,str(scorer),'--help'],text=True)
    jobs=[]
    for i in range(plan['workers']):
        folder=work/f'worker{i}'; shard=folder/'study';shard.mkdir(parents=True,exist_ok=True)
        subset=rows[i::plan['workers']]
        for item in study.iterdir():
            if item.is_dir() and item.name.startswith('prepared'):
                link=shard/item.name
                if not link.exists():link.symlink_to(item,target_is_directory=True)
                assert link.resolve()==item.resolve()
        shutil.copyfile(study/'study_config.json',shard/'study_config.json')
        if (shard/'study.csv').exists():assert read_csv(shard/'study.csv')==(fields,subset)
        else:write_csv(shard/'study.csv',fields,subset)
        jobs.append((folder,shard,subset))
    def run(job):
        folder,shard,subset=job
        cmd=[sys.executable,'-u',str(scorer),'--study',str(shard),'--out',str(folder/'quality'),
             '--reference-dir',str(reference),'--low-space','--resume',
             '--vmaf',str(a.vmaf.resolve())]
        if '--replay-scratch' in supported_flags:cmd+=['--replay-scratch',str(work/'scratch')]
        if a.bundle:cmd+=['--bundle',str(a.bundle.resolve())]
        if a.viewer:cmd+=['--viewer',str(a.viewer.resolve())]
        if a.camera_config:cmd+=['--camera-config',str(a.camera_config.resolve())]
        with (folder/'score.log').open('a') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        print('SCORED',folder.name,len(subset),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=plan['workers']) as pool:list(pool.map(run,jobs))
    results={}; result_fields=None; transfers=[]
    for folder,_,subset in jobs:
        columns,completed=read_csv(folder/'quality/study_with_replay.csv')
        if result_fields is None:result_fields=columns
        assert columns==result_fields
        assert {r['trial'] for r in completed}=={r['trial'] for r in subset}
        for row in completed:
            trial=Path(row['trial']);case=trial.parent.name
            assert row['trial'] not in results
            assert int(row['scored_frames'])==frames
            n=len(range(0,frames,capture_every)) if capture_every else 0
            assert n>0 and int(row['overlapping_captures'])==int(row['identical_captures'])==n
            assert int(row['max_absolute_rgb_difference_8bit'])==0
            assert {p.name for p in (trial/'live/renders').glob('frame_*.png')}=={f'frame_{i+1:06}.png' for i in range(0,frames,capture_every)}
            assert len(list(csv.DictReader((trial/'live/apply_events.csv').open())))==frames
            summary=json.loads((trial/'live/playback_summary.json').read_text())
            assert summary['capture_dropped']==0
            source=folder/'quality'/case
            # A retry can leave incomplete metadata; select the unique complete record.
            candidates=[]
            for q in source.rglob('quality.json'):
                data=json.loads(q.read_text())
                if len(data.get('frames',[]))==frames:candidates.append((q,data))
            assert len(candidates)==1,(case,[str(q) for q,_ in candidates])
            q,data=candidates[0]
            assert [f['image'] for f in data['frames']]==sorted(expected)
            assert data['reference']==row['reference']==str(reference)
            for metric,col in [('psnr','mean_psnr_db'),('ssim','mean_ssim'),('lpips','mean_lpips'),('vmaf','mean_vmaf')]:
                values=[float(f[metric]) for f in data['frames']]
                assert all(math.isfinite(x) for x in values)
                assert math.isclose(statistics.fmean(values),float(row[col]),rel_tol=1e-12,abs_tol=1e-10)
            results[row['trial']]=row;transfers.append((source,out/case))
    assert set(results)=={r['trial'] for r in rows}
    for original in rows:
        result=results[original['trial']]
        assert all(result.get(k)==v for k,v in original.items() if k not in {'mean_psnr_db','mean_ssim','mean_lpips','mean_vmaf'})
    out.mkdir(parents=True,exist_ok=True)
    # Copy small metric evidence, retaining shard sources for safe resume.
    for source,target in transfers:shutil.copytree(source,target,dirs_exist_ok=True)
    write_csv(out/'study_with_replay.csv',result_fields,[results[r['trial']] for r in rows])
    (out/'SCORING.json').write_text(json.dumps({**plan,'verified_cases':len(rows),'image_pairs':frames*len(rows),
        'live_replay_identical_captures':sum(int(r['identical_captures']) for r in results.values()),
        'timing':'offline only, after timed cases; no timed measurements were parallelized'},indent=2)+'\n')
    print('MERGED',len(rows),'cases',frames*len(rows),'image pairs',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['study','out','work','reference','scorer','vmaf']:p.add_argument('--'+flag,type=Path,required=True)
    for flag in ['bundle','viewer','camera-config']:p.add_argument('--'+flag,type=Path)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if not 1<=a.workers<=16:p.error('workers must be between 1 and 16')
    main(a)
