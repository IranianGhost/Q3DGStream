"""Resample actual rendered P0-P3 display events at a fixed playout clock.

Hold the last captured display until the next display timestamp. Never use a
future-arriving enhancement to score an earlier tick. Requires exact-camera
FP32 reference states and an initial image from the SAME capture setup.
"""
import argparse,csv,json,math
from pathlib import Path
from bisect import bisect_right
from evaluation_support import score_sequence


def sample_display_events(events,frames,fps,playout_ms,initial_image):
    if frames<1 or not math.isfinite(fps) or fps<=0 or not math.isfinite(playout_ms) or playout_ms<0:
        raise ValueError('invalid playout clock')
    events=sorted(events,key=lambda r:float(r['display_ms']))
    for r in events:
        if not math.isfinite(float(r['display_ms'])) or float(r['display_ms'])<0 or not r['file']:
            raise ValueError('invalid display event; --save-stages captures are required')
        if int(r['ntc']) < -1 or r['ready_through'] not in ['initial','P0','P1','P2','P3']:
            raise ValueError('invalid stage identity')
    times=[float(r['display_ms']) for r in events]
    rows=[]
    for i in range(frames):
        tick=playout_ms+(i+1)*1000/fps
        n=bisect_right(times,tick)-1
        r=events[n] if n>=0 else dict(file=str(initial_image),ntc=-1,ready_through='initial',display_ms=0)
        rows.append(dict(sample_ms=tick,reference_state=i+1,distorted=r['file'],
                         displayed_ntc=int(r['ntc']),ready_through=r['ready_through'],
                         display_age_ms=tick-float(r['display_ms'])))
    return rows


def merge_display_events(progressive,playback):
    if not playback:raise ValueError('per-tick display captures are required')
    result=[dict(r,event_kind='stage') for r in progressive]
    for r in playback:
        if not r.get('render_file') or not r.get('render_ready_ms'):
            raise ValueError('every tick must have a captured render; use --capture-every 1 without dropped captures')
        ntc=int(r['applied_up_to_ntc'])
        tier=int(r['applied_tiers_latest'])
        result.append(dict(display_ms=float(r['render_ready_ms']),file=r['render_file'],ntc=ntc,
                           ready_through=f'P{tier-1}' if ntc>=0 else 'initial',event_kind='tick'))
    return result


def run(a):
    progressive=list(csv.DictReader(a.events.open()))
    playback=list(csv.DictReader(a.playback.open()))
    if len(playback)!=a.frames or [int(r['ntc']) for r in playback]!=list(range(a.frames)):
        raise ValueError('expected one captured tick per source playout slot')
    events=merge_display_events(progressive,playback)
    rows=sample_display_events(events,a.frames,a.fps,a.playout_ms,a.initial.resolve())
    # Validate all inputs before creating the paired reference/distorted sequence.
    for row in rows:
        ref=a.reference/f"frame_{row['reference_state']:06}.png"
        if not ref.is_file() or not Path(row['distorted']).is_file():
            raise FileNotFoundError('missing exact reference/display image: '+str(ref)+' / '+row['distorted'])
    a.output.mkdir(parents=True,exist_ok=False)
    ref=a.output/'reference';dist=a.output/'displayed';ref.mkdir();dist.mkdir()
    for i,row in enumerate(rows):
        name=f'frame_{i:06}.png'
        (ref/name).symlink_to((a.reference/f"frame_{row['reference_state']:06}.png").resolve())
        (dist/name).symlink_to(Path(row['distorted']).resolve())
    with (a.output/'sample-map.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
    result=score_sequence(ref,dist,a.output/'scores',a.ffmpeg,a.vmaf,fps=a.fps)
    # Stage-event PSNR/SSIM/LPIPS use each NTC's own FP32 state, independently
    # of clock-sampled QoE/VMAF. Never call irregular event samples a video.
    import numpy as np
    from PIL import Image
    import torch,lpips
    from evaluation_support import image_pair_metrics
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=lpips.LPIPS(net='alex').eval().to(device)
    stage_rows=[]
    for event in progressive:
        gt=a.reference/f"frame_{int(event['ntc'])+1:06}.png"
        if not gt.is_file() or not Path(event['file']).is_file():raise FileNotFoundError('missing exact stage reference/capture')
        x=np.asarray(Image.open(gt).convert('RGB'),dtype=np.float32)/255
        y=np.asarray(Image.open(event['file']).convert('RGB'),dtype=np.float32)/255
        stage_rows.append(dict(ntc=int(event['ntc']),ready_through=event['ready_through'],display_ms=float(event['display_ms']),**image_pair_metrics(x,y,model,device)))
    if stage_rows:
        with (a.output/'stage-quality.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,stage_rows[0]);w.writeheader();w.writerows(stage_rows)
    for score,row in zip(result['frames'],rows):score.update(row)
    result.update(fps=a.fps,playout_ms=a.playout_ms,
      caveat='Offline resampling of actual captured renderer outputs, not a new live run or physical display capture. Camera/reference alignment must be verified. PNG readback can affect source timings.')
    (a.output/'timeline-quality.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events',type=Path,required=True);p.add_argument('--playback',type=Path,required=True);
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,required=True);p.add_argument('--fps',type=float,default=30)
    p.add_argument('--playout-ms',type=float,default=500);p.add_argument('--ffmpeg',required=True);p.add_argument('--vmaf')
    run(p.parse_args())
