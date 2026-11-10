"""Recompute receiver playback from raw commit times; export presentation tables."""
import argparse,csv,json,statistics
from pathlib import Path
from .playback import evaluate


def analyze(roots,output):
 output.mkdir(parents=True,exist_ok=True);rows=[]
 for root in roots:
  for f in sorted(root.glob('*/run/summary.json')):
   s=json.loads(f.read_text());frames=json.loads((f.parent/'frames.json').read_text())
   name=f.parent.parent.name
   for buffer in (1,2,4):
    p=evaluate(frames,s['fps'],buffer,s['elapsed_s'],round(s['duration_s']*s['fps']))
    rows.append(dict(run=name,buffer=buffer,valid=s['valid'],transfer_seconds=s['elapsed_s'],
       completed=s['completed'],generated=s['generated'],bytes=sum(f['bytes'] for f in frames if f['status']=='complete'),
       failure=s['failure'],**p))
 (output/'PLAYBACK.json').write_text(json.dumps(rows,indent=2))
 with (output/'PLAYBACK.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# Playback and loss experiments','', '|Run|Transfer s|Initial buffer s|Startup s|Stalls|Stall s|Played s|Valid|', '|---|---:|---:|---:|---:|---:|---:|---|']
 for r in rows:
  if r['buffer']!=2:continue
  startup='none' if r['startup_seconds'] is None else f"{r['startup_seconds']:.2f}"
  lines.append(f"|{r['run']}|{r['transfer_seconds']:.2f}|2|{startup}|{r['stall_count']}|{r['stall_seconds']:.2f}|{r['played_seconds']:.2f}|{r['valid']}|")
 lines+=['','## Limits','','Receiver playback is a media-time simulation, not GPU rendering. Playback requires contiguous complete intervals. Incomplete sequences are censored at observation end; fully received sequences can finish locally without more network data.','', 'VOD transfer latency from producer enqueue is not a media deadline. Use startup, stalls and lateness against the initial playback schedule. Initial buffer sensitivity is replayed offline; only the 2 s setting drives the online buffer controller.', '', 'FP32 files are real; 62,296/66,741-byte alternatives are synthetic loads matching documented external formats. They do not establish quality or recurrent-model decodability.', '', 'Window/CC ablation uses 1,200-byte datagrams; VOD and follow-up trials use 1,400-byte datagrams. Loss realizations differ between runs; two repeats are exploratory. A larger timeout is a recovery budget, not a throughput repair.']
 (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 return rows

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.roots,a.output)
