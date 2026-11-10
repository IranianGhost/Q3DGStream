"""Local Docker/netem evaluation of window, loss recovery and playback-aware control."""
import json,subprocess,sys,time
from pathlib import Path
from .adaptive_matrix import tc,PROFILES

root=Path('data/outputs/playback-matrix');root.mkdir(parents=True,exist_ok=False)
results=[]

def run(name,opts,profile='loss',changes=()):
 out=root/name;out.mkdir();proc=None;applied=[];code=-1
 try:
  tc(PROFILES[profile]);applied.append(dict(t=0,profile=profile))
  with (out/'console.log').open('w') as log:
   proc=subprocess.Popen([sys.executable,'-m','research.quic.priority_transport.experiment','--cert','cert.pem','--key','key.pem','--output',str(out/'run'),'--profile',name,*opts],stdout=log,stderr=subprocess.STDOUT)
   limit=time.monotonic()+20
   while not (out/'run/ready.json').exists() and proc.poll() is None and time.monotonic()<limit:time.sleep(.02)
   if not (out/'run/ready.json').exists():raise RuntimeError('not ready')
   start=json.loads((out/'run/ready.json').read_text())['monotonic']
   for at,profile in changes:
    while time.monotonic()-start<at and proc.poll() is None:time.sleep(.02)
    if proc.poll() is not None:break
    tc(PROFILES[profile],change=True);applied.append(dict(t=time.monotonic()-start,profile=profile))
   code=proc.wait(timeout=180)
 except Exception as e:(out/'error.txt').write_text(repr(e))
 finally:
  if proc is not None and proc.poll() is None:proc.kill();proc.wait()
  try:(out/'netem.json').write_text(subprocess.check_output(['tc','-s','-j','qdisc','show','dev','lo'],text=True))
  finally:tc([])
 (out/'schedule.json').write_text(json.dumps(applied))
 results.append(dict(name=name,code=code));print(name,code,flush=True)

# Same original checkpoint twice, isolate admission window and congestion controller.
for repeat in (1,2):
 for window,cc in [(16,'reno'),(128,'reno'),(128,'cubic')]:
  run(f'loss-{repeat}-{window}-{cc}',['--payload-dir','data/real-ntc','--sequence','--duration','2','--fps','1','--max-inflight',str(window),'--datagram-size','1200','--cc',cc,'--chunk-timeout','120','--drain-timeout','140'])
# Explicit reproduction of old per-chunk timeout vs new recovery budget on burst.
for timeout in (15,120):
 run(f'burst-timeout-{timeout}',['--payload-dir','data/real-ntc','--sequence','--duration','1','--fps','1','--max-inflight','128','--datagram-size','1200','--chunk-timeout',str(timeout),'--drain-timeout','140'],'burst')
# VOD prefetch; fixed-high vs buffer-aware at repository-documented byte scales.
for repeat in (1,2):
 for policy in ('fixed-high','composite','buffer'):
  run(f'vod-{repeat}-{policy}',['--vod','--duration','8','--fps','2','--policy',policy,'--prebuffer','2','--playback-buffers','1','2','4','--max-inflight','128','--drain-timeout','140'],'good',[(6,'bad'),(12,'good')])
(root/'runs.json').write_text(json.dumps(results,indent=2))

if any(r['code'] for r in results if r['name']!='burst-timeout-15'): raise SystemExit(1)
