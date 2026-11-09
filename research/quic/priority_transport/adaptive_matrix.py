"""Isolated Linux container only: real netem, dynamic schedules and negative controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROFILES = {
 'good': ['delay','10ms','0ms','loss','random','0%','rate','24mbit'],
 'bad': ['delay','40ms','10ms','distribution','normal','loss','random','2%','rate','1500kbit'],
 'bandwidth': ['delay','10ms','rate','1500kbit'],
 'delay': ['delay','80ms','rate','12mbit'],
 'jitter': ['delay','30ms','20ms','distribution','normal','rate','12mbit'],
 'loss': ['delay','10ms','loss','random','3%','rate','12mbit'],
 'burst': ['delay','10ms','loss','gemodel','2%','25%','80%','0.1%','rate','12mbit'],
 'reorder': ['delay','30ms','10ms','reorder','20%','50%','rate','12mbit'],
 'clean': [],
}


def tc(opts, change=False):
    if change:
        subprocess.run(["tc","qdisc","change","dev","lo","root","netem","limit","1000",*opts], check=True)
        return
    subprocess.run(['tc','qdisc','del','dev','lo','root'], capture_output=True)
    if opts:
        subprocess.run(['tc','qdisc','add','dev','lo','root','netem','limit','1000',*opts], check=True)


def execute(root, name, policy, seed, schedule, duration, extra=()):
    out = root/name
    out.mkdir()
    applied=[]; stats=[]; proc=None; error=None; code=-1
    cmd=[sys.executable,'-m','research.quic.priority_transport.experiment','--cert','cert.pem','--key','key.pem',
         '--output',str(out/'run'),'--policy',policy,'--profile',name,'--seed',str(seed),'--duration',str(duration),*extra]
    try:
        tc(PROFILES[schedule[0][1]])
        with (out/'console.log').open('w') as log:
            proc=subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            ready=out/'run/ready.json'
            ready_deadline=time.monotonic()+20
            while not ready.exists() and proc.poll() is None and time.monotonic()<ready_deadline:
                time.sleep(.02)
            if not ready.exists():
                raise RuntimeError('experiment did not become ready within 20 seconds')
            begin=json.loads(ready.read_text())['monotonic']
            applied.append(dict(t=0,profile=schedule[0][1]))
            for at, profile in schedule[1:]:
                while time.monotonic()-begin < at and proc.poll() is None: time.sleep(.02)
                if proc.poll() is not None: break
                stats.append(json.loads(subprocess.check_output(['tc','-s','-j','qdisc','show','dev','lo'],text=True)))
                tc(PROFILES[profile], change=True)
                applied.append(dict(t=time.monotonic()-begin,profile=profile))
            code=proc.wait(timeout=duration+70)
    except Exception as exc:
        error=repr(exc)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill(); proc.wait()
        try:
            stats.append(json.loads(subprocess.check_output(['tc','-s','-j','qdisc','show','dev','lo'],text=True)))
        except Exception as exc:
            error=error or repr(exc); code=-1
        finally:
            tc([])
    (out/'netem.json').write_text(json.dumps(dict(schedule=applied,stats=stats,error=error,
        profiles=PROFILES,returncode=code,seed_note='seed labels repeats; kernel netem randomness not fixed'),indent=2))

    print(name,code,flush=True)
    return code


def main():
    if not Path('/.dockerenv').exists(): raise SystemExit('run only in isolated Docker container')
    p=argparse.ArgumentParser(); p.add_argument('--output',default='data/outputs/adaptive-matrix'); p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args(); root=Path(a.output); root.mkdir(parents=True,exist_ok=False)
    results={}
    for seed in range(1,a.repeats+1):
        for policy in ['fixed-low','fixed-high','queue','rtt','composite']:
            name=f'dynamic-{seed}-{policy}'
            results[name]=execute(root,name,policy,seed,[(0,'good'),(6,'bad'),(12,'good')],18)
    for profile in ['clean','bandwidth','delay','jitter','loss','burst','reorder']:
        name=f'static-{profile}'
        results[name]=execute(root,name,'fixed-high',1,[(0,profile)],8)
    results['receiver-slow']=execute(root,'receiver-slow','composite',1,[(0,'clean')],8,['--receiver-delay-ms','150'])
    results['clean-overload']=execute(root,'clean-overload','composite',1,[(0,'good')],8,['--fps','16'])
    for profile in ['good','loss','burst']:
        name=f'real-{profile}'
        results[name]=execute(root,name,'fixed-high',1,[(0,profile)],10,
            ['--payload-dir','data/real-ntc','--sequence','--fps','0.2','--deadline-ms','8000'])
    (root/'runs.json').write_text(json.dumps(results,indent=2))
    if any(results.values()): raise SystemExit(1)

if __name__=='__main__': main()
