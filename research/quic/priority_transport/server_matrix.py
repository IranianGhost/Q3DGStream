"""Quinn pilot matrix on private loopback. Real saliency orders are required by default.

--address-control permits an explicitly labelled transport-only control using
real checkpoint bytes in ascending block-address order. It is NOT saliency.
"""
import argparse,csv,hashlib,json,os,subprocess,threading,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from evaluation_support import parse_byte_counters

PROFILES={
 'clean':[],
 'bandwidth':['rate','5mbit'],
 'delay':['delay','40ms','rate','5mbit'],
 'jitter':['delay','40ms','10ms','distribution','normal','rate','5mbit'],
 'loss':['delay','20ms','loss','random','2%','rate','5mbit'],
 'burst':['delay','20ms','loss','gemodel','1%','25%','80%','0.1%','rate','5mbit'],
 'dynamic':['delay','10ms','rate','20mbit'],
}
METHODS=['original','sequential','concurrent','p0_first','amirali_fifo','amirali_strict','amirali_weighted']

def validate_isolation(parent,current,interfaces):
    if not parent or parent==current or interfaces!=['lo']:
        raise RuntimeError('requires a fresh private user/network namespace containing only lo')

def require_orders(folder,frames):
    missing=[str(folder/f'NTC_{i:06d}.npz') for i in range(frames) if not (folder/f'NTC_{i:06d}.npz').is_file()]
    if missing:raise FileNotFoundError(f'missing {len(missing)} real priority files; first: {missing[0]}')


def address_control_orders(wire,folder,frames):
    import torch
    folder.mkdir(exist_ok=False)
    for i in range(frames):
        source=wire/f'NTC_{i:06d}.pth'
        state=torch.load(source,map_location='cpu',weights_only=True)
        ids=np.sort(state['model.params_sparse_int4_block_indices'].numpy())
        # Equal zero scores explicitly express that no saliency measurement exists.
        np.savez_compressed(folder/f'NTC_{i:06d}.npz',ordered_block_ids=ids,
             priority_scores=np.zeros(len(ids),np.float32),source_wire=str(source.resolve()),ntc_index=np.int32(i))
    (folder/'PROVENANCE.json').write_text(json.dumps(dict(method='ascending block address; all priority scores zero; NOT visual saliency',source=str(wire),frames=frames),indent=2))


def shape(tokens):
    subprocess.run(['tc','qdisc','del','dev','lo','root'],capture_output=True)
    if tokens:subprocess.run(['tc','qdisc','add','dev','lo','root','netem','limit','1000',*tokens],check=True)


def change_shape(tokens):
    # Preserve the active queue and its cumulative counters during a trial.
    subprocess.run(['tc','qdisc','change','dev','lo','root','netem','limit','1000',*tokens],check=True)


def create_output(output):
    output.mkdir(parents=True,exist_ok=False)


def percentile(values,q):
    return float(np.quantile(values,q)) if values else None


def main(a):
    links=json.loads(subprocess.check_output(['ip','-j','link','show']))
    validate_isolation(os.environ.get('Q3DG_PARENT_NETNS'),os.readlink('/proc/self/ns/net'),[x['ifname'] for x in links])
    subprocess.run(['ip','link','set','lo','up'],check=True)
    create_output(a.output)
    frames=max(a.frames,60 if 'dynamic' in a.profiles else 0)
    if a.address_control:
        orders=a.output/'address-control-orders'
        if not orders.exists():address_control_orders(a.wire,orders,frames)
        provenance='REAL CHECKPOINT BYTES; ASCENDING ADDRESS CONTROL; NOT PUBLIO SALIENCY'
    else:orders=a.orders;provenance='supplied saliency order files'
    require_orders(orders,frames)
    (a.output/'METHOD.json').write_text(json.dumps(dict(upstream='db1d0dc5e6c785cef428013e827abaccca304d0d',wire=str(a.wire),orders=str(orders),provenance=provenance,
       inputs=[dict(frame=i,wire_sha256=hashlib.sha256((a.wire/f'NTC_{i:06d}.pth').read_bytes()).hexdigest(),order_sha256=hashlib.sha256((orders/f'NTC_{i:06d}.npz').read_bytes()).hexdigest()) for i in range(frames)],
       profiles=PROFILES,frames=a.frames,dynamic_frames=60,fps=30,playout_ms=500,max_inflight=128,repeats=a.repeats,
       note='netem applies both directions on isolated loopback; transport deadlines only; fallback compute costs 17ms, not measured GPU; UDP snapshots before connection close'),indent=2))
    import run_deadline as runner
    rows=[]
    try:
      for rep in range(1,a.repeats+1):
       for profile in a.profiles:
        for selection in (['adaptive'] if profile=='dynamic' else a.selections):
         for priority in a.priorities:
          methods=a.methods if rep%2 else list(reversed(a.methods))
          for method in methods:
            name=f'{profile}-{selection}-{priority}-{method}-{rep}'
            out=a.output/name;summary_file=out/'pilot-summary.json'
            if out.exists():raise ValueError('incomplete output exists; inspect or choose new output: '+str(out))
            shape(PROFILES[profile]);epoch=threading.Event();stop=threading.Event();network_events=[];phase_errors=[]
            raw=[];lock=threading.Lock()
            def pump(who,process,messages):
                for line in process.stdout:
                    line=line.rstrip('\n')
                    with lock:raw.append(f'[{who}] {line}')
                    if who=='SENDER' and line.startswith('SENDER_EPOCH '):epoch.set()
                    messages.put((who,line))
                messages.put((who,'__EXIT__'))
            runner.pump=pump
            def phases():
                try:
                    if not epoch.wait(20):return
                    start=time.monotonic()
                    for at,tokens in [(0.5,['delay','40ms','10ms','distribution','normal','loss','random','2%','rate','1mbit']),
                                      (1.5,PROFILES['dynamic'])]:
                        if stop.wait(max(0,start+at-time.monotonic())):return
                        change_shape(tokens);network_events.append(dict(at=time.monotonic()-start,tokens=tokens,qdisc=json.loads(subprocess.check_output(['tc','-j','-s','qdisc','show','dev','lo']))))
                except Exception as exc:phase_errors.append(repr(exc))
            thread=threading.Thread(target=phases,daemon=True) if profile=='dynamic' else None
            if thread:thread.start()
            args=SimpleNamespace(frames=60 if profile=='dynamic' else a.frames,start_frame=0,timeout=120.,live=False,
                metrics_json=None,progressive=False,rate_profile='',wire_dir=str(a.wire),priority_dir=str(orders),out=str(out),
                prepare_only=False,netns=False,rate_mbps=30,initial_mbps=30 if profile=='clean' else (20 if profile=='dynamic' else 5),
                delay_ms=0,loss_pct=0,port=55443,fps=30.,feedback_ms=100,playout_ms=500.,scheduling_mode=method,
                quinn_priority=priority,tier_selection=selection,fixed_tiers=4,max_inflight=128,tier_delays_ms='0,0,0,0',
                delivery_timeout_ms=60000,utilization=.92,inference_ms=8.,render_ms=8.,decode_ms=1.)
            started=time.monotonic();row=dict(case=name,profile=profile,selection=selection,priority=priority,method=method,repeat=rep,frames=args.frames,provenance=provenance)
            try:
                runner.run(args)
                if phase_errors:raise RuntimeError(str(phase_errors))
                if profile=='dynamic' and len(network_events)!=2:raise RuntimeError('both dynamic transitions must occur')
                counters=parse_byte_counters(raw)
                summary=json.loads((out/'priority/summary.json').read_text())
                records=list(csv.DictReader((out/'priority/per_frame_deadline.csv').open()))
                unique=summary['payload_bytes_verified']
                if counters['sender']['unique_application_payload_bytes']!=unique:raise ValueError('counter and verified-file byte sum mismatch')
                row.update(complete=True,payload_sent_bytes=unique,payload_received_bytes=counters['receiver']['unique_application_payload_bytes'],
                    sender_udp_tx_bytes=counters['sender']['udp_tx_bytes'],sender_udp_rx_bytes=counters['sender']['udp_rx_bytes'],
                    receiver_udp_tx_bytes=counters['receiver']['udp_tx_bytes'],receiver_udp_rx_bytes=counters['receiver']['udp_rx_bytes'],
                    declared_loss_packets=counters['sender']['lost_packets'],deadline_misses=summary['deadline_missed_transport'],
                    p0_deadline_misses=summary['p0_deadline_missed_transport'],tiers_mean=sum(float(r['selected_tiers']) for r in records)/len(records),
                    p0_latency_p95_ms=percentile([float(r['p0_complete_ms_connection'])-float(r['sender_decision_ms'])-float(r['sender_to_receiver_clock_offset_ms']) for r in records],.95),
                    rtt_median_ms=percentile([float(r['rtt_ms']) for r in records],.5),
                    backlog_max_ms=max(float(r['backlog_ms']) for r in records),
                    p0_first_fraction=sum(int(r['p0_net_ready_rank'])==1 for r in records)/len(records))
                (out/'byte-counters.json').write_text(json.dumps(counters,indent=2))
            except Exception as exc:
                row.update(complete=False,error=repr(exc));(out/'failure.txt').write_text(traceback.format_exc())
            finally:
                stop.set()
                if thread:thread.join(timeout=2)
                row['elapsed_s']=time.monotonic()-started
                (out/'raw-process.log').write_text('\n'.join(raw)+'\n')
                (out/'qdisc.json').write_bytes(subprocess.check_output(['tc','-j','-s','qdisc','show','dev','lo']))
                (out/'phase-events.json').write_text(json.dumps(network_events,indent=2))
            summary_file.write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
            (a.output/'runs.json').write_text(json.dumps(rows,indent=2)+'\n')
            print('PILOT',name,row['complete'],round(row['elapsed_s'],2),flush=True)
    finally:shape([])
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wire',type=lambda x:Path(x).resolve(),required=True)
    p.add_argument('--orders',type=lambda x:Path(x).resolve())
    p.add_argument('--address-control',action='store_true',help='explicitly labelled control only, not source saliency')
    p.add_argument('--output',type=lambda x:Path(x).resolve(),required=True)
    p.add_argument('--frames',type=int,default=12);p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--priorities',nargs='+',choices=['equal','tiered'],default=['equal','tiered'])
    p.add_argument('--selections',nargs='+',choices=['full','adaptive'],default=['full','adaptive'])
    p.add_argument('--profiles',nargs='+',choices=list(PROFILES),default=list(PROFILES))
    a=p.parse_args()
    if (not a.orders and not a.address_control) or (a.orders and a.address_control):p.error('choose real --orders or explicit --address-control')
    if a.frames<1 or a.frames>299 or a.repeats<1:p.error('invalid frames/repeats')
    main(a)
