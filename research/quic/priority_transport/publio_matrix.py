"""Matched real-checkpoint transfer trials; all shaping is in a private netns.

Static trials compare byte-identical inputs, not unavailable legacy code.
Priority stress assigns frame urgency experimentally: the dataset supplies no
P0-P3 saliency groups. Sender qlog records endpoint STREAM-frame construction.
"""
import argparse, asyncio, gzip, json, os, statistics, subprocess, tempfile, time
from pathlib import Path
from aioquic.asyncio import connect, serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.logger import QuicLogger
from .profiles import PROFILES
from .transport import ALPN, Receiver, FileJob, file_hash, send_files, send_whole_files


def dump(path,value):
    path.write_text(json.dumps(value,indent=2))


def stats():
    return json.loads(subprocess.check_output(['tc','-j','-s','qdisc','show','dev','lo']))


def shape(profile):
    subprocess.run(['tc','qdisc','del','dev','lo','root'],capture_output=True)
    if PROFILES[profile]:
        subprocess.run(['tc','qdisc','add','dev','lo','root','netem','limit','1000',*PROFILES[profile]],check=True)


async def trial(a,name,jobs,mode,profile,repeat,variant,policy='strict',window=128):
    out=a.output/name
    if out.exists():raise ValueError('trial output already exists')
    out.mkdir()
    shape(profile)
    logger=QuicLogger(); samples=[]; rows=[]; error=None; started=None; commits=[]
    with tempfile.TemporaryDirectory(dir=a.output) as temp:
        cache=Path(temp)
        rx=Receiver(cache,on_file=lambda path:commits.append({'path':path,'at':time.monotonic()}))
        server_config=QuicConfiguration(is_client=False,alpn_protocols=ALPN)
        server_config.load_cert_chain(a.cert,a.key)
        server=await serve('127.0.0.1',0,configuration=server_config,stream_handler=rx.handler)
        conf=QuicConfiguration(is_client=True,alpn_protocols=ALPN,server_name='localhost',
                               congestion_control_algorithm='cubic',max_datagram_size=1400,quic_logger=logger)
        conf.load_verify_locations(a.cert)
        acked=0; submitted=0;done=False
        def event(kind,row):
            nonlocal acked,submitted
            if kind=='acked':acked+=row['size']
            else:submitted+=row['size']
        try:
            async with connect('127.0.0.1',server._transport.get_extra_info('sockname')[1],configuration=conf) as connection:
                started=time.monotonic()
                async def monitor():
                    while not done:
                        r=connection._quic._loss
                        samples.append({'t':time.monotonic()-started,'srtt_ms':r._rtt_smoothed*1000,
                         'cwnd_bytes':r.congestion_window,'bytes_in_flight':r.bytes_in_flight,
                         'quic_retained_bytes':sum(len(s.sender._buffer) for s in connection._quic._streams.values()),
                         'app_unacked_bytes':submitted-acked if mode=='chunked' else None,
                         'acked_bytes':acked if mode=='chunked' else None})
                        await asyncio.sleep(.1)
                monitor_task=asyncio.create_task(monitor())
                try:
                    if mode=='whole':rows=await asyncio.wait_for(send_whole_files(connection,jobs,timeout=120),180)
                    else:rows=await asyncio.wait_for(send_files(connection,jobs,policy=policy,max_inflight=window,timeout=120,on_event=event),180)
                    elapsed=time.monotonic()-started
                finally:
                    done=True;await monitor_task
                for job in jobs:
                    if file_hash(cache/job.remote)!=file_hash(job.path):raise AssertionError('wire hash mismatch')
                if rx.errors or len(commits)!=len(jobs):raise AssertionError('missing or rejected commits')
        except Exception as exc:
            elapsed=time.monotonic()-started if started else None
            error=repr(exc)
        finally:
            server.close();await rx.close()
        qlog=logger.to_dict()
        qevents=[e for t in qlog['traces'] for e in t['events']]
        first=[];seen=set();finished=set()
        for e in qevents:
            if e['name']!='transport:packet_sent':continue
            for f in e['data'].get('frames',[]):
                if f.get('frame_type')=='stream':
                    sid=f['stream_id']
                    if sid not in seen:first.append(sid);seen.add(sid)
                    if f.get('fin'):finished.add(sid)
        priorities={r['stream_id']:r['priority'] for r in rows}
        p0ranks=[i+1 for i,sid in enumerate(first) if priorities.get(sid)==0]
        source={job.remote:job for job in jobs}
        latencies={}
        for c in commits:
            job=source[c['path']]
            latency=(c['at']-started)*1000-job.release_ms
            latencies.setdefault(str(job.priority),[]).append(latency)
        if not error and seen!=set(priorities):error='qlog first-STREAM coverage mismatch'
        if not error and policy=='strict' and mode=='chunked':
            pending={(r['path'],r['offset']):r for r in rows}
            for r in sorted(rows,key=lambda x:x['submitted_ms']):
                ready=[v for v in pending.values() if v['release_ms']<=r['selected_ms']]
                if ready and r['priority']!=min(v['priority'] for v in ready):
                    error='strict admission audit failed';break
                pending.pop((r['path'],r['offset']))
        report={'name':name,'variant':variant,'profile':profile,'repeat':repeat,'mode':mode,'policy':policy,
                'files':len(jobs),'committed_files':len(commits),'bytes':sum(j.path.stat().st_size for j in jobs),
                'elapsed_s':elapsed,'checkpoint_commit_s':max((c['at']-started for c in commits),default=None),
                'valid':error is None,'error':error,'chunks':len(rows),'cc':'cubic','datagram_size':1400,
                'max_inflight':window if mode=='chunked' else None,
                'p0_first_stream_rank_median':statistics.median(p0ranks) if p0ranks else None,
                'file_latency_by_priority_ms':{p:{'median':statistics.median(v),'p95':sorted(v)[int(.95*(len(v)-1))]} for p,v in latencies.items()},
                'quic_declared_loss':sum(e['name']=='recovery:packet_lost' for e in qevents),
                'netem':stats(),'receiver_errors':rx.errors,'qlog_streams_verified':len(seen),
                'scope':'steady-state full checkpoint after handshake; no scene or rendering' if len(jobs)==1 else 'constructed urgent/future real-checkpoint workload, not supplied saliency',
                'sources':[{'path':str(j.path),'remote':j.remote,'sha256':file_hash(j.path),'priority':j.priority,'release_ms':j.release_ms} for j in jobs]}
        dump(out/'summary.json',report);dump(out/'events.json',rows);dump(out/'samples.json',samples)
        dump(out/'receiver-events.json',rx.events)
        with gzip.open(out/'qlog.json.gz','wt') as f:json.dump(qlog,f)
        print(name,'PASS' if report['valid'] else 'FAIL',round(elapsed or 0,3),flush=True)
        return report


async def main(a):
    if os.environ.get('Q3DG_PARENT_NETNS')==os.readlink('/proc/self/ns/net') or not os.environ.get('Q3DG_PARENT_NETNS'):
        raise RuntimeError('private netns launcher required')
    links=json.loads(subprocess.check_output(['ip','-j','link','show']))
    if [x['ifname'] for x in links]!=['lo']:raise RuntimeError('refusing to shape a nonisolated network')
    subprocess.run(['ip','link','set','lo','up'],check=True)
    a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    variants=['FP32','FP16','INT8_keep20_PQAT','INT4_keep20_PQAT','INT4_keep10_PQAT','INT4_keep05_PQAT','INT2_keep05_PQAT']
    try:
        for profile in (a.profiles or PROFILES):
            for repeat in range(1,a.repeats+1):
                frame=[0,114,298][(repeat-1)%3]
                if a.scope=='static':
                    for variant in variants:
                        jobs=[FileJob(a.dataset/variant/f'NTC_{frame:06d}.pth',f'NTCs/NTC_{frame:06d}.pth',0)]
                        for mode in (['whole','chunked'] if repeat%2 else ['chunked','whole']):
                            name=f'{profile}-{repeat}-{variant}-{mode}'
                            results.append(await trial(a,name,jobs,mode,profile,repeat,variant))
                else:
                    # Large future-frame backlog: 288 chunks of P3, before 72 chunks
                    # of P0 appear. The 128-slot transport ceiling stays unchanged.
                    jobs=[]
                    for p,indices in [(3,range(100,132)),(0,range(0,8)),(1,range(8,16)),(2,range(16,24))]:
                        for i in indices:
                            jobs.append(FileJob(a.dataset/'INT4_keep05_PQAT'/f'NTC_{i:06d}.pth',f'NTCs/NTC_{i:06d}.pth',p,0 if p==3 else 150))
                    policies=['fifo','strict','weighted'] if repeat%2 else ['weighted','strict','fifo']
                    if a.policies:policies=[p for p in policies if p in a.policies]
                    for policy in policies:
                        name=f'{profile}-{repeat}-{policy}'
                        results.append(await trial(a,name,jobs,'chunked',profile,repeat,'INT4_keep05_PQAT',policy))
                dump(a.output/'runs.json',results)
    finally:subprocess.run(['tc','qdisc','del','dev','lo','root'],capture_output=True)
    if any(not r['valid'] for r in results):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',choices=['static','priority'],required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cert',required=True);p.add_argument('--key',required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--profiles',nargs='+',choices=list(PROFILES))
    p.add_argument('--policies',nargs='+',choices=['fifo','strict','weighted'])
    asyncio.run(main(p.parse_args()))
