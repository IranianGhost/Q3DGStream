"""Real Publio representations in the original CUDA viewer over live QUIC.

Every update sends one complete same-frame model plus original additions. No
optional synthetic bytes and no partially applied neural transformation.
"""
import argparse, asyncio, gzip, json, math, statistics, time
from pathlib import Path
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from .experiment import CountingLogger
from . import native_playback_run as base
from .send_scene import collect_scene
from .transport import ALPN, FileJob, send_files, send_whole_files
from .live_playback import write_snapshot

LADDER=['INT2_keep05_PQAT','INT4_keep05_PQAT','INT4_keep10_PQAT','INT4_keep20_PQAT','INT8_keep20_PQAT']
GOOD=['delay','10ms','rate','24mbit']
BAD=['delay','40ms','10ms','distribution','normal','loss','random','2%','rate','1500kbit']


def select_level(sizes,feedback,now,interval,target,previous,addition_bytes=0):
    """Pick future same-frame representation using fresh buffer/rate feedback."""
    if not feedback or not 0<=now-feedback['at']<=1:
        return 0,'stale_feedback'
    rate=feedback.get('delivery_service_Bps',feedback['committed_Bps']);buffer=feedback['buffer_seconds']
    if not math.isfinite(rate) or not math.isfinite(buffer) or rate<0 or buffer<0:
        return 0,'invalid_feedback'
    budget=rate*interval*.8-addition_bytes
    safe=max((i for i,s in enumerate(sizes) if s<=budget),default=0)
    if buffer<max(.25,target*.5):
        return max(0,min(previous-1,safe)),'buffer_low'
    if buffer>=target*.75:
        return min(previous+1,safe),'buffer_and_capacity_probe'
    return min(previous,safe),'capacity_hold'


async def stream(a,port,started,network):
    init,config,frames,has_additions=collect_scene(a.scene,a.updates)
    manifest=a.output/'manifest.json'
    write_snapshot(manifest,dict(require_additions=has_additions,updates=a.updates,
                    checkpoint_format='publio-complete-pqat',complete_updates_only=True))
    log=CountingLogger()
    tls=QuicConfiguration(is_client=True,alpn_protocols=ALPN,server_name='localhost',
              congestion_control_algorithm=a.cc,max_datagram_size=1400,quic_logger=log)
    tls.load_verify_locations(a.output/'cert.pem')
    frames_log=[];samples=[];phase_task=None;monitor_task=None;done=False
    acked_bytes=0;submitted_bytes=0;last_ack=time.monotonic();media_start=None
    receiver_work=[];last_delivery_rate=None
    def event(kind,row):
        nonlocal acked_bytes,submitted_bytes,last_ack
        if kind=='acked':
            acked_bytes+=row['size'];last_ack=time.monotonic()
            if row.get('processing_ms') is not None:receiver_work.append(row['processing_ms'])
        else:submitted_bytes+=row['size']
    try:
        async with connect('127.0.0.1',port,configuration=tls) as connection:
            async def transfer(jobs):
                if a.mode=='whole':return await send_whole_files(connection,jobs,timeout=120)
                return await send_files(connection,jobs,max_inflight=128,policy='strict',timeout=120,on_event=event)
            async def monitor():
                previous_ack=0;previous_t=time.monotonic();previous_lost=0;previous_sent=0
                while not done:
                    now=time.monotonic();r=connection._quic._loss
                    eligible=(min(a.updates,max(0,int((now-media_start)*a.fps)+1))
                              if a.paced else a.updates) if media_start is not None else 0
                    unfinished=max(0,eligible-len(frames_log))
                    viewer_feedback=json.loads((a.output/'feedback.json').read_text())
                    oldest=(max(0,now-media_start-len(frames_log)/a.fps)*1000
                            if unfinished and a.paced else None)
                    samples.append(dict(t=now-started,srtt_ms=r._rtt_smoothed*1000,
                           min_rtt_ms=r._rtt_min*1000,rtt_variance_ms=r._rtt_variance*1000,
                           cwnd_bytes=r.congestion_window,bytes_in_flight=r.bytes_in_flight,
                           quic_retained_bytes=sum(len(st.sender._buffer) for st in connection._quic._streams.values()),
                           app_unacked_bytes=submitted_bytes-acked_bytes if a.mode=='chunked' else None,
                           app_ack_goodput_Bps=(acked_bytes-previous_ack)/max(1e-6,now-previous_t) if a.mode=='chunked' else None,
                           ack_silence_ms=(now-last_ack)*1000 if a.mode=='chunked' else None,
                           playable_buffer_seconds=viewer_feedback.get('buffer_seconds'),
                           viewer_feedback_age_seconds=now-viewer_feedback['at'],
                           presented_frame=viewer_feedback.get('frame'),eligible_media_updates=eligible,
                           unfinished_due_updates=unfinished,oldest_due_update_age_ms=oldest,
                           lost_packets=log.lost-previous_lost,sent_packets=log.sent-previous_sent,
                           receiver_processing_ms=statistics.median(receiver_work[-128:]) if receiver_work else None,
                           delivery_service_Bps=last_delivery_rate))
                    previous_ack=acked_bytes;previous_t=now;previous_lost=log.lost;previous_sent=log.sent
                    await asyncio.sleep(.1)
            monitor_task=asyncio.create_task(monitor())
            rows=await transfer([FileJob(manifest,'priority_manifest.json',0),
                                 FileJob(init,'init_3dgs.ply',0),FileJob(config,'NTCs/config.json',0)])
            write_snapshot(a.output/'metadata.json',dict(finished_seconds=time.monotonic()-started,
                          chunks=len(rows),bytes=sum(r['size'] for r in rows)))
            media_start=time.monotonic()
            async def phases():
                for offset,profile,options in [(6.,'bad',BAD),(12.,'good',GOOD)]:
                    await asyncio.sleep(max(0,media_start+offset-time.monotonic()))
                    before=base.qdisc()
                    base.command('tc','qdisc','change','dev','lo','root','netem','limit','4096',*options)
                    network.append(dict(phase=profile,requested_media_wall_s=offset,
                         at=time.monotonic()-started,before=before,after=base.qdisc()))
            if a.profile=='dynamic':phase_task=asyncio.create_task(phases())
            previous=len(LADDER)-1
            for index,_,addition in frames:
                if a.paced:await asyncio.sleep(max(0,media_start+index/a.fps-time.monotonic()))
                frame_started=time.monotonic()
                feedback=json.loads((a.output/'feedback.json').read_text())
                if last_delivery_rate is not None:
                    feedback['delivery_service_Bps']=last_delivery_rate
                addition_bytes=addition.stat().st_size if addition else 0
                if a.rep_policy=='fixed':variant=a.variant;reason='fixed_representation'
                else:
                    sizes=[(a.dataset/v/f'NTC_{index:06d}.pth').stat().st_size for v in LADDER]
                    if a.rep_policy=='fixed-high':level=len(LADDER)-1;reason='fixed_high'
                    elif a.rep_policy=='fixed-low':level=0;reason='fixed_low'
                    elif index==0:level=len(LADDER)-1;reason='shared_high_initial_model'
                    else:level,reason=select_level(sizes,feedback,frame_started,1/a.fps,a.prebuffer,previous,addition_bytes)
                    previous=level;variant=LADDER[level]
                ntc=a.dataset/variant/f'NTC_{index:06d}.pth'
                jobs=[FileJob(ntc,f'NTCs/NTC_{index:06d}.pth',0)]
                if addition:jobs.append(FileJob(addition,f'additional_3dgs/{addition.name}',0))
                rows=await transfer(jobs)
                frame_done=time.monotonic()
                last_delivery_rate=(ntc.stat().st_size+addition_bytes)/max(1e-6,frame_done-frame_started)
                frames_log.append(dict(index=index,representation=variant,reason=reason,
                         checkpoint_bytes=ntc.stat().st_size,required_bytes=ntc.stat().st_size+addition_bytes,
                         optional_bytes=0,started_seconds=frame_started-started,
                         finished_seconds=frame_done-started,chunks=len(rows),
                         delivery_service_Bps=last_delivery_rate,
                         feedback_age_seconds=frame_started-feedback['at'],feedback=feedback))
                write_snapshot(a.output/'sender.json',frames_log)
            if phase_task:await phase_task
    finally:
        done=True
        if phase_task and not phase_task.done():phase_task.cancel();await asyncio.gather(phase_task,return_exceptions=True)
        if monitor_task:await asyncio.gather(monitor_task,return_exceptions=True)
        write_snapshot(a.output/'transport-samples.json',samples)
        with gzip.open(a.output/'qlog.json.gz','wt') as f:json.dump(log.to_dict(),f)
    return frames_log


def validate_schedule(a):
    if a.profile=='dynamic' and (not a.paced or not math.isfinite(a.fps) or a.fps<=0
                                or (a.updates-1)/a.fps<18 or a.prebuffer<.5):
        raise ValueError('dynamic trials require paced >=18s media and >=0.5s prebuffer')


async def run(a):
    validate_schedule(a)
    if a.profile=='dynamic':base.PROFILES['dynamic']=GOOD
    try:
        await base.run(a,stream_sender=stream)
    finally:
        report_path=a.output/'REPORT.json'
        if report_path.exists():
            report=json.loads(report_path.read_text())
            report.update(dataset=str(a.dataset),representation=(a.variant if a.rep_policy=='fixed' else ('mixed per update' if a.rep_policy=='adaptive' else (LADDER[-1] if a.rep_policy=='fixed-high' else LADDER[0]))),representation_policy=a.rep_policy,
                          paced=a.paced,network_schedule='6s bad / 12s recovered after metadata, independent of update commits' if a.profile=='dynamic' else None,
                          controller_rate='last active transfer payload bytes / submit-to-commit ACK time; excludes source pacing idle')
            write_snapshot(report_path,report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=lambda s:Path(s).resolve(),required=True)
    p.add_argument('--dataset',type=lambda s:Path(s).resolve(),required=True)
    p.add_argument('--output',type=lambda s:Path(s).resolve(),required=True)
    p.add_argument('--variant',choices=['FP32','FP16',*LADDER],default='INT4_keep05_PQAT')
    p.add_argument('--rep-policy',choices=['fixed','fixed-high','fixed-low','adaptive'],default='fixed')
    p.add_argument('--desktop-root',default='/local/users/abeigi/q3dg-desktop')
    p.add_argument('--display',type=int,default=97)
    p.add_argument('--updates',type=int,default=8);p.add_argument('--fps',type=float,default=30.)
    p.add_argument('--prebuffer',type=float,default=.1);p.add_argument('--timeout',type=float,default=170.)
    p.add_argument('--profile',choices=base.PROFILES,default='clean')
    p.add_argument('--cc',choices=['reno','cubic'],default='cubic')
    p.add_argument('--mode',choices=['chunked','whole'],default='chunked')
    p.add_argument('--paced',action='store_true')
    a=p.parse_args();a.policy='none';a.optional_bytes=0
    asyncio.run(run(a))
