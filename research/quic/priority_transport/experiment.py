"""Periodic offered-load experiment over real QUIC; localhost clocks only.

Run inside isolated netem container for impairment experiments. --payload-dir
uses WHOLE files sorted by size as a size ladder; default payloads are synthetic
and are never advertised as valid lower-quality NTC models.
"""
import argparse
import asyncio
from collections import deque
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time

from aioquic.asyncio import connect, serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.logger import QuicLogger
from .adaptive import Controller
from .playback import Playback, evaluate
from .transport import ALPN, Receiver, FileJob, send_files, file_hash


class CountingLogger(QuicLogger):
    """Count recovery events at emission; retain standard qlog for audit."""
    def __init__(self):
        super().__init__()
        self.lost = 0
        self.sent = 0

    def start_trace(self, *args, **kwargs):
        trace = super().start_trace(*args, **kwargs)
        original = trace.log_event
        def record(*, category, event, data):
            if category == "recovery" and event == "packet_lost":
                self.lost += 1
            if category == "transport" and event == "packet_sent":
                self.sent += 1
            original(category=category, event=event, data=data)
        trace.log_event = record
        return trace


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs)-1, int((len(xs)-1)*p))] if xs else None


async def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('output must be empty')
    traces, frames, events = [], [], []
    started = time.monotonic()
    queue = deque()
    current = None
    produced_done = False
    worker_done = False
    acked_bytes = submitted_bytes = 0
    last_ack = started
    failure = None
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if args.payload_dir:
            sources = sorted((p for p in Path(args.payload_dir).iterdir() if p.is_file()), key=lambda p:p.stat().st_size)
            if not sources:
                raise ValueError('payload directory empty')
            kind = 'whole supplied files; size levels do not imply visual quality'
        else:
            sources = []
            rng = random.Random(20261109)
            for size in args.sizes:
                p = tmp / f'{size}.bin'; p.write_bytes(rng.randbytes(size)); sources.append(p)
            kind = 'synthetic byte load surrogate; not valid NTC'
        sizes = [p.stat().st_size for p in sources]
        controller = Controller('composite' if args.policy=='buffer' else args.policy, sizes)
        playback=Playback(round(args.duration*args.fps),args.fps,args.prebuffer)
        manifest = [{'name':p.name, 'bytes':p.stat().st_size, 'sha256':file_hash(p)} for p in sources]
        commits={}
        loop=asyncio.get_running_loop()
        def register(path):
            index=int(Path(path).stem); now=time.monotonic()-started
            commits[index]=now; playback.commit(index,now)
        rx = Receiver(tmp / 'rx',on_file=lambda path:loop.call_soon_threadsafe(register,path), processing_delay=args.receiver_delay_ms / 1000)
        server_conf = QuicConfiguration(is_client=False, alpn_protocols=ALPN)
        server_conf.load_cert_chain(args.cert, args.key)
        server = await serve('127.0.0.1', 0, configuration=server_conf, stream_handler=rx.handler)
        logger = CountingLogger()
        conf = QuicConfiguration(is_client=True, alpn_protocols=ALPN, server_name='localhost', quic_logger=logger, congestion_control_algorithm=args.cc,max_datagram_size=args.datagram_size)
        conf.load_verify_locations(args.cert)
        def stamp(): return time.monotonic() - started
        def on_event(kind, row):
            nonlocal acked_bytes, submitted_bytes, last_ack
            row = dict(row, event=kind, t=stamp())
            events.append(row)
            if kind == 'acked':
                acked_bytes += row['size']; last_ack = time.monotonic()
            else:
                submitted_bytes += row['size']

        async def producer():
            nonlocal produced_done
            for i in range(round(args.duration * args.fps)):
                target = started if args.vod else started + i / args.fps
                await asyncio.sleep(max(0, target - time.monotonic()))
                level = i % len(sources) if args.sequence else controller.level
                f = dict(id=i, release_s=stamp(), nominal_release_s=i/args.fps,
                         level=level, bytes=sizes[level], status='queued', source=sources[level].name)
                frames.append(f)
                while args.vod and len(queue)>=args.queue_limit:
                    await asyncio.sleep(.005)
                if len(queue) >= args.queue_limit:
                    f['status'] = 'dropped_before_admission'
                else:
                    queue.append(f)
            produced_done = True

        async def worker(connection):
            nonlocal current, worker_done
            try:
                while not produced_done or queue:
                    if not queue:
                        await asyncio.sleep(.005); continue
                    current = queue.popleft()
                    if args.vod and not args.sequence and not args.policy.startswith('fixed'):
                        current.update(level=controller.level,bytes=sizes[controller.level],source=sources[controller.level].name)
                    current['started_s'] = stamp()
                    current['status'] = 'sending'
                    remote = f"frames/{current['id']:06d}.bin"
                    rows = await send_files(connection, [FileJob(sources[current['level']], remote, 0)],
                                            timeout=args.chunk_timeout, max_inflight=args.max_inflight, on_event=on_event)
                    current['completed_s'] = stamp()
                    current['ack_latency_ms'] = (current['completed_s'] - current['release_s'])*1000
                    file_acks = [e for e in events if e['event']=='acked' and e['path']==remote]
                    current['received_s'] = commits[current['id']]
                    current['latency_ms'] = (current['received_s'] - current['release_s'])*1000
                    current['deadline_missed'] = current['latency_ms'] > args.deadline_ms
                    current['status'] = 'complete'
                    # Receiver already verifies whole-file SHA; independently verify committed bytes.
                    assert file_hash(tmp / 'rx' / remote) == manifest[current['level']]['sha256']
                    current = None
            finally:
                worker_done = True

        async def monitor(connection):
            previous_ack = 0
            previous_t = stamp()
            previous_q = 0
            previous_lost = previous_sent = 0
            event_cursor = 0
            while not worker_done:
                await asyncio.sleep(.25)
                now = stamp(); dt = max(.001, now - previous_t)
                items = list(queue) + ([current] if current else [])
                remaining = sum(f['bytes'] for f in items)
                # Outstanding offered bytes count includes unsent chunks and unacknowledged chunks.
                if current:
                    remaining -= sum(e['size'] for e in events if e['event']=='acked' and e['path']==f"frames/{current['id']:06d}.bin")
                recovery = connection._quic._loss  # pinned aioquic 1.7.2 diagnostic adapter
                streams = connection._quic._streams.values()
                new_acks = [e for e in events[event_cursor:] if e['event']=='acked']
                event_cursor = len(events)
                trace = dict(t=now, queue_bytes=max(0, remaining), queue_frames=len(queue),
                    age_ms=max((now-f['release_s'])*1000 for f in items) if items else 0,
                    queue_growth_bps=(remaining-previous_q)/dt,
                    goodput_bps=(acked_bytes-previous_ack)/dt,
                    app_unacked_bytes=submitted_bytes-acked_bytes,
                    ack_silence_ms=(time.monotonic()-last_ack)*1000,
                    srtt_ms=recovery._rtt_smoothed*1000,
                    min_rtt_ms=recovery._rtt_min*1000,
                    rtt_variance_ms=recovery._rtt_variance*1000,
                    bytes_in_flight=recovery.bytes_in_flight,
                    congestion_window=recovery.congestion_window,
                    quic_retained_bytes=sum(len(s.sender._buffer) for s in streams),
                    level=controller.level, producer_active=not produced_done,
                    lost_packets=logger.lost-previous_lost, sent_packets=logger.sent-previous_sent,
                    receiver_processing_ms=percentile([e['processing_ms'] for e in new_acks if e.get('processing_ms') is not None], .5),
                    app_ack_ms=percentile([e['acked_ms']-e['submitted_ms'] for e in new_acks], .5))
                trace.update(playback.snapshot(now))
                if args.policy=='buffer':
                    # Buffer covers media time; delivery rate limits the next job size.
                    rate=trace['goodput_bps']
                    budget=rate/args.fps*.75
                    safe=max((i for i,size in enumerate(sizes) if size<=budget),default=0)
                    if trace['buffer_seconds'] < max(.5,args.prebuffer*.5):
                        controller.level=max(0,min(controller.level-1,safe))
                        controller.healthy=0;controller.reason='buffer_low'
                    elif trace['buffer_seconds']>=args.prebuffer and rate>0:
                        controller.healthy+=1
                        if controller.healthy>=8:
                            controller.level=min(controller.level+1,safe);controller.healthy=0
                        controller.reason='buffer_capacity_probe'
                    trace['next_level']=controller.level
                else:
                    trace['next_level'] = controller.update(trace)
                trace['reason'] = controller.reason
                traces.append(trace)
                previous_ack, previous_t, previous_q = acked_bytes, now, remaining
                previous_lost, previous_sent = logger.lost, logger.sent
        try:
            async with connect('127.0.0.1', server._transport.get_extra_info('sockname')[1], configuration=conf) as connection:
                started = time.monotonic(); last_ack = started
                (out / "ready.tmp").write_text(json.dumps({"monotonic": started}))
                (out / "ready.tmp").replace(out / "ready.json")
                tasks = [asyncio.create_task(producer()), asyncio.create_task(worker(connection)), asyncio.create_task(monitor(connection))]
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks), args.duration + args.drain_timeout)
                finally:
                    for t in tasks: t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as exc:
            failure = repr(exc)
        finally:
            elapsed = stamp()
            server.close(); await rx.close()
        qlog = logger.to_dict()
        # Aggregate packet loss from the protocol's own recovery events, not netem configuration.
        qevents = [e for tr in qlog['traces'] for e in tr['events']]
        lost = sum(e['name']=='recovery:packet_lost' for e in qevents)
        sent = sum(e['name']=='transport:packet_sent' for e in qevents)
        complete = [f for f in frames if f['status']=='complete']
        ontime = [f for f in complete if not f['deadline_missed']]
        playback_results=[evaluate(frames,args.fps,b,elapsed,round(args.duration*args.fps)) for b in args.playback_buffers]
        summary = dict(playback=playback_results, max_inflight=args.max_inflight, datagram_size=args.datagram_size, cc=args.cc, chunk_timeout=args.chunk_timeout, prebuffer=args.prebuffer, vod=args.vod, policy=args.policy, profile=args.profile, seed=args.seed, payload_kind=kind, sequence=args.sequence,
            source_manifest=manifest, fps=args.fps, duration_s=args.duration, elapsed_s=elapsed,
            deadline_ms=args.deadline_ms, queue_limit=args.queue_limit,
            generated=len(frames), completed=len(complete), ontime=len(ontime),
            ontime_bytes=sum(f['bytes'] for f in ontime),
            dropped=sum(f['status']=='dropped_before_admission' for f in frames),
            incomplete=sum(f['status'] in ('queued','sending') for f in frames),
            p95_latency_ms=percentile([f['latency_ms'] for f in complete],.95),
            max_queue_bytes=max((s['queue_bytes'] for s in traces), default=0),
            switches=sum(s['level']!=s['next_level'] for s in traces),
            transmitted_level_switches=sum(a['level']!=b['level'] for a,b in zip(complete,complete[1:])),
            packets_sent=sent, packets_declared_lost=lost,
            valid=failure is None and not rx.errors, failure=failure, receiver_errors=rx.errors,
            receiver_delay_ms=args.receiver_delay_ms)
        for name, obj in [('summary',summary),('samples',traces),('frames',frames),('events',events),('qlog',qlog)]:
            (out / f'{name}.json').write_text(json.dumps(obj, indent=2))
        print(json.dumps(summary), flush=True)
        return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True); p.add_argument('--cert', required=True); p.add_argument('--key', required=True)
    p.add_argument('--policy', choices=['fixed-low','fixed-high','queue','rtt','composite','buffer'], default='composite')
    p.add_argument('--profile', default='clean'); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--duration', type=float, default=18); p.add_argument('--fps', type=float, default=4)
    p.add_argument('--deadline-ms', type=float, default=1000); p.add_argument('--queue-limit', type=int, default=8)
    p.add_argument('--drain-timeout', type=float, default=60); p.add_argument('--receiver-delay-ms', type=float, default=0)
    p.add_argument('--datagram-size',type=int,default=1400)
    p.add_argument('--max-inflight',type=int,default=128);p.add_argument('--chunk-timeout',type=float,default=120)
    p.add_argument('--cc',choices=['reno','cubic'],default='cubic')
    p.add_argument('--prebuffer',type=float,default=2);p.add_argument('--playback-buffers',type=float,nargs='+',default=[1,2,4])
    p.add_argument('--sizes',type=int,nargs='+',default=[62296,66741,7968015])
    p.add_argument('--vod',action='store_true',help='prefetch finite video as fast as queue capacity permits')
    p.add_argument('--payload-dir'); p.add_argument('--sequence', action='store_true', help='send source files cyclically, independently of adaptive size policy')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.duration <= 0 or args.fps <= 0 or args.queue_limit < 1 or args.receiver_delay_ms < 0:
        raise SystemExit('invalid workload parameters')
    result = asyncio.run(run(args))
    if not result['valid']: raise SystemExit(1)
