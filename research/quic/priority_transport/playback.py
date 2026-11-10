"""Deterministic receiver timeline. Complete sequential jobs represent media intervals."""
class Playback:
    def __init__(self, count, fps, prebuffer):
        if count<1 or fps<=0 or prebuffer<0: raise ValueError('invalid timeline')
        self.count,self.fps,self.target=count,fps,prebuffer
        self.received=set(); self.contiguous=0; self.position=0.; self.time=0.
        self.started=None; self.waiting=True; self.stalls=0; self.stall_seconds=0.

    def advance(self, now):
        if now<self.time: raise ValueError('time reversed')
        dt=now-self.time
        if self.started is not None and self.position<self.count/self.fps:
            if self.waiting: self.stall_seconds+=dt
            else:
                available=max(0.,self.contiguous/self.fps-self.position)
                consume=min(dt,available); self.position+=consume
                if consume<dt and self.position<self.count/self.fps-1e-9:
                    self.waiting=True; self.stalls+=1; self.stall_seconds+=dt-consume
        self.time=now

    def commit(self, index, now):
        self.advance(now); self.received.add(index)
        while self.contiguous in self.received: self.contiguous+=1
        buffer=max(0.,self.contiguous/self.fps-self.position)
        required=min(self.target,(self.count/self.fps-self.position))
        if self.waiting and buffer>0 and buffer+1e-9>=required:
            self.waiting=False
            if self.started is None: self.started=now

    def snapshot(self,now):
        self.advance(now)
        return dict(buffer_seconds=max(0.,self.contiguous/self.fps-self.position),
            startup_seconds=self.started,stall_count=self.stalls,stall_seconds=self.stall_seconds,
            played_seconds=self.position,finished=self.position>=self.count/self.fps-1e-9)


def evaluate(frames,fps,prebuffer,end,planned_count=None):
    p=Playback(planned_count or len(frames),fps,prebuffer)
    for f in sorted((f for f in frames if 'received_s' in f), key=lambda f:f['received_s']):
        p.commit(f['id'],f['received_s'])
    complete=len(p.received)==p.count
    result=p.snapshot(max(end,p.time)+(p.count/fps+prebuffer if complete else 0))
    result['unplayed_seconds']=max(0,p.count/fps-p.position)
    result['prebuffer_seconds']=prebuffer
    result['late_to_initial_schedule']=sum('received_s' not in f or result['startup_seconds'] is None or f['received_s']>result['startup_seconds']+f['id']/fps+1e-9 for f in frames)
    result['stall_measurement_censored']=not result['finished']
    return result
