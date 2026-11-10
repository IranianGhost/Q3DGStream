from research.quic.priority_transport.playback import Playback,evaluate

def test_start_requires_contiguous_media_and_stall_is_counted_once():
 p=Playback(4,1,2)
 p.commit(1,.1);assert p.started is None
 p.commit(0,.2);assert p.started==.2
 p.advance(3.2);assert p.stalls==1
 p.commit(2,4);assert p.waiting
 p.commit(3,5);assert not p.waiting
 assert p.snapshot(7)['finished']
 assert abs(p.stall_seconds-2.8)<1e-8

def test_complete_prefetch_has_no_stalls():
 frames=[dict(id=i,received_s=.1*i) for i in range(10)]
 r=evaluate(frames,2,2,5)
 assert r['finished'] and r['stall_count']==0
 assert abs(r['startup_seconds']-.3)<1e-9

def test_missing_frame_blocks_contiguous_playback():
 r=evaluate([dict(id=0,received_s=0),dict(id=1),dict(id=2,received_s=1)],1,1,4)
 assert not r['finished'] and r['stall_measurement_censored']

def test_censored_stall_stops_at_observation_end():
 r=evaluate([dict(id=0,received_s=0),dict(id=1)],1,1,4)
 assert r['stall_seconds']==3 and r['unplayed_seconds']==1

def test_truncated_producer_does_not_shorten_media():
 r=evaluate([dict(id=0,received_s=0)],1,1,4,planned_count=5)
 assert not r['finished'] and r['unplayed_seconds']==4
