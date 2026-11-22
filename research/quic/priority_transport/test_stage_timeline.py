import unittest
from stage_timeline import sample_display_events, merge_display_events
class TimelineTests(unittest.TestCase):
    def test_no_future_frame_leak_and_hold_last_display(self):
        events=[dict(display_ms=110.,ntc=0,ready_through='P0',file='p0.png'),dict(display_ms=125.,ntc=0,ready_through='P1',file='p1.png'),dict(display_ms=170.,ntc=1,ready_through='P0',file='next.png')]
        rows=sample_display_events(events,4,30.,100.,'initial.png')
        self.assertEqual([r['distorted'] for r in rows],['p1.png','p1.png','next.png','next.png'])
        self.assertEqual([r['reference_state'] for r in rows],[1,2,3,4])
        self.assertAlmostEqual(rows[0]['sample_ms'],133.333333333)
    def test_invalid_times_are_rejected(self):
        with self.assertRaises(ValueError):sample_display_events([dict(display_ms=float('nan'),file='a',ntc=0,ready_through='P0')],2,30,100,'initial.png')
        with self.assertRaises(ValueError):sample_display_events([],2,0,100,'initial.png')

    def test_on_time_nonzero_render_latency_is_not_marked_stale(self):
        events=[dict(display_ms=110+i*1000/30,ntc=i,ready_through='P0',file=f'{i}.png') for i in range(4)]
        rows=sample_display_events(events,4,30,100,'initial.png')
        self.assertEqual([r['displayed_ntc'] for r in rows],list(range(4)))
    def test_catchup_tick_is_included_and_uncaptured_ticks_rejected(self):
        ticks=[dict(render_ready_ms='190',render_file='caught.png',applied_up_to_ntc='1',applied_tiers_latest='2')]
        rows=merge_display_events([],ticks)
        self.assertEqual(rows[0]['ntc'],1)
        self.assertEqual(rows[0]['ready_through'],'P1')
        with self.assertRaises(ValueError):merge_display_events([], [dict(ticks[0],render_file='')])
