"""Feedback policies for offered workload size, NOT NTC quality encoders.

Inputs in milliseconds, bytes and payload bytes/sec. Fixed thresholds are
predeclared experimental settings, not universal congestion thresholds.
"""


class Controller:
    def __init__(self, policy, sizes):
        if policy not in ('fixed-low', 'fixed-high', 'queue', 'rtt', 'composite'):
            raise ValueError('unknown feedback policy')
        if not sizes or any(s <= 0 for s in sizes) or sorted(sizes) != list(sizes):
            raise ValueError('sizes must be positive and sorted')
        self.policy, self.sizes = policy, sizes
        self.level = 0 if policy == 'fixed-low' else len(sizes) - 1
        self.healthy = 0
        self.reason = 'initial'

    def update(self, s):
        if self.policy.startswith('fixed'):
            self.reason = 'fixed'
            return self.level
        queue = s['age_ms'] > 500
        rtt = s['srtt_ms'] > max(100, 1.5 * s['min_rtt_ms'])
        stalled = s['queue_bytes'] > 0 and s['ack_silence_ms'] > 1000
        pressure = queue if self.policy == 'queue' else rtt
        if self.policy == 'composite':
            pressure = queue or rtt or stalled
        if pressure:
            self.level = max(0, self.level - 1)
            self.healthy = 0
            self.reason = '+'.join(n for n, v in [('queue', queue), ('rtt', rtt), ('no_ack', stalled)] if v)
        else:
            self.healthy += 1
            self.reason = 'hold'
            if self.healthy >= 8:
                self.level = min(len(self.sizes) - 1, self.level + 1)
                self.healthy = 0
                self.reason = 'probe_up'
        return self.level
