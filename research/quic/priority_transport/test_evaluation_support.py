import unittest
import numpy as np
from evaluation_support import parse_byte_counters, image_pair_metrics

class EvaluationTests(unittest.TestCase):
    def test_counters_require_both_endpoints_and_matching_payload(self):
        lines=['BYTE_COUNTERS {"role":"sender","unique_application_payload_bytes":67,"udp_tx_bytes":120,"udp_rx_bytes":20}', 'BYTE_COUNTERS {"role":"receiver","unique_application_payload_bytes":67,"udp_tx_bytes":21,"udp_rx_bytes":119}']
        self.assertEqual(parse_byte_counters(lines)['sender']['udp_tx_bytes'],120)
        with self.assertRaises(ValueError):parse_byte_counters(lines[:1])
        with self.assertRaises(ValueError):parse_byte_counters([lines[0],lines[1].replace(':67',':66')])
    def test_metrics_use_rgb_range_one_and_shape_checks(self):
        a=np.zeros((16,16,3),dtype=np.float32)
        b=np.ones_like(a)*.5
        self.assertAlmostEqual(image_pair_metrics(a,b)['psnr'],6.0205999,places=5)
        self.assertEqual(image_pair_metrics(a,a)['ssim'],1.)
        with self.assertRaises(ValueError):image_pair_metrics(a,b[:8])
        with self.assertRaises(ValueError):image_pair_metrics(a,b*3)
