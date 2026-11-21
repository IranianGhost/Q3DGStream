import json
import time
import unittest
from urllib.request import urlopen

import numpy as np

from live_preview import LivePreview


class LivePreviewTests(unittest.TestCase):
    def test_browser_receives_current_frame(self):
        preview = LivePreview(frames=299, port=0)
        preview.start()
        try:
            pixels = np.zeros((16, 20, 3), dtype=np.uint8)
            preview.publish(pixels, frame=114, tier=2, ms=325.0)
            deadline = time.monotonic() + 3
            while True:
                with urlopen(preview.url + "/status") as response:
                    status = json.load(response)
                if status["seq"] > 0:
                    break
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            self.assertEqual((status["frame"], status["tier"], status["total"]),
                             (114, 2, 299))
            with urlopen(preview.url + "/frame.jpg") as response:
                self.assertEqual(response.headers.get_content_type(), "image/jpeg")
                self.assertGreater(len(response.read()), 100)
        finally:
            preview.close()
