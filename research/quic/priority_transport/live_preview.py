"""A small browser preview for the rendered NTC frames."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import queue
import threading


PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>NTC live preview</title>
<style>
body { margin: 0; background: #151515; color: #eee; font: 16px system-ui; }
header { padding: 12px 20px; display: flex; gap: 24px; }
img { display: block; margin: auto; max-width: 100vw; max-height: calc(100vh - 60px); }
</style>
<header><span id="frame">Waiting for P0…</span><span id="tier"></span><span id="rate"></span></header>
<img id="image" alt="Live 3D Gaussian splatting frame">
<script>
let last = -1;
let previousFrame = -1;
const frameTimes = [];
async function poll() {
  try {
    const response = await fetch('/status', {cache: 'no-store'});
    const state = await response.json();
    if (state.seq !== last && state.seq > 0) {
      last = state.seq;
      document.getElementById('frame').textContent =
        'NTC ' + (state.frame + 1) + ' / ' + state.total;
      document.getElementById('tier').textContent =
        'P0–P' + state.tier + ' · ' + Math.round(state.ms) + ' ms into frame';
      if (state.frame !== previousFrame) {
        previousFrame = state.frame;
        frameTimes.push(performance.now());
      }
      const now = performance.now();
      while (frameTimes.length && now - frameTimes[0] > 1500) frameTimes.shift();
      if (frameTimes.length > 1) {
        const rate = (frameTimes.length - 1) * 1000 /
          (frameTimes[frameTimes.length - 1] - frameTimes[0]);
        document.getElementById('rate').textContent = rate.toFixed(1) + ' NTC/s';
      }
      document.getElementById('image').src = '/frame.jpg?seq=' + last;
    }
  } catch (_) {}
  setTimeout(poll, 20);
}
poll();
</script>
</html>
"""


class LivePreview:
    def __init__(self, frames, port=8765):
        self._lock = threading.Lock()
        self._image = b""
        self._error = None
        self._status = {"seq": 0, "frame": -1, "tier": 0, "ms": 0, "total": frames}
        self._jobs = queue.Queue(maxsize=8)
        self._server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._web = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._encoder = threading.Thread(target=self._encode, daemon=True)

    def _handler(self):
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                with preview._lock:
                    status = dict(preview._status)
                    image = preview._image
                if path == "/":
                    body, mime = PAGE.encode(), "text/html; charset=utf-8"
                elif path == "/status":
                    body, mime = json.dumps(status).encode(), "application/json"
                elif path == "/frame.jpg" and image:
                    body, mime = image, "image/jpeg"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        return Handler

    def start(self):
        self._web.start()
        self._encoder.start()

    def publish(self, rgb, frame, tier, ms):
        if self._error is not None:
            raise RuntimeError("Preview encoding stopped") from self._error
        item = (rgb, frame, tier, ms)
        try:
            self._jobs.put_nowait(item)
        except queue.Full:
            # The viewer must never stall the GPU because JPEG encoding lags.
            try:
                self._jobs.get_nowait()
                self._jobs.task_done()
            except queue.Empty:
                pass
            try:
                self._jobs.put_nowait(item)
            except queue.Full:
                pass

    def _encode(self):
        from PIL import Image
        while True:
            item = self._jobs.get()
            if item is None:
                self._jobs.task_done()
                return
            try:
                rgb, frame, tier, ms = item
                picture = Image.fromarray(rgb)
                picture.thumbnail((960, 720))
                output = BytesIO()
                picture.save(output, "JPEG", quality=80)
                with self._lock:
                    self._image = output.getvalue()
                    self._status = {
                        "seq": self._status["seq"] + 1, "frame": frame,
                        "tier": tier, "ms": ms, "total": self._status["total"],
                    }
            except Exception as error:
                self._error = error
            finally:
                self._jobs.task_done()

    def close(self):
        self._jobs.put(None)
        self._jobs.join()
        self._server.shutdown()
        self._server.server_close()
