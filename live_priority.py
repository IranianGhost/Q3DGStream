"""Priority QUIC reception for Publio's existing player, using complete NTC updates.

This reuses its lazy NTC/additions providers. The GUI/inference still needs CUDA.
"""

import asyncio
import json
from pathlib import Path
import threading
import time

from live_tcp import LiveTCPState
from research.quic.priority_transport.player_bridge import (
    PlayerCacheBridge,
    ready_display_frame,
)


class LivePriorityState(LiveTCPState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hold_last_enabled = False
        self._base_idx = 0
        self._require_additions = True
        self._bridge = PlayerCacheBridge(self.cache_root, self._q.put)

    def _ensure_ntc_single(self, renderer):
        # Metadata can arrive before the checkpoint carrying its spatial bounds.
        if (
            not self.ntc_paths
            and not (self.cache_root / "NTCs/progressive/bounds.json").exists()
        ):
            return
        return super()._ensure_ntc_single(renderer)

    def _load_raw_checkpoint(self, path):
        import torch

        return torch.load(path, map_location="cpu", weights_only=True)

    def _ensure_bounds(self, renderer):
        if self._xyz_min is not None and self._xyz_max is not None:
            return
        import torch

        bounds = self.cache_root / "NTCs/progressive/bounds.json"
        if bounds.exists():
            data = json.loads(bounds.read_text())
            lo = torch.tensor(
                data["xyz_bound_min"], dtype=torch.float32, device=self._torch_device()
            )
            hi = torch.tensor(
                data["xyz_bound_max"], dtype=torch.float32, device=self._torch_device()
            )
        elif self.ntc_paths:
            from util_3dgstream import _extract_bounds_from_ckpt

            checkpoint = self._load_raw_checkpoint(self.ntc_paths[min(self.ntc_paths)])
            lo, hi = _extract_bounds_from_ckpt(
                checkpoint, torch.device(self._torch_device())
            )
        else:
            raise RuntimeError("waiting for checkpoint bounds")
        if (
            lo is None
            or hi is None
            or tuple(lo.shape) != (3,)
            or tuple(hi.shape) != (3,)
            or not bool(
                torch.isfinite(lo).all()
                and torch.isfinite(hi).all()
                and (hi > lo).all()
            )
        ):
            raise ValueError("invalid received NTC bounds")
        self._xyz_min, self._xyz_max = lo, hi

    def _committed(self, name):
        self._bridge.committed(name)
        now = time.perf_counter()
        size = (self.cache_root / name).stat().st_size
        with self._lock:
            self.files += 1
            self.total_bytes += size
            self.last_file = name
            self.last_file_mb = size / (1024**2)
            elapsed = max(1e-6, now - self._monitor_start_t)
            self.rx_mbps_avg = self.total_bytes * 8 / 1e6 / elapsed
            if name.endswith(".pth") or name.endswith(".qblk"):
                self.ntc_bytes += size
                if self._first_ntc_t is None:
                    self._first_ntc_t = now
                self._last_ntc_t = now
                self.ntc_files = len(list((self.cache_root / "NTCs").glob("NTC_*.pth")))
            elif name == "init_3dgs.ply":
                self.init_bytes += size
            elif name == "NTCs/config.json":
                self.config_bytes += size
            self.status = "connected"

    def start_receiver(self, bind_host, port, clear_cache=False, cert=None, key=None):
        if bind_host not in ("127.0.0.1", "localhost"):
            raise ValueError("local-only QUIC receiver")
        if clear_cache:
            raise ValueError(
                "Use a new empty --priority_cache; automatic deletion is disabled"
            )
        if self.cache_root.exists() and any(self.cache_root.iterdir()):
            raise ValueError("priority cache must be empty")
        if not cert or not key:
            raise ValueError("priority TLS certificate and key required")
        self.cache_root.mkdir(parents=True, exist_ok=True)

        async def listen():
            from aioquic.asyncio import serve
            from aioquic.quic.configuration import QuicConfiguration
            from research.quic.priority_transport.transport import ALPN, Receiver

            receiver = Receiver(self.cache_root, on_file=self._committed)
            conf = QuicConfiguration(is_client=False, alpn_protocols=ALPN)
            conf.load_cert_chain(cert, key)
            server = await serve(
                "127.0.0.1", port, configuration=conf, stream_handler=receiver.handler
            )
            self.bound_port = server._transport.get_extra_info("sockname")[1]
            with self._lock:
                self.status = "listening"
                self.connected = f"QUIC 127.0.0.1:{self.bound_port}"
            try:
                while not self._stop.is_set():
                    if receiver.errors:
                        with self._lock:
                            self.err = receiver.errors[-1]
                    await asyncio.sleep(0.05)
            finally:
                server.close()
                await receiver.close()

        def worker():
            try:
                asyncio.run(listen())
            except Exception as exc:
                with self._lock:
                    self.err = repr(exc)
                    self.status = "error"

        self._thread = threading.Thread(
            target=worker, daemon=True, name="priority-quic-receiver"
        )
        self._thread.start()

    def process_new_files(self, renderer, camera):
        super().process_new_files(renderer, camera)
        manifest = self.cache_root / "priority_manifest.json"
        if manifest.exists():
            self._require_additions = json.loads(manifest.read_text()).get(
                "require_additions", True
            )
        self.max_play = (
            ready_display_frame(
                self.ntc_paths,
                self.add_paths,
                self.total_frames_hint,
                self._require_additions,
            )
            if self.init_ply_ok and self.config_ok
            else 0
        )

    def request_stop(self):
        super().request_stop()
        try:
            self._prefetch_q.put_nowait(None)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=3)
