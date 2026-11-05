"""Actual network-to-player-state test with a CPU model holder (no GPU rendering)."""

import argparse
import asyncio
import json
from pathlib import Path
import time
import types
import torch
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from live_priority import LivePriorityState
from NTC import NeuralTransformationCache
from .transport import ALPN
from .send_scene import send_scene


class CPUDisplay:
    current_timestep = 0

    def update_gaussian_data(self, gaussians):
        self.gaussians = gaussians

    def sort_and_update(self, camera):
        pass


def cpu_model_holder(self, renderer):
    if self._ntc_single is not None:
        return
    if not self.ntc_paths:
        return
    self._ensure_ntc_conf()
    self._ensure_bounds(renderer)
    raw = self._load_raw_checkpoint(self.ntc_paths[min(self.ntc_paths)])
    model = torch.nn.Module()
    model.register_parameter(
        "params", torch.nn.Parameter(torch.zeros_like(raw["model.params"]))
    )
    self._ntc_single = NeuralTransformationCache(model, self._xyz_min, self._xyz_max)


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    live = LivePriorityState(
        str(args.output / "cache"), total_frames_hint=args.updates + 1
    )
    live._ensure_ntc_single = types.MethodType(cpu_model_holder, live)
    renderer = CPUDisplay()
    live.start_receiver("127.0.0.1", 0, cert=args.cert, key=args.key)
    try:
        deadline = time.monotonic() + 10
        while not hasattr(live, "bound_port"):
            if live.status == "error":
                raise RuntimeError(live.err)
            if time.monotonic() > deadline:
                raise TimeoutError("receiver startup")
            await asyncio.sleep(0.01)
        conf = QuicConfiguration(
            is_client=True, alpn_protocols=ALPN, server_name="localhost"
        )
        conf.load_verify_locations(args.cert)
        async with connect(
            "127.0.0.1", live.bound_port, configuration=conf
        ) as connection:
            rows = await send_scene(connection, args.scene, args.updates)
        deadline = time.monotonic() + 15
        while len(live.ntc_paths) < args.updates or not live._q.empty():
            live.process_new_files(renderer, None)
            if time.monotonic() > deadline:
                raise TimeoutError(live.err)
            await asyncio.sleep(0.01)
        live.process_new_files(renderer, None)
        assert not live.err, live.err
        assert live.max_play == args.updates
        assert live._resolve_available_ntc_index(args.updates + 10) == (None, False)
        for frame in range(args.updates):
            ntc = live.get_ntc_for_index(frame)
            if ntc is None:
                raise RuntimeError(live.err)
            expected = torch.load(
                args.scene / f"NTCs/NTC_{frame:06d}.pth",
                map_location="cpu",
                weights_only=True,
            )
            assert torch.equal(ntc.model.params.detach(), expected["model.params"])
            assert torch.equal(ntc.xyz_bound_min, expected["xyz_bound_min"])
            assert torch.equal(ntc.xyz_bound_max, expected["xyz_bound_max"])
            assert live.get_add_for_index(frame) is not None
        report = dict(
            updates=args.updates,
            network_chunks=len(rows),
            max_play=live.max_play,
            source="Original Flame Steak trained FP32 checkpoints",
            verified="Real QUIC receiver thread, committed-file bridge, original lazy player providers, exact parameter/bounds loading, additions, contiguous playback gate",
            unverified="CUDA inference, GPU rasterization and graphical playback; CPU model holder and renderer stub replace CUDA model construction, rendering and GPU additions conversion",
        )
        (args.output / "REPORT.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        live.request_stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--updates", type=int, default=3)
    p.add_argument("--cert", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--output", type=Path, required=True)
    asyncio.run(run(p.parse_args()))
