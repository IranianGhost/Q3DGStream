"""Run the native CUDA viewer and QUIC sender in a private network namespace."""
import argparse
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration

from .live_playback import choose_optional_bytes, write_snapshot
from .send_scene import collect_scene
from .transport import ALPN, FileJob, send_files, send_whole_files

PROFILES = {
    "clean": [],
    "jitter": ["delay", "15ms", "5ms", "distribution", "normal", "rate", "60mbit"],
    "loss1": ["delay", "10ms", "loss", "random", "1%", "rate", "60mbit"],
    "loss3": ["delay", "10ms", "loss", "random", "3%", "rate", "60mbit"],
    "dynamic": [],
}


def command(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def qdisc():
    return json.loads(command("tc", "-j", "-s", "qdisc", "show", "dev", "lo", capture_output=True).stdout)


def terminate(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


async def wait_for_feedback(output, viewer):
    until = time.monotonic() + 30
    while time.monotonic() < until and viewer.poll() is None:
        feedback = output / "feedback.json"
        if feedback.exists():
            state = json.loads(feedback.read_text())
            if state["port"]:
                return state["port"]
        await asyncio.sleep(0.05)
    raise RuntimeError("viewer did not publish a QUIC listener; see viewer.log")


async def stream(args, port, started, network):
    init, config, frames, has_additions = collect_scene(args.scene, args.updates)
    manifest = args.output / "manifest.json"
    write_snapshot(manifest, dict(require_additions=has_additions, updates=args.updates,
                                  checkpoint_format="original", complete_updates_only=True))
    tls = QuicConfiguration(is_client=True, alpn_protocols=ALPN, server_name="localhost",
                            congestion_control_algorithm=args.cc, max_datagram_size=1400)
    tls.load_verify_locations(args.output / "cert.pem")
    frames_log = []
    async with connect("127.0.0.1", port, configuration=tls) as connection:
        async def transfer(jobs):
            if args.mode == "whole":
                return await send_whole_files(connection, jobs)
            return await send_files(connection, jobs, max_inflight=128, policy="strict")

        rows = await transfer([
            FileJob(manifest, "priority_manifest.json", 0),
            FileJob(init, "init_3dgs.ply", 0),
            FileJob(config, "NTCs/config.json", 0),
        ])
        write_snapshot(args.output / "metadata.json", dict(
            finished_seconds=time.monotonic() - started, chunks=len(rows),
            bytes=sum(row["size"] for row in rows),
        ))
        for index, ntc, addition in frames:
            frame_started = time.monotonic()
            if args.profile == "dynamic" and index in (args.updates // 3, 2 * args.updates // 3):
                impaired = index == args.updates // 3
                network.append(dict(frame=index, before=qdisc(), at=frame_started - started))
                command("tc", "qdisc", "replace", "dev", "lo", "root", "netem", "limit", "4096",
                        *(PROFILES["loss1"] if impaired else ["delay", "0ms"]))
            jobs = [FileJob(ntc, f"NTCs/{ntc.name}", 0)]
            if addition is not None:
                jobs.append(FileJob(addition, f"additional_3dgs/{addition.name}", 0))
            required = sum(job.path.stat().st_size for job in jobs)
            feedback = json.loads((args.output / "feedback.json").read_text())
            optional = 0
            if args.policy == "fixed":
                optional = args.optional_bytes
            elif args.policy == "adaptive":
                optional = choose_optional_bytes(feedback, frame_started, required, 1 / args.fps,
                                                 args.prebuffer, args.optional_bytes)
            if optional:
                payload = args.output / "optional.payload"
                with payload.open("wb") as handle:
                    handle.truncate(optional)
                jobs.append(FileJob(payload, f"experiment/optional_{index:06d}.bin", 3))
            rows = await transfer(jobs)
            frames_log.append(dict(index=index, required_bytes=required, optional_bytes=optional,
                                   started_seconds=frame_started - started,
                                   finished_seconds=time.monotonic() - started,
                                   chunks=len(rows), feedback=feedback))
            write_snapshot(args.output / "sender.json", frames_log)
    return frames_log


async def run(args):
    parent_namespace = os.environ.get("Q3DG_PARENT_NETNS")
    interfaces = json.loads(command("ip", "-j", "link", "show", capture_output=True).stdout)
    if not parent_namespace or parent_namespace == os.readlink("/proc/self/ns/net") or [
        interface["ifname"] for interface in interfaces
    ] != ["lo"]:
        raise RuntimeError("run only via the private network namespace launcher")
    if args.updates < 1 or args.optional_bytes < 0 or args.fps <= 0:
        raise ValueError("invalid workload")
    if args.mode == "whole" and args.policy != "none":
        raise ValueError("optional loads require the chunk scheduler")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    command("ip", "link", "set", "lo", "up")
    if args.profile != "clean":
        command("tc", "qdisc", "add", "dev", "lo", "root", "netem", "limit", "4096",
                *(PROFILES[args.profile] or ["delay", "0ms"]))
    network = [dict(frame=0, before=qdisc())]
    root = Path(args.desktop_root)
    command("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(args.output / "key.pem"), "-out", str(args.output / "cert.pem"),
            "-days", "1", "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    display = f":{args.display}"
    auth = args.output / "Xauthority"
    auth.touch(mode=0o600)
    command("xauth", "-f", str(auth), "add", display, ".", secrets.token_hex(16))
    env = dict(os.environ, DISPLAY=display, XAUTHORITY=str(auth), PYOPENGL_PLATFORM="egl",
               Q3DG_GLFW_CONTEXT_API="egl", NTC_DEBUG="0", NTC_DEBUG_DXYZ="0", LOOP_PLAYBACK="0",
               LD_LIBRARY_PATH=f"{root}/root/usr/lib:{os.environ.get('LD_LIBRARY_PATH', '')}")
    camera = json.loads(Path("research/quic/priority_transport/evidence/gpu-server/REPORT.json").read_text())["camera"]
    camera.update(up=[0, -1, 0], yaw=-1.5707963267948966, pitch=0, zfar=250)
    write_snapshot(args.output / "camera.json", camera)
    env.update(FIXED_CAMERA_JSON=str(args.output / "camera.json"), LOCK_CAMERA="1")
    with ExitStack() as stack:
        desktop_log = stack.enter_context((args.output / "xvnc.log").open("w"))
        xvnc = subprocess.Popen([
            str(root / "root/opt/TurboVNC/bin/Xvnc"), display, "-auth", str(auth),
            "-nolisten", "tcp", "-localhost", "-rfbport", "0",
            "-dridir", "/usr/lib/x86_64-linux-gnu/dri", "-geometry", "1440x900", "-depth", "24",
        ], stdout=desktop_log, stderr=desktop_log)
        stack.callback(terminate, xvnc)
        for _ in range(100):
            if subprocess.run(["xdpyinfo"], env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0:
                break
            if xvnc.poll() is not None:
                raise RuntimeError("private X display failed; see xvnc.log")
            await asyncio.sleep(0.05)
        else:
            raise TimeoutError("private X display")
        viewer_log = stack.enter_context((args.output / "viewer.log").open("w"))
        viewer = subprocess.Popen([
            str(root / "root/opt/VirtualGL/bin/vglrun"), "-d", "egl0", sys.executable, "-u", "main.py",
            "--priority_listen", "0", "--priority_cache", str(args.output / "cache"),
            "--priority_cert", str(args.output / "cert.pem"), "--priority_key", str(args.output / "key.pem"),
            "--frames", str(args.updates + 1), "--video_fps", str(args.fps),
            "--priority_experiment", str(args.output), "--priority_prebuffer", str(args.prebuffer),
            "--priority_experiment_timeout", str(args.timeout),
        ], env=env, stdout=viewer_log, stderr=viewer_log)
        stack.callback(terminate, viewer)
        error = None
        try:
            port = await wait_for_feedback(args.output, viewer)
            started = time.monotonic()
            write_snapshot(args.output / "transfer-start.json", dict(at=started))
            await asyncio.wait_for(stream(args, port, started, network), args.timeout)
            write_snapshot(args.output / "transfer-done.json", dict(at=time.monotonic()))
            await asyncio.to_thread(viewer.wait, max(1, args.timeout - (time.monotonic() - started) + 3))
            if viewer.returncode or not (args.output / "playback.json").exists():
                raise RuntimeError(f"viewer exit {viewer.returncode}; see viewer.log")
        except Exception as exc:
            error = repr(exc)
        sent = args.output / "sender.json"
        rows = json.loads(sent.read_text()) if sent.exists() else []
        playback = args.output / "playback.json"
        result = dict(
            profile=args.profile, policy=args.policy, cc=args.cc, mode=args.mode, fps=args.fps,
            updates=args.updates, prebuffer=args.prebuffer, optional_limit_bytes=args.optional_bytes,
            frames_transferred=len(rows), optional_bytes=sum(row["optional_bytes"] for row in rows),
            required_update_bytes=sum(row["required_bytes"] for row in rows),
            network=network + [dict(final=qdisc())], error=error,
            metadata=json.loads((args.output / "metadata.json").read_text())
            if (args.output / "metadata.json").exists() else None,
            playback=json.loads(playback.read_text()) if playback.exists() else None,
        )
        write_snapshot(args.output / "REPORT.json", result)
        print(json.dumps({**result, "playback": {
            key: value for key, value in (result["playback"] or {}).items()
            if key not in ("presentations_seconds", "readiness_seconds")
        }}), flush=True)
        if error or not result["playback"]["finished"]:
            raise RuntimeError(error or "playback did not finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--output", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--desktop-root", default="/local/users/abeigi/q3dg-desktop")
    parser.add_argument("--display", type=int, default=97)
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--prebuffer", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--profile", choices=PROFILES, default="clean")
    parser.add_argument("--policy", choices=["none", "fixed", "adaptive"], default="none")
    parser.add_argument("--optional-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--cc", choices=["reno", "cubic"], default="cubic")
    parser.add_argument("--mode", choices=["chunked", "whole"], default="chunked")
    asyncio.run(run(parser.parse_args()))
