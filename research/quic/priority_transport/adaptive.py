"""Prepare Públio's verified models and run paired precision/priority experiments.

Models remain outside Git. The runtime and optional viewer are in this repository.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRACE = 'experiments/startup/traces/wifi_campus_231115-192852.txt'
PROFILES = ('campus', 'delay', 'jitter', 'loss', 'burst', 'controlled')


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def trace_profile(path):
    raw = path.read_bytes()
    points = [tuple(map(float, line.split())) for line in raw.decode().splitlines() if line.strip()]
    if (len(points) < 2 or any(len(point) != 2 for point in points) or points[0][0] != 0
            or any(not math.isfinite(t) or not math.isfinite(rate) or rate <= 0 for t, rate in points)
            or any(b[0] <= a[0] for a, b in zip(points, points[1:]))):
        raise ValueError('Trace must contain finite, strictly increasing timestamps and positive rates')
    return ','.join(f'{t:g}:{min(rate, 70):g}' for t, rate in points), hashlib.sha256(raw).hexdigest()


def prepare(args):
    from prepare_precision import prepare as prepare_groups
    if not 1 <= args.frames <= 299:
        raise ValueError('Frame count must be between 1 and 299')
    resources = args.resources.expanduser().resolve()
    manifest_path = resources / 'SHA256.json'
    expected_manifest = json.loads((HERE / 'UPSTREAM.json').read_text())['resource_manifest_sha256']
    if file_sha(manifest_path) != expected_manifest:
        raise ValueError('Resource manifest differs from the pinned reproduction bundle')
    manifest = json.loads(manifest_path.read_text())['files']
    entries = {}
    for name, digest in manifest.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or len(path.parts) < 2:
            raise ValueError('Invalid resource manifest path')
        relative = str(PurePosixPath(*path.parts[1:]))
        if relative in entries:
            raise ValueError('Ambiguous resource manifest')
        entries[relative] = digest

    def verify(relative):
        path = resources / relative
        if not path.resolve().is_relative_to(resources) or file_sha(path) != entries[relative]:
            raise ValueError(f'Resource checksum mismatch: {relative}')
        return path

    verify('inputs/scene/init_3dgs.ply')
    verify('inputs/scene/NTCs/config.json')
    verify('inputs/keep05/wire/NTC_000000.pth')
    trace = verify(TRACE)
    trace_spec, trace_sha = trace_profile(trace)
    results = verify('results/precision.zip')
    for frame in range(args.frames):
        for bits in (2, 4):
            verify(f'inputs/paired/int{bits}_keep20/wire/NTC_{frame:06}.pth')
    out = args.out.expanduser().resolve()
    config = dict(schema='q3dg_paired_assets_v1', frames=args.frames,
                  resources=str(resources), resource_manifest_sha256=file_sha(manifest_path),
                  scene=str(resources / 'inputs/scene'),
                  bootstrap_dir=str(resources / 'inputs/keep05/wire'),
                  trace=str(trace), trace_sha256=trace_sha)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'assets.json').exists() and json.loads((out / 'assets.json').read_text()) != config:
        raise ValueError('Asset workspace differs; use a new output directory')
    orders = out / 'priority_orders'
    orders.mkdir(exist_ok=True)
    with ZipFile(results) as archive:
        for frame in range(args.frames):
            name = f'NTC_{frame:06}.npz'
            raw = archive.read('priority_orders/' + name)
            target = orders / name
            if target.exists() and target.read_bytes() != raw:
                raise ValueError(f'Cached order changed: {name}')
            target.write_bytes(raw)
    prepared = out / 'prepared'
    if prepared.exists():
        sequence = prepared / 'sequence_manifest.json'
        if not sequence.is_file():
            prepared.rename(out / f'prepared-interrupted-{time.time_ns()}')
        else:
            data = json.loads(sequence.read_text())
            if data['frames'] != args.frames:
                raise ValueError('Prepared sequence length differs')
            for frame in data['manifests'].values():
                for bits, variant in frame['variants'].items():
                    for tier, group in variant['groups'].items():
                        path = prepared / 'send' / f"NTC_{frame['frame']:06}" / f'int{bits}' / f'{tier}.bin'
                        if file_sha(path) != group['sha256']:
                            raise ValueError(f'Prepared group changed: {path}')
    if not prepared.exists():
        prepare_groups(argparse.Namespace(
            int2_dir=resources / 'inputs/paired/int2_keep20/wire',
            int4_dir=resources / 'inputs/paired/int4_keep20/wire',
            orders=orders, bootstrap_dir=resources / 'inputs/keep05/wire',
            out=prepared, frames=args.frames, block_index_order=False))
    verify_prepared_recording(prepared, results, args.frames)
    pending = out / 'assets.pending'
    pending.write_text(json.dumps(config, indent=2) + '\n')
    pending.replace(out / 'assets.json')
    print(json.dumps(dict(assets=str(out), frames=args.frames, trace_profile=trace_spec), indent=2))


def verify_prepared_recording(prepared, results, frames):
    actual = json.loads((prepared / 'sequence_manifest.json').read_text())['manifests']
    with ZipFile(results) as archive:
        expected = json.loads(archive.read('study/prepared/sequence_manifest.json'))['manifests']
    for frame in range(frames):
        a, b = dict(actual[str(frame)]), dict(expected[str(frame)])
        a.pop('source_wire', None)
        b.pop('source_wire', None)
        if a != b:
            raise ValueError(f'Prepared groups differ from recorded input for NTC {frame}')


def build_command(args):
    assets = args.assets.expanduser().resolve()
    data = json.loads((assets / 'assets.json').read_text())
    if data.get('schema') != 'q3dg_paired_assets_v1':
        raise ValueError('Prepare a paired asset workspace first')
    if (not 1 <= args.frames <= data['frames'] or not math.isfinite(args.startup_ms)
            or args.startup_ms < 0 or not 1 <= args.max_buffer_frames <= 128
            or not 1 <= args.tiers <= 4 or args.capture_every < 0):
        raise ValueError('Invalid frame, startup, buffer, tier, or capture setting')
    if args.selector == 'fixed' and args.precision == 'adaptive':
        raise ValueError('Fixed-tier baselines require an explicit int2 or int4 precision')
    sequence = json.loads((assets / 'prepared/sequence_manifest.json').read_text())
    if sequence['frames'] != args.frames:
        raise ValueError('Frame count must match the prepared workspace; prepare a separate shorter workspace for a smoke test')
    startup = min(args.frames, max(1, math.ceil(args.startup_ms * 30 / 1000)))
    if startup > args.max_buffer_frames:
        raise ValueError('Startup target exceeds the playable buffer limit')
    cmd = [sys.executable, '-u', str(HERE / 'run_deadline.py'),
           '--frames', str(args.frames), '--fps', '30',
           '--prepared', str(assets / 'prepared'), '--wire-dir', data['bootstrap_dir'],
           '--scene-base', data['scene'], '--viewer', str(HERE / 'viewer'),
           '--out', str(args.out.expanduser().resolve()), '--live', '--netns',
           '--width', '1345', '--height', '1008', '--capture-every', str(args.capture_every),
           '--vod-startup-frames', str(startup), '--vod-max-buffer-frames', str(args.max_buffer_frames),
           '--precision-policy', {'adaptive': 'adaptive', 'int2': 'fixed2', 'int4': 'fixed4'}[args.precision],
           '--tier-selection', {'deadline': 'adaptive', 'buffer': 'buffer', 'fixed': 'fixed'}[args.selector],
           '--fixed-tiers', str(args.tiers), '--quinn-priority', args.priority,
           '--scheduling-mode', args.submission, '--bandwidth-estimator', args.estimator,
           '--feedback-ms', '50', '--initial-mbps', '70', '--utilization', '.92',
           '--sample-network', '--timeout', '300', '--delivery-timeout-ms', '90000']
    if args.profile == 'controlled':
        cmd += ['--rate-profile', '0:28,3:70,5:16.8,7:28', '--delay-ms', '10', '--loss-pct', '0']
    else:
        spec, digest = trace_profile(Path(data['trace']))
        if digest != data['trace_sha256']:
            raise ValueError('Campus trace changed after preparation')
        cmd += ['--rate-profile', spec, '--wifi-impairment', args.profile]
    if args.camera_config:
        cmd += ['--camera-config', str(args.camera_config.expanduser().resolve())]
    return cmd


def run(args):
    args.out = args.out.expanduser().resolve()
    cmd = build_command(args)
    if args.dry_run:
        print(json.dumps(dict(command=cmd, network='isolated Linux netem'), indent=2))
        return
    if sys.platform != 'linux':
        raise RuntimeError('Run GPU/network experiments on the Linux server')
    if args.out.exists():
        raise ValueError('Run output already exists; choose a fresh output directory')
    if os.environ.get('QS_REPRO_ROOTLESS_READY') != '1':
        cmd = [str(REPO / 'scripts/rootless_exec.sh'), *cmd]
    result = subprocess.run(cmd)
    if args.out.is_dir():
        files = ['src/main.rs', 'run_deadline.py', 'live_client.py', 'paired_groups.py', 'adaptive.py']
        record = dict(status='completed' if result.returncode == 0 else 'failed', returncode=result.returncode,
                      command=cmd, profile=args.profile, assets=str(args.assets.expanduser().resolve()),
                      frames=args.frames, fps=30, size=[1345,1008], capture_every=args.capture_every,
                      camera_config='camera-config.json' if args.camera_config else None,
                      runtime_sha256={name: file_sha(HERE / name) for name in files})
        (args.out / 'SESSION.json').write_text(json.dumps(record, indent=2) + '\n')
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def score(args):
    """Wrap a single recorded run in the existing study scorer's input schema."""
    run = args.run.expanduser().resolve()
    record = json.loads((run / 'SESSION.json').read_text())
    if record['status'] != 'completed':
        raise ValueError('Only completed runs can be scored')
    assets = Path(record['assets'])
    asset_data = json.loads((assets / 'assets.json').read_text())
    trial = run / 'priority'
    config = dict(frames=record['frames'], fps=record['fps'], size=record['size'],
                  capture_every=record['capture_every'], scene=asset_data['scene'], viewer=str(HERE / 'viewer'))
    study = run / 'scoring-study'
    study.mkdir(exist_ok=True)
    (study / 'study_config.json').write_text(json.dumps(config, indent=2)+'\n')
    link = study / 'prepared'
    if not link.exists(): link.symlink_to(assets / 'prepared', target_is_directory=True)
    if link.resolve() != (assets / 'prepared').resolve():
        raise ValueError('Scoring manifest changed')
    import csv
    with (study / 'study.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=['trial'])
        writer.writeheader();writer.writerow(dict(trial=str(trial)))
    cmd=[sys.executable, '-u', str(HERE / 'score_vod_study.py'), '--study', str(study),
         '--out', str(run / 'offline_quality'), '--vmaf', str(args.vmaf.resolve()), '--low-space']
    if (run / 'offline_quality').exists(): cmd.append('--resume')
    if record.get('camera_config'):
        cmd+=['--camera-config',str(run / record['camera_config'])]
    if args.reference_dir: cmd+=['--reference-dir',str(args.reference_dir.resolve())]
    subprocess.run(cmd, check=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='operation', required=True)
    a = commands.add_parser('prepare', help='Verify the supplied bundle and prepare exact paired groups')
    a.add_argument('--resources', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--frames', type=int, default=299)
    a.set_defaults(function=prepare)
    a = commands.add_parser('run', help='Run a real QUIC/GPU case using an explicit network preset')
    a.add_argument('--assets', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--profile', choices=PROFILES, default='campus')
    a.add_argument('--selector', choices=('deadline', 'buffer', 'fixed'), default='deadline')
    a.add_argument('--precision', choices=('adaptive', 'int2', 'int4'), default='adaptive')
    a.add_argument('--tiers', type=int, default=4)
    a.add_argument('--frames', type=int, default=299)
    a.add_argument('--startup-ms', type=float, default=500)
    a.add_argument('--max-buffer-frames', type=int, default=60)
    a.add_argument('--priority', choices=('equal', 'tiered'), default='tiered')
    a.add_argument('--submission', choices=('p0_first', 'concurrent', 'amirali_strict'), default='p0_first')
    a.add_argument('--estimator', choices=('hybrid', 'goodput'), default='hybrid')
    a.add_argument('--capture-every', type=int, default=30)
    a.add_argument('--camera-config', type=Path)
    a.add_argument('--dry-run', action='store_true')
    a.set_defaults(function=run)
    a = commands.add_parser('score', help='Replay applied tiers and compare sampled live pixels and image quality')
    a.add_argument('--run', type=Path, required=True)
    a.add_argument('--vmaf', type=Path, required=True)
    a.add_argument('--reference-dir', type=Path, help='Optional matching FP32 or aligned video PNGs')
    a.set_defaults(function=score)
    return p


if __name__ == '__main__':
    p = parser()
    try:
        args = p.parse_args()
        args.function(args)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        p.exit(1, f'ERROR: {exc}\n')
