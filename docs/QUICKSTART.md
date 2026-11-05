# Quickstart

A checkout can be tested without CUDA. GPU inference and rendering require the NVIDIA dependencies described in [Ubuntu setup](linux.md).

## Source and format tests

```bash
git clone https://github.com/publioelon/Q3DGStream.git
cd Q3DGStream

python3 q3dgstream.py demo
python3 q3dgstream.py test
python3 q3dgstream.py doctor
```

`demo` creates a synthetic INT4 priority stage. `test` runs Python source and binary-format checks, and `doctor` reports which optional dependencies and scene files are available.

## CUDA and scene setup

Install the dependencies with `./scripts/setup_linux_venv.sh`, or activate an existing compatible environment:

```bash
source .venv/bin/activate
source scripts/env.sh
python3 q3dgstream.py doctor
```

The downloader places the Flame Steak example under `data/scenes/flame_steak/` unless `Q3DGSTREAM_SCENE_DIR` is set:

```bash
./scripts/download_flame_steak.sh
```

For externally stored checkpoints and saliency predictions, see [Configuration](CONFIGURATION.md).

## TCP baseline

Start the viewer and sender in separate terminals from the repository root.

**Viewer**

```bash
python main.py --tcp_listen 5001 --tcp_bind 127.0.0.1 \
  --tcp_cache "$Q3DGSTREAM_CACHE_DIR/tcp" --tcp_clear_cache \
  --frames 300 --video_fps 30 --autoplay
```

**Sender**

```bash
python tcp_fvv_sender.py --host 127.0.0.1 --port 5001 \
  --root "$Q3DGSTREAM_SCENE_DIR" \
  --start 0 --end 298 --ntc_stride 1 --no_additions
```

This runs the original TCP playback path. It does not send prioritized INT4 stages.

## Inspect a compressed stage

After generating the priority files:

```bash
python3 q3dgstream.py inspect-stage \
  "$Q3DGSTREAM_STAGES_DIR/NTC_000114_P0.qblk"
```

The inspector prints the frame number, priority, and block count. Details are in [Formats](FORMATS.md). For the GPU reconstruction measurements and live QUIC experiments, follow [GPU validation](GPU_VALIDATION.md).

## Prioritized QUIC reception

The existing player now has an optional `--priority_listen` mode. For commands,
original-checkpoint tests, measurements and GPU limitations, see
[Priority transport and player report](PRIORITY_PLAYER_REPORT.md).
