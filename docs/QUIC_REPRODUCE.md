# Reproduce on gpu1

Run workloads and keep assets/output on the GPU server. The Mac only connects by SSH or views the remote desktop. The reference setup is Quadro RTX 8000, PyTorch 2.5.1+cu121, CUDA 12.1 and aioquic 1.7.2. Use an unoccupied private trial display; leave the interactive demo alone.

```bash
ssh -i ~/.ssh/q3dg_aau_itec abeigi@gpu1.itec.aau.at
cd /local/users/abeigi/Q3DGStream
source scripts/gpu1_env.sh
scene=/local/users/abeigi/Q3DGStream/data/scenes/flame_steak
dataset=/local/users/abeigi/Q3DGStream/data/datasets/publio-20261119
run_dir=$(mktemp -d /local/users/abeigi/quic-review.XXXXXX)
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "$run_dir/key.pem" -out "$run_dir/cert.pem" -days 7 \
  -subj /CN=localhost -addext subjectAltName=DNS:localhost

# Focused source/transport/player checks; an occupied GUI test port may skip.
python -m pytest -q research/quic/priority_transport tests/test_raster_compat.py tests/test_remote_player_start.py

# Transfer comparison: seven models, six profiles, two methods, three repeats.
bash scripts/run_gpu1_publio_matrix.sh --scope static --dataset "$dataset" \
  --cert "$run_dir/cert.pem" --key "$run_dir/key.pem" --output "$run_dir/static"

# Constructed urgency comparison: FIFO / strict / weighted, three repeats.
bash scripts/run_gpu1_publio_matrix.sh --scope priority --dataset "$dataset" \
  --cert "$run_dir/cert.pem" --key "$run_dir/key.pem" --output "$run_dir/priority"

# Actual viewer, one fixed-model case (other profiles/models: --help).
bash scripts/run_gpu1_publio_trial.sh --scene "$scene" --dataset "$dataset" \
  --variant INT4_keep05_PQAT --profile clean --output "$run_dir/native-fixed"

# Changing network: repeat with fixed-high and fixed-low for controls.
bash scripts/run_gpu1_publio_trial.sh --scene "$scene" --dataset "$dataset" \
  --profile dynamic --rep-policy adaptive --paced --updates 48 --fps 2 \
  --prebuffer 2 --timeout 170 --output "$run_dir/native-adaptive"
```

Each command needs a fresh output directory. Native trials require the prepared VirtualGL/Xvfb desktop dependencies. The launchers create a private user/network namespace and shape only its loopback; never apply these qdisc commands to the host network. Failure summaries remain in the output and the matrix returns a nonzero exit status for failed transfers.

Source archive: Google Drive file `1XFDbrOs_7DIXHAifwybihANpmlOzm8x6`; server file `data/downloads/publio-20261119/QNTC_Dataset.zip`, 3,560,597,072 bytes, SHA-256 `c57b84af76e8d4a90b9fcd84ff58b5a04bf607f07a1ad5682fc8487609159b06`.
The CSV was condensed from `/local/users/abeigi/q3dg-publio-eval-20261119/report/results.json`. That directory contains per-run inputs/hashes, events, samples, qlog, native playback records and GPU images; these are kept outside Git.

For interactive viewing use `scripts/start_gpu1_desktop.sh` and `scripts/run_gpu1_live_player.sh` on the server. In a second server terminal, read `run_dir=$(cat /local/users/abeigi/q3dg-desktop/latest-live)` and send with `python -m research.quic.priority_transport.send_scene --scene "$scene" --updates 30 --cert "$run_dir/cert.pem" --log "$run_dir/sender.json"`. From the Mac, `scripts/connect_gpu1_desktop.sh` opens an SSH tunnel and prints the browser URL. Camera settings are now explicit in `camera_flame_steak.json`, independent of removed historical reports.
