"""Validated endpoint counters and full-reference image metrics for Quinn runs."""
import argparse,json,math,os,shutil,subprocess,tempfile,time
from pathlib import Path
import numpy as np


def parse_byte_counters(lines):
    rows={}
    for line in lines:
        if 'BYTE_COUNTERS ' not in line:continue
        row=json.loads(line.split('BYTE_COUNTERS ',1)[1])
        role=row['role']
        if role not in ('sender','receiver') or role in rows:
            raise ValueError('duplicate or unknown endpoint counters')
        for key in ('unique_application_payload_bytes','udp_tx_bytes','udp_rx_bytes'):
            if not isinstance(row[key],int) or row[key]<0:raise ValueError('invalid byte count')
        rows[role]=row
    if set(rows)!=set(('sender','receiver')):raise ValueError('missing endpoint counters')
    if rows['sender']['unique_application_payload_bytes']!=rows['receiver']['unique_application_payload_bytes']:
        raise ValueError('selected and received unique payload differ')
    return rows


def image_pair_metrics(reference,distorted,lpips_model=None,device='cpu'):
    from skimage.metrics import structural_similarity
    a,b=np.asarray(reference),np.asarray(distorted)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<7:
        raise ValueError('expected matching RGB images at least 7x7')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(),b.min())<0 or max(a.max(),b.max())>1:
        raise ValueError('RGB data must be finite in [0,1]')
    mse=float(np.mean((a.astype(np.float64)-b)**2))
    row=dict(psnr=10*math.log10(1/mse) if mse else None,identical=mse==0,
             ssim=float(structural_similarity(a,b,channel_axis=2,data_range=1.)))
    if lpips_model is not None:
        import torch
        x=torch.from_numpy(a.copy()).permute(2,0,1)[None].to(device=device,dtype=torch.float32)*2-1
        y=torch.from_numpy(b.copy()).permute(2,0,1)[None].to(device=device,dtype=torch.float32)*2-1
        with torch.inference_mode():row['lpips']=float(lpips_model(x,y).item())
    return row


def score_vmaf(reference, distorted, names, output, ffmpeg, binary, fps, shape):
    height, width, _ = shape
    pad = f"format=rgb24,pad={width + width % 2}:{height + height % 2}:0:0,format=yuv420p"
    with tempfile.TemporaryDirectory(dir=output) as directory:
        work = Path(directory)
        inputs = {}
        # Present exactly the same frame ordering to FFmpeg as to the RGB metrics.
        for side, source in (("ref", reference), ("dist", distorted)):
            folder = work / side
            folder.mkdir()
            for index, name in enumerate(names):
                (folder / f"{index:06}.png").symlink_to((source / name).resolve())
            inputs[side] = ["-framerate", str(fps), "-start_number", "0", "-i", str(folder / "%06d.png")]
        base = [ffmpeg, "-v", "error", "-nostdin", "-threads", "2"]
        if not binary:
            filters = f"[0:v]{pad}[d];[1:v]{pad}[r];[d][r]libvmaf=log_fmt=json:log_path=vmaf.json:n_threads=2"
            subprocess.run([*base, *inputs["dist"], *inputs["ref"], "-filter_complex_threads", "2",
                            "-lavfi", filters, "-f", "null", "-"], cwd=output, check=True, capture_output=True)
            return

        pipes = {side: work / f"{side}.y4m" for side in inputs}
        for fifo in pipes.values():
            os.mkfifo(fifo, 0o600)
        processes = []

        def start(command, label):
            log = work / f"{label}.stderr"
            with log.open("wb") as stderr:
                proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=stderr)
            processes.append((proc, log))

        try:
            start([binary, "-r", str(pipes["ref"]), "-d", str(pipes["dist"]),
                   "-o", str(output / "vmaf.json"), "--json", "--threads", "2", "--quiet"], "vmaf")
            for side in inputs:
                start([*base, "-y", *inputs[side], "-filter_threads", "2", "-vf", pad,
                       "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", str(pipes[side])], side)
            deadline = time.monotonic() + 900
            while True:
                codes = [proc.poll() for proc, _ in processes]
                for code, (_, log) in zip(codes, processes):
                    if code not in (None, 0):
                        raise RuntimeError(f"{log.stem} exited {code}: {log.read_text(errors='replace').strip()}")
                if all(code == 0 for code in codes):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out waiting for FFmpeg and VMAF")
                time.sleep(.05)
        finally:
            for proc, _ in processes:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()


def score_sequence(reference_dir,distorted_dir,output,ffmpeg,vmaf_binary=None,fps=30.):
    from PIL import Image
    import torch,lpips
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    ref,dist=Path(reference_dir),Path(distorted_dir)
    names=sorted(x.name for x in dist.glob('frame_*.png'))
    if not names:raise ValueError('no captured frames')
    if any(not (ref/n).is_file() for n in names):raise ValueError('missing aligned reference frame')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=lpips.LPIPS(net='alex').eval().to(device)
    rows = []
    shape = None
    for name in names:
        with Image.open(ref / name) as reference, Image.open(dist / name) as distorted:
            a = np.asarray(reference.convert("RGB"))
            b = np.asarray(distorted.convert("RGB"))
        if shape is not None and shape != a.shape:
            raise ValueError("varying frame dimensions")
        shape = a.shape
        row = image_pair_metrics(a.astype(np.float32) / 255, b.astype(np.float32) / 255, model, device)
        rows.append(dict(image=name, **row))
    score_vmaf(ref, dist, names, output, ffmpeg, vmaf_binary, fps, shape)
    vmaf=json.loads((output/'vmaf.json').read_text())
    if len(vmaf['frames'])!=len(rows):raise ValueError('VMAF frame count mismatch')
    for r,v in zip(rows,vmaf['frames']):r['vmaf']=v['metrics']['vmaf']
    result=dict(reference=str(ref),distorted=str(dist),frames=rows,
       fps=fps,protocol='index-aligned RGB PSNR/SSIM, LPIPS alex [-1,1], VMAF default model via even-padded yuv420p',
       caveat='The caller must establish camera, timestamp and reference alignment; a short render sanity clip is not streaming QoE.')
    (output/'quality.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--distorted',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ffmpeg',default=shutil.which('ffmpeg'))
    p.add_argument('--vmaf',help='standalone official VMAF CLI when ffmpeg lacks libvmaf')
    a=p.parse_args()
    if not a.ffmpeg:p.error('ffmpeg with libvmaf is required')
    print(json.dumps(score_sequence(a.reference,a.distorted,a.output.resolve(),a.ffmpeg,a.vmaf),indent=2))
