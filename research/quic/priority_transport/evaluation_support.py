"""Validated endpoint counters and full-reference image metrics for Quinn runs."""
import argparse,json,math,shutil,subprocess,tempfile
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


def score_sequence(reference_dir,distorted_dir,output,ffmpeg,vmaf_binary=None,fps=30.):
    from PIL import Image
    import torch,lpips
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    ref,dist=Path(reference_dir),Path(distorted_dir)
    names=sorted(x.name for x in dist.glob('frame_*.png'))
    if not names:raise ValueError('no captured frames')
    if any(not (ref/n).is_file() for n in names):raise ValueError('missing aligned reference frame')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=lpips.LPIPS(net='alex').eval().to(device)
    rows=[];shape=None
    with tempfile.TemporaryDirectory(dir=output) as temp:
        temp=Path(temp)
        with (temp/'ref.rgb').open('wb') as rf,(temp/'dist.rgb').open('wb') as df:
            for name in names:
                a=np.asarray(Image.open(ref/name).convert('RGB'));b=np.asarray(Image.open(dist/name).convert('RGB'))
                if shape is not None and shape!=a.shape:raise ValueError('varying frame dimensions')
                shape=a.shape
                row=image_pair_metrics(a.astype(np.float32)/255,b.astype(np.float32)/255,model,device)
                rows.append(dict(image=name,**row));rf.write(a.tobytes());df.write(b.tobytes())
        h,w,_=shape
        if vmaf_binary:
            for stem in ['dist','ref']:
                subprocess.run([ffmpeg,'-v','error','-threads','2','-f','rawvideo','-pix_fmt','rgb24',
                    '-s',f'{w}x{h}','-r',str(fps),'-i',str(temp/(stem+'.rgb')),
                    '-pix_fmt','yuv420p','-f','yuv4mpegpipe',str(temp/(stem+'.y4m'))],
                    check=True,capture_output=True)
            subprocess.run([vmaf_binary,'-r',str(temp/'ref.y4m'),'-d',str(temp/'dist.y4m'),
                '-o',str(output/'vmaf.json'),'--json','--threads','2','--quiet'],check=True,capture_output=True)
        else:
            cmd=[ffmpeg,'-v','error','-threads','2']
            for f in ['dist.rgb','ref.rgb']:cmd+=['-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i',str(temp/f)]
            cmd+=['-filter_complex_threads','2','-lavfi',f'[0:v]format=yuv420p[d];[1:v]format=yuv420p[r];[d][r]libvmaf=log_fmt=json:log_path={output/"vmaf.json"}:n_threads=2','-f','null','-']
            subprocess.run(cmd,check=True,capture_output=True)
    vmaf=json.loads((output/'vmaf.json').read_text())
    if len(vmaf['frames'])!=len(rows):raise ValueError('VMAF frame count mismatch')
    for r,v in zip(rows,vmaf['frames']):r['vmaf']=v['metrics']['vmaf']
    result=dict(reference=str(ref),distorted=str(dist),frames=rows,
       fps=fps,protocol='index-aligned RGB PSNR/SSIM, LPIPS alex [-1,1], VMAF default model via yuv420p',
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
