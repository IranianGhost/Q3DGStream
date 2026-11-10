"""Real QUIC -> native player providers -> CUDA NTC -> native CUDA rasterizer.

No GUI context is created. Presentation-only OpenGL initialization is bypassed;
Gaussian upload, NTC updates, additions and rasterization reuse player methods.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from live_priority import LivePriorityState
from renderer_cuda import CUDARenderer
import util
from .transport import ALPN, file_hash
from .send_scene import send_scene


class HeadlessRenderer(CUDARenderer):
    def __init__(self,width=640,height=480):
        self.NTCs=[];self.additional_3dgs=[];self.current_timestep=0
        self.gaussians=None;self.init_gaussians=None
        self.use_ntc_mask=True;self.ntc_debug_mask=False;self.ntc_debug_dxyz=False
        self.ntc_debug_interval=50;self.ntc_rotation_mode='none'
        self.ntc_rotation_scale=.08;self.ntc_translation_scale=.42;
        self.additions_opacity_scale=1.;self.additions_scale_scale=1.
        self.raster_settings=dict(image_height=height,image_width=width,tanfovx=1.,tanfovy=1.,
            bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=None,projmatrix=None,
            sh_degree=1,campos=None,prefiltered=False,antialiasing=False,debug=False)


async def run(a):
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
    a.output.mkdir(parents=True,exist_ok=False)
    live=LivePriorityState(str(a.output/'cache'),total_frames_hint=a.updates+1)
    renderer=HeadlessRenderer()
    live.start_receiver('127.0.0.1',0,cert=a.cert,key=a.key)
    try:
        limit=time.monotonic()+15
        while not hasattr(live,'bound_port'):
            if live.status=='error':raise RuntimeError(live.err)
            if time.monotonic()>limit:raise TimeoutError('receiver startup')
            await asyncio.sleep(.01)
        conf=QuicConfiguration(is_client=True,alpn_protocols=ALPN,server_name='localhost')
        conf.load_verify_locations(a.cert)
        async with connect('127.0.0.1',live.bound_port,configuration=conf) as connection:
            rows=await send_scene(connection,a.scene,a.updates,max_inflight=128)
        limit=time.monotonic()+60
        while len(live.ntc_paths)<a.updates or not live._q.empty():
            live.process_new_files(renderer,None)
            if live.err:raise RuntimeError(live.err)
            if time.monotonic()>limit:raise TimeoutError('player loading')
            await asyncio.sleep(.01)
        live.process_new_files(renderer,None)
        assert live.max_play==a.updates and renderer.gaussians.xyz.is_cuda
        camera=util.Camera(480,640)
        xyz=renderer.gaussians.xyz
        low=torch.quantile(xyz,.02,dim=0).cpu().numpy()
        high=torch.quantile(xyz,.98,dim=0).cpu().numpy()
        center=(low+high)/2;radius=float(np.linalg.norm(high-low)/2)
        camera.target=center.astype(np.float32)
        camera.position=(center+np.array([0,0,-max(2.,radius*1.25)])).astype(np.float32)
        camera.zfar=max(100.,radius*10)
        renderer.update_camera_pose(camera);renderer.update_camera_intrin(camera)
        frames=[]
        with torch.no_grad():
            for i in range(a.updates+1):
                t=time.monotonic()
                if i:
                    ntc=live.get_ntc_for_index(i-1)
                    if ntc is None or live.err:raise RuntimeError(live.err or 'NTC unavailable')
                    expected=torch.load(a.scene/f'NTCs/NTC_{i-1:06d}.pth',map_location='cpu',weights_only=True)
                    assert ntc.model.params.dtype==expected['model.params'].dtype
                    assert torch.equal(ntc.model.params.detach().cpu(),expected['model.params'])
                    assert ntc.model.params.is_cuda and ntc.model.params.numel()==expected['model.params'].numel()
                    assert torch.equal(ntc.xyz_bound_min.cpu(),expected['xyz_bound_min'])
                    assert torch.equal(ntc.xyz_bound_max.cpu(),expected['xyz_bound_max'])
                    active,delta_xyz,delta_rot=ntc(renderer.gaussians.xyz)
                    assert bool(active.any()) and bool(torch.isfinite(delta_xyz).all()) and bool(torch.isfinite(delta_rot).all())
                    before=renderer.gaussians.xyz.clone()
                    renderer.query_NTC(renderer.gaussians.xyz,i-1)
                    renderer.current_timestep=i
                    motion=float((renderer.gaussians.xyz-before).norm(dim=1).max().item())
                    gaussians=renderer.cat_additions(i-1)
                else:motion=0.;gaussians=renderer.gaussians
                image,radii=renderer._rasterize(gaussians)
                torch.cuda.synchronize()
                assert torch.isfinite(image).all() and float(image.max())>0
                pixels=(image.clamp(0,1).permute(1,2,0)*255).byte().cpu().numpy()
                path=a.output/f'frame_{i:03d}.png';Image.fromarray(pixels).save(path)
                frames.append(dict(frame=i,gaussians=len(gaussians),motion_max=motion,
                    visible=int((radii>0).sum().item()),diagnostic_step_wall_ms=(time.monotonic()-t)*1000,
                    image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        assert any(f['motion_max']>0 for f in frames)
        assert len({f['image_sha256'] for f in frames})>1
        report=dict(camera=dict(position=camera.position.tolist(),target=camera.target.tolist(),fovy=float(camera.fovy),width=640,height=480),
            sources=[dict(path=str(p.relative_to(a.scene)),sha256=file_hash(p)) for p in sorted(a.scene.rglob('*')) if p.is_file() and (p.name=='init_3dgs.ply' or p.name=='config.json' or p.name in [f'NTC_{i:06d}.pth' for i in range(a.updates)] or p.name in [f'additions_{i:06d}.ply' for i in range(a.updates)])],
            gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
            renderer_settings=dict(translation_scale=renderer.ntc_translation_scale,rotation_mode=renderer.ntc_rotation_mode),
            updates=a.updates,chunks=len(rows),max_play=live.max_play,frames=frames,
            peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
            ntc_parameters=int(live._ntc_single.model.params.numel()),
            parameter_dtype=str(live._ntc_single.model.params.dtype),
            verified='Post-transfer integration: real QUIC, native lazy CUDA NTC providers, weight parity, original NTC wrapper, native renderer Gaussian motion/additions/rasterization, nonblack finite PNG output',
            unverified='Concurrent network/playback, interactive OpenGL presentation, camera equivalence to original capture, image quality metrics, compressed Publio assets')
        (a.output/'REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:live.request_stop()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cert',required=True);p.add_argument('--key',required=True);p.add_argument('--updates',type=int,default=3)
    asyncio.run(run(p.parse_args()))
