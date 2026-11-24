"""Validated OpenCV camera poses and pinhole intrinsics for the optional viewer."""
import hashlib,json,math
from pathlib import Path
import numpy as np


def load_camera(path,scene_ply=None):
    data=json.loads(Path(path).read_text())
    if data.get('schema')!='q3dg_camera_v1':raise ValueError('Unknown camera schema')
    width,height=data['width'],data['height']
    if (not isinstance(width,int) or not isinstance(height,int) or min(width,height)<1
            or max(width,height)>16384):raise ValueError('Invalid camera dimensions')
    k=np.asarray(data['intrinsics'],dtype=np.float64);m=np.asarray(data['world_to_camera'],dtype=np.float64)
    if (k.shape!=(3,3) or m.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(m).all()
            or k[0,0]<=0 or k[1,1]<=0 or abs(k[0,1])+abs(k[1,0])>1e-10
            or not np.allclose(k[2],[0,0,1]) or not np.allclose(m[3],[0,0,0,1])
            or not np.allclose(m[:3,:3]@m[:3,:3].T,np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(m[:3,:3]),1.,atol=1e-5)):
        raise ValueError('Invalid pinhole intrinsics or rigid camera pose')
    if scene_ply is not None and data.get('scene_init_sha256'):
        digest=hashlib.sha256()
        with Path(scene_ply).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=data['scene_init_sha256']:raise ValueError('Camera belongs to a different scene')
    return data


def apply_camera(camera,data):
    if not hasattr(camera,'calibrated_intrinsics'):
        raise ValueError('Exact intrinsics require the repository optional viewer')
    m=np.asarray(data['world_to_camera'],dtype=np.float64);rot=m[:3,:3]
    center=-rot.T@m[:3,3]
    camera.position=center.astype(np.float32)
    camera.target=(center+rot.T[:,2]).astype(np.float32)
    camera.up=(-rot.T[:,1]).astype(np.float32)
    camera.calibrated_intrinsics=(np.asarray(data['intrinsics'],dtype=np.float64),data['width'],data['height'])
    fy=data['intrinsics'][1][1]
    camera.fovy=2*math.atan(data['height']/(2*fy))
    camera.is_pose_dirty=True;camera.is_intrin_dirty=True
