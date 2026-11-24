import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
import pytest
from camera_config import load_camera,apply_camera

spec=importlib.util.spec_from_file_location('adaptive_viewer_util',Path(__file__).parent/'viewer/util.py')
util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)


def config():
 return dict(schema='q3dg_camera_v1',width=200,height=100,
             intrinsics=[[100.,0.,80.],[0.,120.,60.],[0.,0.,1.]],world_to_camera=np.eye(4).tolist())


def test_calibrated_camera_projects_cv_points_with_independent_focals(tmp_path):
 p=tmp_path/'camera.json';p.write_text(json.dumps(config()));camera=util.Camera(100,200)
 apply_camera(camera,load_camera(p))
 point=np.array([.2,.1,2.,1.]);clip=camera.get_project_matrix()@camera.get_view_matrix()@point
 ndc=clip[:3]/clip[3];pixel=[(ndc[0]+1)*100,(1-ndc[1])*50]
 np.testing.assert_allclose(pixel,[90.,66.],atol=1e-4)
 camera.update_resolution(200,400)
 clip=camera.get_project_matrix()@camera.get_view_matrix()@point;ndc=clip[:3]/clip[3]
 np.testing.assert_allclose([(ndc[0]+1)*200,(1-ndc[1])*100],[180.,132.],atol=2e-4)
 np.testing.assert_allclose(camera.get_htanfovxy_focal(),[1.,100/240.,240.])


def test_centered_calibration_preserves_existing_perspective(tmp_path):
 camera=util.Camera(100,200);before=camera.get_project_matrix().copy()
 d=config();f=100/(2*np.tan(camera.fovy/2));d['intrinsics']=[[f,0.,100.],[0.,f,50.],[0.,0.,1.]]
 p=tmp_path/'camera.json';p.write_text(json.dumps(d));apply_camera(camera,load_camera(p))
 np.testing.assert_allclose(camera.get_project_matrix(),before,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('bad',[dict(intrinsics=[[0,0,80],[0,120,60],[0,0,1]]),dict(world_to_camera=np.zeros((4,4)).tolist())])
def test_invalid_calibration_is_rejected(tmp_path,bad):
 d=config();d.update(bad);p=tmp_path/'camera.json';p.write_text(json.dumps(d))
 with pytest.raises(ValueError):load_camera(p)


def test_camera_cannot_silently_bind_to_different_scene(tmp_path):
 scene=tmp_path/'scene.ply';scene.write_bytes(b'original scene');d=config();d['scene_init_sha256']=hashlib.sha256(scene.read_bytes()).hexdigest()
 p=tmp_path/'camera.json';p.write_text(json.dumps(d));load_camera(p,scene)
 scene.write_bytes(b'changed scene')
 with pytest.raises(ValueError,match='scene'):load_camera(p,scene)
