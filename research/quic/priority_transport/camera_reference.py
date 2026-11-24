"""Build a scene-bound cam00 configuration from the official 3DGStream fixture.

Run on the server in a CPU environment with OpenCV, NumPy and Pillow.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from colmap_model import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat

TRAINING_COMMIT = '747ddfef646edf3ea628f2bd13b7bedce7c5fe47'
INIT_BLOB = '823792edc9d815a77a6ebb693861c8b1e8385712'
TRAINING_URL = 'https://github.com/SJoJoK/3DGStream'
DATA_URL = 'https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_steak.zip'
FIXTURE = 'test/flame_steak_suite/frame000000'


def digest(path, algorithm='sha256', git_blob=False):
    result = hashlib.new(algorithm)
    if git_blob:
        result.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def make_reference(args):
    import cv2
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(2)
    root = args.training_repository.resolve()

    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()

    if git('rev-parse', 'HEAD') != TRAINING_COMMIT:
        raise ValueError('Use the pinned official training fixture commit')

    def verified(relative):
        path = root / relative
        expected = git('ls-tree', 'HEAD', '--', relative).split()[2]
        if digest(path, 'sha1', True) != expected:
            raise ValueError(f'Training fixture changed: {relative}')
        return path

    init = args.init_ply.resolve()
    if digest(init, 'sha1', True) != INIT_BLOB:
        raise ValueError('Initial Gaussians do not match this official camera coordinate frame')
    model = root / FIXTURE
    files = {}
    for part in ['sparse/0/cameras.bin', 'sparse/0/images.bin',
                 'distorted/sparse/0/cameras.bin', 'distorted/sparse/0/images.bin',
                 'images/cam00.png']:
        files[part] = verified(FIXTURE + '/' + part)
    cameras = read_intrinsics_binary(files['sparse/0/cameras.bin'])
    images = read_extrinsics_binary(files['sparse/0/images.bin'])
    image = next(item for item in images.values() if item.name == 'cam00.png')
    camera = cameras[image.camera_id]
    raw_cameras = read_intrinsics_binary(files['distorted/sparse/0/cameras.bin'])
    raw_images = read_extrinsics_binary(files['distorted/sparse/0/images.bin'])
    raw_image = next(item for item in raw_images.values() if item.name == 'cam00.png')
    raw_camera = raw_cameras[raw_image.camera_id]
    if camera.model != 'PINHOLE' or raw_camera.model != 'OPENCV':
        raise ValueError('This fixture must contain undistorted PINHOLE and original OPENCV cameras')
    if not np.allclose(image.qvec, raw_image.qvec) or not np.allclose(image.tvec, raw_image.tvec):
        raise ValueError('Distorted and undistorted extrinsics disagree')
    fx, fy, cx, cy = camera.params
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], dtype=np.float64)
    raw_fx, raw_fy, raw_cx, raw_cy, *distortion = raw_camera.params
    raw_k = np.array([[raw_fx, 0, raw_cx], [0, raw_fy, raw_cy], [0, 0, 1.]])
    rotation = qvec2rotmat(image.qvec)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = image.tvec
    if args.scale < 1 or camera.width % args.scale or camera.height % args.scale:
        raise ValueError('Use an integer scale dividing the undistorted dimensions')
    width, height = camera.width // args.scale, camera.height // args.scale
    scaled = k.copy()
    scaled[:2] /= args.scale
    source = dict(training_repository=TRAINING_URL, training_commit=TRAINING_COMMIT,
                  camera_name='cam00.png', initial_scene_git_blob=INIT_BLOB,
                  video_source=DATA_URL, video_sha256=digest(args.video),
                  fixture_sha256={name: digest(path) for name, path in files.items()},
                  data_license='CC-BY-NC-4.0')
    config = dict(schema='q3dg_camera_v1', convention='opencv', camera_name='cam00',
                  width=width, height=height, intrinsics=scaled.tolist(),
                  world_to_camera=matrix.tolist(), scene_init_sha256=digest(init), source=source,
                  time_alignment=dict(initial_scene_video_frame=0, ntc_zero_video_frame=1, fps=30))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'cam00.json').write_text(json.dumps(config, indent=2) + '\n')
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError('Cannot decode source cam00 video')
    fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if abs(fps - 30) > 1e-6 or count != 300:
        raise ValueError(f'Expected 300 frames at 30 Hz, found {count} at {fps}')
    okay, first = capture.read()
    if not okay or first.shape[:2] != (raw_camera.height, raw_camera.width):
        raise ValueError('Source video dimensions disagree with the original camera')
    map_x, map_y = cv2.initUndistortRectifyMap(raw_k, np.asarray(distortion), np.eye(3), k,
                                            (camera.width, camera.height), cv2.CV_32FC1)

    def undistort(frame):
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    reference = cv2.imread(str(files['images/cam00.png']))
    original = first if reference.shape == first.shape else undistort(first)
    if reference.shape != original.shape:
        raise ValueError('Training frame dimensions cannot be matched to the video')
    mse = float(np.mean((reference.astype(np.float64) - original.astype(np.float64)) ** 2))
    psnr = 10 * math.log10(255 ** 2 / mse) if mse else None
    proof = dict(initial_scene_identical=True, camera_id=image.camera_id, image_name=image.name,
                 video_frames=count, video_fps=fps, frame_zero_identical=mse == 0,
                 frame_zero_psnr_db=psnr, frame_zero_max_difference=int(np.abs(
                     reference.astype(np.int16) - original.astype(np.int16)).max()),
                 time_pixel_alignment_verified=mse == 0 or psnr >= 40,
                 decoded_reference_frames=0, opencv_version=cv2.__version__)
    if args.extract:
        if not proof['time_pixel_alignment_verified']:
            raise ValueError('Video frame zero does not match the official training frame; review alignment')
        frames = args.out / 'frames'
        frames.mkdir(exist_ok=True)
        for index in range(1, 300):
            okay, frame = capture.read()
            if not okay:
                raise ValueError(f'Source video ended before frame {index}')
            image = cv2.resize(undistort(frame), (width, height), interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(frames / f'frame_{index:06}.png'), image,
                               [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise IOError('Failed writing reference image')
        proof['decoded_reference_frames'] = 299
    capture.release()
    (args.out / 'REFERENCE.json').write_text(json.dumps(dict(config=config, validation=proof), indent=2) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-repository', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--init-ply', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scale', type=int, default=2)
    p.add_argument('--extract', action='store_true', help='Generate aligned video frames1..299')
    make_reference(p.parse_args())
