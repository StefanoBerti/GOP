import copy
import pickle
import threading
from pathlib import Path, WindowsPath
import random

import PIL
import cv2
import open3d as o3d
import pyrealsense2 as rs
import numpy as np
import numpy.ma as ma
import scipy.io as scio
# import yarp
import torch
from PIL import Image
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries
from torchvision.transforms import transforms

from lib.cad import Model


class RealSense:
    def __init__(self):
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)

        self.align = rs.align(rs.stream.color)

    def intrinsics(self):
        return self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def pointcloud(self, rgb_image, depth_image):
        depth_image = o3d.geometry.Image(depth_image)
        rgb_image = o3d.geometry.Image(rgb_image)
        rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(rgb_image, depth_image,
                                                                    convert_rgb_to_intensity=False,
                                                                    depth_scale=1500)

        intrinsics = self.intrinsics()
        camera = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx,
                                                   intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        return pcd

    def stop(self):
        self.pipeline.stop()


# class iCubGazebo:
#
#
#     def __init__(self, rgb_port="/icubSim/cam/left/rgbImage:o", depth_port='/icubSim/cam/left/depthImage:o'):
#         yarp.Network.init()
#
#         # Create a port and connect it to the iCub simulator virtual camera
#         self.rgb_port, self.depth_port = yarp.Port(), yarp.Port()
#         self.rgb_port.open("/rgb-port")
#         self.depth_port.open("/depth-port")
#         yarp.Network.connect(rgb_port, "/rgb-port")
#         yarp.Network.connect(depth_port, "/depth-port")
#
#         self.rgb_array = np.zeros((240, 320, 3), dtype=np.uint8)
#         self.rgb_image = yarp.ImageRgb()
#         self.rgb_image.resize(320, 240)
#         self.rgb_image.setExternal(self.rgb_array, self.rgb_array.shape[1], self.rgb_array.shape[0])
#
#         self.depth_array = np.zeros((240, 320), dtype=np.float32)
#         self.depth_image = yarp.ImageFloat()
#         self.depth_image.resize(320, 240)
#         self.depth_image.setExternal(self.depth_array, self.depth_array.shape[1], self.depth_array.shape[0])
#
#     def read(self):
#         self.rgb_port.read(self.rgb_image)
#         self.depth_port.read(self.depth_image)
#
#         return self.rgb_array, self.depth_array
#
#
# if __name__ == '__main__':
#     input = iCubGazebo()
#     while True:
#         rgb, depth = input.read()
#         cv2.imshow('RGB', rgb)
#
#         key = cv2.waitKey(1) & 0xFF
#         # if the `q` key was pressed, break from the loop
#         if key == ord("q"):
#             break
def get_bbox(label):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    img_width = 480
    img_length = 640

    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax



class YCBVideo:
    def __init__(self, root, train=True, noise=True):
        self.root = Path(root)
        self.config = 'dataset_config'
        self.dataset = 'YCB_Video_Dataset'

        self.noise = noise
        self.noise_trans = 0.03
        self.num_pt = 1000
        self.front_num = 2
        self.minimum_num_pt = 50
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

        if train:
            input_file = open(self.root / self.config / 'train_data_list.txt')
        else:
            input_file = open(self.root / self.config / 'test_data_list.txt')

        self.frames = [line.strip() for line in input_file.readlines()]
        self.real_frames = [frame for frame in self.frames if frame.split('/')[0] == 'data']
        self.syn_frames = [frame for frame in self.frames if frame.split('/')[0] == 'data_syn']
        input_file.close()

        with open(self.root / self.config / 'classes.txt') as file:
            models_names = [line.rstrip() for line in file.readlines()]

        self.models = {}
        for idx, model_name in enumerate(models_names, 1):
            model_path = self.root / self.dataset / 'models' / model_name / 'points.xyz'
            with open(model_path) as file:
                model = \
                    Model(pointcloud=np.array([line.rstrip().split() for line in file.readlines()], dtype=float),
                          name=model_name,
                          id=idx)
                self.models[idx] = model

        self.intrinsics_1 = {
            'cx': 312.9869, 'cy': 241.3109,
            'fx': 1066.778, 'fy': 1067.487,
        }

        self.intrinsics_2 = {
            'cx': 323.7872, 'cy': 279.6921,
            'fx': 1077.836, 'fy': 1078.189,
        }

    def __getitem__(self, idx):
        frame = self.frames[idx]
        img = Image.open(self.root / self.dataset / f'{frame}-color.png')
        depth = np.array(Image.open(self.root / self.dataset / f'{frame}-depth.png'), dtype=np.float32)
        label = np.array(Image.open(self.root / self.dataset / f'{frame}-label.png'))
        meta = scio.loadmat(str(self.root / self.dataset / f'{frame}-meta'))

        is_syn = len(self.frames[idx].split('/')) == 2
        if not is_syn and int(self.frames[idx].split('/')[2]) >= 60:
            intrinsics = self.intrinsics_1
        else:
            intrinsics = self.intrinsics_2

        objs_cls = meta['cls_indexes'].flatten().astype(np.int32)

        targets = []
        for i, c in enumerate(objs_cls):
            t = np.vstack([meta['poses'][:, :, i], np.eye(4)[3, :]])

            target = copy.deepcopy(self.models[c]).transform(t)
            targets.append(target)

        return img, depth, targets, intrinsics


    def read(self):
        rgb_frame = np.array(PIL.Image.open(str(Path(self.path) / f'{self.head + 1:06}-color.png')))
        depth_frame = np.array(PIL.Image.open(str(Path(self.path) / f'{self.head + 1:06}-depth.png')))
        meta = scio.loadmat(str(Path(self.path) / f'{self.head + 1:06}-meta'))

        targets = []
        for i, idx in enumerate(meta['cls_indexes'].flatten().astype(np.int32)):
            targets.append(Model(self.models_path / self.models_names[idx - 1] / 'points.xyz', idx))
            targets[i].transform(np.vstack([meta['poses'][:, :, i], np.eye(4)[3, :]]))

        self.head += 1
        return rgb_frame, depth_frame.astype(np.float32), targets

    def stop(self):
        pass

    def pointcloud(self, rgb_image, depth_image):
        depth_image = o3d.geometry.Image(depth_image)
        rgb_image = o3d.geometry.Image(rgb_image)
        rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(rgb_image, depth_image,
                                                                    convert_rgb_to_intensity=False,
                                                                    depth_scale=10000.0)
        # Parameters from ycb video from 00 to 59
        cam_cx = 312.9869
        cam_cy = 241.3109
        cam_fx = 1066.778
        cam_fy = 1067.487
        cam_scale = 1000.0

        camera = o3d.camera.PinholeCameraIntrinsic(640, 480, cam_fx,
                                                   cam_fy, cam_cx, cam_cy)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)
        #
        # pt2 = np.array(depth_image) / cam_scale
        # pt0 = (ymap - cam_cx) * pt2 / cam_fx
        # pt1 = (xmap - cam_cy) * pt2 / cam_fy
        # cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        #
        # pt0 = pt2.flatten()
        # pt1 = pt2.flatten()
        # pt2 = pt2.flatten()
        #
        # cloud = []
        # for i in range(pt0.shape[0]):
        #     cloud.append((pt0[i], pt1[i], pt2[i]))
        #
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(np.array(cloud))

        # pcd.transform([[1, 0, 0, 0],
        #                [0, -1, 0, 0],
        #                [0, 0, -1, 0],
        #                [0, 0, 0, 1]])

        return pcd

    def project(self, rgb, models):

        k = np.eye(3)
        k[0, :] = np.array([1066.778, 0, 312.9869])
        k[1, 1:] = np.array([1067.487, 241.3109])

        for model in models:
            points = np.array(model.points) * 10000.0
            uv = k @ points.T
            uv = uv[0:2] / uv[2, :]

            uv = np.round(uv, 0).astype(int)

            uv[0, :] = np.clip(uv[0, :], 0, 640)
            uv[1, :] = np.clip(uv[1, :], 0, 480)

            rgb[uv[1, :], uv[0, :], :] = np.tile((np.array(model.color) * 255).astype(int), (uv.shape[1], 1))

        cv2.imshow('projection', rgb)


class VideoCaptureAsync:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)

        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()
        self.release()

    def release(self):
        self.cap.release()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


class DepthVideo:
    def __init__(self):
        self.frame_list = []
        self.head = 0

    def write(self, frame):
        self.frame_list.append(frame)

    def read(self):
        ret = self.frame_list[self.head]
        self.head += 1
        return ret

    def release(self):
        pass
