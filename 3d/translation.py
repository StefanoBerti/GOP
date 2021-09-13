import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud
from open3d.visualization import Visualizer
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from utils import params
import copy
from segmentation import Segmentation


class Translation:
    def __init__(self, cam, show_debug=False):
        self.cam = cam
        self.full_pcd = PointCloud()
        self.render_setup = False
        self.face_detector = MTCNN(keep_all=True, device="cuda:0")
        self.debug = show_debug
        if self.debug:
            self.vis = Visualizer()
        # Segmentation
        self.segmentation = Segmentation()

    def get_mean_depth(self, color_image, depth_image):
        pc = self.cam.pointcloud(color_image, depth_image)

        pc = pc.voxel_down_sample(voxel_size=0.001)
        pc, ind = pc.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)

        self.full_pcd += pc
        aux_pcd = PointCloud()
        aux_pcd += pc

        points = np.array(aux_pcd.points)
        if len(points) == 0:
            return None
        mean_x = sum(points[:, 0]) / len(points)
        mean_y = sum(points[:, 1]) / len(points)
        mean_z = sum(points[:, 2]) / len(points)

        return mean_x, mean_y, mean_z

    def get_xyz(self, color_image, depth_image):
        """""Return a dictionary with face position and found objects position"""
        _, cat = self.segmentation.segment(color_image)

        objects = {"face": None,
                   "objects": {}
                   }
        frame = copy.deepcopy(color_image)
        depth = copy.deepcopy(depth_image)

        # Look for faces inside it
        boxes, confidences = self.face_detector.detect(Image.fromarray(color_image))

        if self.debug:
            self.vis.create_window()

        self.full_pcd.clear()

        ##################
        # GET FACE DEPTH #
        ##################
        if boxes is not None:
            for i, elem in enumerate(boxes):
                if confidences[i] < params["face_min_conf"]:
                    continue
                # Get x_min, y_min, x_max, y_max, conf
                x_min = elem[0]
                y_min = elem[1]
                x_max = elem[2]
                y_max = elem[3]

                face_img = copy.deepcopy(color_image[int(y_min):int(y_max), int(x_min):int(x_max)])
                face_depth = copy.deepcopy(depth_image[int(y_min):int(y_max), int(x_min):int(x_max)])

                color_image.fill(0)
                depth_image.fill(0)

                color_image[int(y_min):int(y_max), int(x_min):int(x_max)] = face_img
                depth_image[int(y_min):int(y_max), int(x_min):int(x_max)] = face_depth

                color_image[cat != self.segmentation.classes.index("person")] = 0
                depth_image[cat != self.segmentation.classes.index("person")] = 0

        res = self.get_mean_depth(color_image, depth_image)
        if res is not None:
            face_x, face_y, face_z = res
            objects["face"] = (face_x, face_y, face_z)

        #####################
        # GET OBJECTS DEPTH #
        #####################

        found = list(np.unique(cat))

        for elem in found:
            if elem in [self.segmentation.classes.index("background"), self.segmentation.classes.index("person")]:
                continue
            name = self.segmentation.classes[elem]

            img_good = copy.deepcopy(frame)
            depth_good = copy.deepcopy(depth)

            img_good[cat != elem] = 0
            depth_good[cat != elem] = 0

            ret = self.get_mean_depth(img_good, depth_good)
            if ret is not None:
                obj_x, obj_y, obj_z = ret
                objects["objects"][name] = (obj_x, obj_y, obj_z)

        # debug
        if self.debug:
            for obj in objects["objects"].keys():
                x, y, z = objects["objects"][obj]
                self.full_pcd.points = o3d.utility.Vector3dVector(
                    np.append(np.array(self.full_pcd.points), np.array([x, y, z])[None, ...], axis=0))

            self.render([self.full_pcd], color_image, depth_image, frame)

        return objects

    def render(self, pcds, c, d, frame):

        cv2.imshow('Depth', d)
        cv2.imshow('Color', c)

        if not self.render_setup:
            for pc in pcds:
                self.vis.add_geometry(pc)
            self.render_setup = True

        for pc in pcds:
            self.vis.update_geometry(pc)

        self.vis.poll_events()
        self.vis.update_renderer()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            exit()


if __name__ == "__main__":
    from realsense import RealSense
    ca = RealSense()
    vis = Translation(ca, show_debug=True)
    while True:
        co, de = ca.read()
        vis.get_xyz(co, de)
