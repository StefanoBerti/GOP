import open3d as o3d
import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.visualization import Visualizer
from torchvision import models
import cv2
import numpy as np

from model.segmentation1 import Segmentation
from lib.cad import Model, ICP
from lib.input import RealSense, YCBVideo
from model.segmentation2 import SegNet
from sklearn.neighbors import NearestNeighbors


def loss(prediction, target):
    nbrs = NearestNeighbors(n_neighbors=1).fit(target)
    distances, indices = nbrs.kneighbors(prediction, n_neighbors=1)
    return np.mean(distances)


class Loop:
    def __init__(self, input, segmentation, icps):
        self.input = input
        self.segmentation = segmentation
        self.icps = icps

        self.vis = Visualizer()
        self.vis.create_window('Pose Estimation')

        self.full_pcd = PointCloud()
        self.render_setup = False

    def run(self):
        color_image, depth_image = self.input.read()
        segmented, categories = segmentation(color_image)

        for i, icp in enumerate(icps):
            model = icp.model

            segmented_depth = segmentation.segment_depth(categories, depth_image, model.id)

            cv2.imshow('Segmented', segmented)
            cv2.imshow('Color', color_image)

            object_pcd = input.pointcloud(color_image, segmented_depth)

            object_pcd = object_pcd.voxel_down_sample(voxel_size=0.001)
            object_pcd, ind = object_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            # object_pcd.estimate_normals( # Used for point to plane icp
            #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            icp(object_pcd)

            icp.model.pc.get_max_bound()

        self.full_pcd.clear()

        self.full_pcd += input.pointcloud(color_image,
                                          depth_image)  # input.pointcloud(color_image, depth_image) #object_pcd
        # model_pcd = icp.model.pc

        # input.project(color_image, [icp.model for icp in icps])
        self.render([self.full_pcd] + [icp.model.pc for icp in icps]) #, model_pcd])

    def render(self, pcds):
        if not self.render_setup:
            for pc in pcds:
                self.vis.add_geometry(pc)

            self.render_setup = True

        for pc in pcds:
            self.vis.update_geometry(pc)

        self.vis.poll_events()
        self.vis.update_renderer()

    def start(self):
        try:
            while True:
                self.run()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.vis.close()
            self.input.stop()


if __name__ == '__main__':
    icps = []
    objects = [('021_bleach_cleanser', 5, [1, 0, 0])]

    for obj in objects:
        model_path = f"C:/Users/arosasco/Desktop/ycb/YCB_Video_Dataset/models/{obj[0]}/points.xyz"
        with open(model_path) as file:
            model = \
                Model(pointcloud=np.array([line.rstrip().split() for line in file.readlines()], dtype=float),
                      name=obj[0], id=obj[1], color=obj[2])
        # model.transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
        icps.append(ICP(model, 10))

    input = RealSense()
    segmentation = Segmentation(models.segmentation.fcn_resnet101(pretrained=True).eval(), device='cuda')

    Loop(input, segmentation, icps).start()

