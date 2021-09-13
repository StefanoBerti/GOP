import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud
from open3d.visualization import Visualizer
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from facenet_pytorch import MTCNN


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
        m = np.array(rgb_image)
        width = m.shape[1]
        height = m.shape[0]
        camera = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsics.fx,
                                                   intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        return pcd

    def stop(self):
        self.pipeline.stop()


class Loop:
    def __init__(self, input):
        self.input = input

        self.vis = Visualizer()
        self.vis.create_window('Pose Estimation')

        self.full_pcd = PointCloud()
        self.render_setup = False
        self.face_detector = MTCNN(keep_all=True, device="cuda:0")

    def run(self):
        color_image, depth_image = self.input.read()

        # Look for faces inside it
        boxes, confidences = self.face_detector.detect(Image.fromarray(color_image))
        # # If at least one face is recognized
        if boxes is not None:
            for i, elem in enumerate(boxes):
                # Get x_min, y_min, x_max, y_max, conf
                x_min = elem[0]
                y_min = elem[1]
                x_max = elem[2]
                y_max = elem[3]
                import copy
                face_img = copy.deepcopy(color_image[int(y_min):int(y_max), int(x_min):int(x_max)])
                face_depth = copy.deepcopy(depth_image[int(y_min):int(y_max), int(x_min):int(x_max)])

                color_image.fill(0)
                depth_image.fill(0)

                color_image[int(y_min):int(y_max), int(x_min):int(x_max)] = face_img
                depth_image[int(y_min):int(y_max), int(x_min):int(x_max)] = face_depth

        cv2.imshow('Segmented', depth_image)
        cv2.imshow('Color', color_image)

        self.full_pcd.clear()

        self.full_pcd += input.pointcloud(color_image,
                                          depth_image)  # input.pointcloud(color_image, depth_image) #object_pcd
        # model_pcd = icp.model.pc

        # input.project(color_image, [icp.model for icp in icps])
        self.render([self.full_pcd])  #  + [icp.model.pc for icp in icps]) #, model_pcd])

    def render(self, pcds):

        points = np.array(pcds[0].points)
        mean_x = sum(points[:, 0]) / len(points)
        mean_y = sum(points[:, 1]) / len(points)
        mean_z = sum(points[:, 2]) / len(points)

        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

        pcds[0].points = o3d.utility.Vector3dVector(np.append(np.array(pcds[0].points), np.array([mean_x, mean_y, mean_z])[None, ...], axis=0))

        print("x: ", mean_x, ", y: ", mean_y, ", z:", mean_z)

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
    input = RealSense()

    Loop(input).start()


