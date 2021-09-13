import open3d as o3d
from open3d.visualization import Visualizer
from translation import Translation
from realsense import RealSense
from rotation import Rotation
from scipy.spatial.transform import Rotation as R
import cv2
from utils import params


class VisualizeHead:
    def __init__(self):
        self.cam = RealSense()
        self.vis = Visualizer()
        self.translation = Translation(self.cam)
        self.rotation = Rotation(self.cam)
        self.i = 0

    def loop(self):
        self.vis.create_window()
        rendered = False

        while True:
            # Create box
            mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.5,
                                                            height=0.5,
                                                            depth=0.5)
            mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

            mesh_coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
            mesh_cone = o3d.geometry.TriangleMesh.create_cone(radius=0.5, height=params["cone_height"],
                                                              resolution=20, split=1)
            mesh_cone.paint_uniform_color([0.1, 0.9, 0.1])

            # Get rotation and translation
            color_image, depth_image = self.cam.read()
            cv2.imshow("frame", color_image)  # TODO DECOMMENT TO SEE WEBCAM

            objects = self.translation.get_xyz(color_image, depth_image)
            if objects["face"] is None:
                continue
            xyz = objects["face"]
            depth_factor = 5
            xyz = (xyz[0], xyz[1], xyz[2] * depth_factor)

            rpy = self.rotation.get_rpy(color_image, depth_image)
            if rpy is None:
                continue
            rpy = (-rpy[0], -rpy[1], -rpy[2])

            # Create orientation transformation matrix
            r = R.from_euler('yxz', rpy, degrees=True).as_matrix()

            # POSITION BOX
            mesh_box.translate((-0.25, -0.25, -0.5))  # move box to have only positive z values
            mesh_box.rotate(r)  # rotate accordingly to head orientation
            # mesh_box.translate((0, 0, -0.5), relative=True)  # move cylinder to have only positive z values
            mesh_box.translate(xyz, relative=True)

            # POSITION CONE
            # mesh_cone.translate((0, 0, params["cone_height"]))  # move cylinder to have only positive z values
            # flip 180 degrees
            flip = R.from_euler('yxz', (0, 180, 0), degrees=True).as_matrix()
            mesh_cone.rotate(flip)
            # translate to have only positive z values
            mesh_cone.translate((0, 0, params["cone_height"]))
            # rotate accordingly to head position
            # mesh_cone.translate((0.15, 0, 0))
            mesh_cone.rotate(r, center=(0, 0, 0))  # rotate accordingly to head orientation
            mesh_cone.translate(xyz, relative=True)  # translate accordingly to head position

            # POSITION OBJECTS
            objs = objects["objects"]
            objects_to_add = []
            for obj in objs.keys():
                o = o3d.geometry.TriangleMesh.create_box(width=0.5,
                                                         height=0.5,
                                                         depth=0.5)
                o.paint_uniform_color([0.1, 0.1, 0.9])
                where = (objs[obj][0], objs[obj][1], objs[obj][2] * depth_factor)
                o.translate(where)
                objects_to_add.append(o)

            # o3d.cpu.pybind.t.geometry.TSDFVoxelGrid TODO ADD INTERSECTION

            self.vis.clear_geometries()
            if rendered:
                self.vis.update_geometry(mesh_box)
                self.vis.update_geometry(mesh_coord)
                self.vis.update_geometry(mesh_cone)
            else:
                # rendered = True  # TODO activate when absolute rotation
                for o in objects_to_add:
                    self.vis.add_geometry(o)
                self.vis.add_geometry(mesh_box)
                self.vis.add_geometry(mesh_coord)
                self.vis.add_geometry(mesh_cone)

            # self.vis.update_geometry(mesh_box)
            self.vis.poll_events()
            self.vis.update_renderer()
            self.i += 1


if __name__ == "__main__":
    vis = VisualizeHead()
    vis.loop()
