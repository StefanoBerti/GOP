# examples/Python/Basic/visualization.py

import numpy as np
import open3d as o3d
from open3d.visualization import Visualizer

if __name__ == "__main__":
    vis = Visualizer()
    vis.create_window('Pose Estimation')

    print("Let's draw some primitives")
    i=0
    while True:

        # Clear
        vis.clear_geometries()

        # Create box
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0,
                                                        height=1.0,
                                                        depth=1.0)

        # Rotate box
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh_box.rotate(mesh.get_rotation_matrix_from_axis_angle((np.pi / 2, i, np.pi / 4)))
        mesh_box.translate([i, i, i])
        i += 0.01

        # Add geometry
        mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
        vis.add_geometry(mesh_box)
        vis.update_geometry(mesh_box)
        vis.poll_events()
        vis.update_renderer()
    # mesh_box.compute_vertex_normals()


    # mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    # mesh_sphere.compute_vertex_normals()
    # mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    # mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3,
    #                                                           height=4.0)
    # mesh_cylinder.compute_vertex_normals()
    # mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])


    mesh_box.rotate(mesh.get_rotation_matrix_from_axis_angle((np.pi / 2, 0, np.pi / 4)))

    print("We draw a few primitives using collection.")
    o3d.visualization.draw_geometries(
        [mesh_box])  # , mesh_sphere, mesh_cylinder, mesh_frame])

    for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        mesh_box.rotate(mesh.get_rotation_matrix_from_axis_angle((np.pi / 2, i, np.pi / 4)))

    # print("We draw a few primitives using + operator of mesh.")
    # o3d.visualization.draw_geometries(
    #     [mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])
    #
    # print("Let's draw a cubic using o3d.geometry.LineSet.")
    # points = [
    #     [0, 0, 0],
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [1, 1, 0],
    #     [0, 0, 1],
    #     [1, 0, 1],
    #     [0, 1, 1],
    #     [1, 1, 1],
    # ]
    # lines = [
    #     [0, 1],
    #     [0, 2],
    #     [1, 3],
    #     [2, 3],
    #     [4, 5],
    #     [4, 6],
    #     [5, 7],
    #     [6, 7],
    #     [0, 4],
    #     [1, 5],
    #     [2, 6],
    #     [3, 7],
    # ]
    # colors = [[1, 0, 0] for i in range(len(lines))]
    # line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(points),
    #     lines=o3d.utility.Vector2iVector(lines),
    # )
    # line_set.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([line_set])
    #
    # print("Let's draw a textured triangle mesh from obj file.")
    # textured_mesh = o3d.io.read_triangle_mesh("../../TestData/crate/crate.obj")
    # textured_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([textured_mesh])