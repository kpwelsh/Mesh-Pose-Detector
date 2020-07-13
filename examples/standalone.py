import open3d as o3d
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import morphology
from scipy.ndimage import correlate
import os
from PoseDetector import PoseDetector
from PoseDetector.DetectedObject import DetectedObject
import json


if __name__ == '__main__':
    K = np.array([
		918.91259765625,
		0.0,
		0.0,
		0.0,
		918.51983642578125,
		0.0,
		960.29638671875,
		553.99737548828125,
		1.0
	]).reshape((3,3)).T

    with open('standalone.json') as fin:
        detector = PoseDetector(**json.load(fin))

    rgb = np.array(o3d.io.read_image('color.jpg'))
    depth = np.array(o3d.io.read_image('depth.png'))
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    detected_objects = detector.detect(rgb, depth, K)
    for obj in detected_objects:
        R, o, mesh = obj.R, obj.o, obj.Mesh
        vis.add_geometry(o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector((((mesh.vertices  + o) @ R.T))),
            o3d.utility.Vector3iVector(mesh.faces)
        ))

    mask = depth > 0
    depth_values = depth[mask]
    y, x = np.where(mask)
    img_coords = np.column_stack((x,y) + (np.ones_like(depth_values),))
    camera_coords = np.linalg.inv(K) @ (img_coords.T * depth_values) / 1000
    vis.add_geometry(o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(camera_coords.T)
    ))
    ctrl = vis.get_view_control()
    ctrl.set_front(np.array((0,0,-1)))
    ctrl.set_up((np.array((0,-1,0))))
    old_params = ctrl.convert_to_pinhole_camera_parameters().intrinsic
    params = o3d.camera.PinholeCameraParameters()
    params.extrinsic = np.eye(4)
    params.intrinsic = old_params
    ctrl.convert_from_pinhole_camera_parameters(
        params
    )
    vis.run()