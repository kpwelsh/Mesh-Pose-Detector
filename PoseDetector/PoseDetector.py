import open3d as o3d
import torch, torchvision
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
import trimesh
from trimesh.proximity import ProximityQuery
import numpy as np
import cv2
from PoseDetector.HuIndex import HuIndex
from sklearn.cluster import DBSCAN
from statistics import mode
from scipy.ndimage import morphology
from collections import Counter
from PoseDetector.HoughTransform import HoughTransform
from PoseDetector.BoundingBox import BoundingBox
import functools

from PoseDetector.DetectedObject import DetectedObject
from PoseDetector.MeshProfile import MeshProfile
import cupy as cp



valid_positions_device = cp.RawKernel(
r'''
extern "C" __global__
void valid_positions_device(
        float* K, float* R, float* vertices, int* depth, int* d_shape, 
        float grid_size, float* mask, int* m_shape, float* lower, int n)
    
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int m_size = m_shape[0]*m_shape[1]*m_shape[2];
    if (id >= (m_size * n)) {
        return;
    }
    int i = id / m_size;
    id = id % m_size;
    int idx = id / (m_shape[2] * m_shape[1]);
    int idy = id / m_shape[2];
    idy = idy - (idy / m_shape[1]) * m_shape[1];
    int idz = id - idx * (m_shape[2] * m_shape[1]) - idy * m_shape[2];
    float pos_x = vertices[i] + lower[0] + (0.5 + idx) * grid_size;
    float pos_y = vertices[i + n] + lower[1] + (0.5 + idy) * grid_size;
    float pos_z = vertices[i + 2 * n] + lower[2] + (0.5 + idz) * grid_size; 
    float x = R[0] * pos_x + R[3] * pos_y + R[6] * pos_z; 
    float y = R[1] * pos_x + R[4] * pos_y + R[7] * pos_z; 
    float z = R[2] * pos_x + R[5] * pos_y + R[8] * pos_z;

    float img_z = K[2] * x + K[5] * y + K[8] * z;
    int img_x = (K[0] * x + K[3] * y + K[6] * z) / img_z;
    int img_y = (K[1] * x + K[4] * y + K[7] * z) / img_z;

    if (img_y >= 0 && img_y < d_shape[0]
        && img_x >= 0 && img_x < d_shape[1]
        && depth[img_y + img_x * d_shape[0]] - z * 1000 > 20){
        
        mask[id] = 0;
    }
}
''', 'valid_positions_device')

# @cuda.jit
# def valid_positions_device(K, R, vertices, depth, d_shape, grid_size, mask, m_shape, lower):
#     id = cuda.blockIdx.x * 512 + cuda.threadIdx.x
#     if id >= mask.size:
#         return
#     idx = id // (m_shape[2] * m_shape[1])
#     idy = id // m_shape[2]
#     idy = idy - (idy // m_shape[1]) * m_shape[1]
#     idz = id - idx * (m_shape[2] * m_shape[1]) - idy * m_shape[2]
#     pos_x = lower[0] + (0.5 + idx) * grid_size
#     pos_y = lower[1] + (0.5 + idy) * grid_size
#     pos_z = lower[2] + (0.5 + idz) * grid_size

#     for i in range(len(vertices)):
#         v_x, v_y, v_z = vertices[i,0], vertices[i,1], vertices[i,2]
#         v_x += pos_x
#         v_y += pos_y
#         v_z += pos_z
#         x = R[0,0] * v_x + R[0,1] * v_y + R[0,2] * v_z
#         y = R[1,0] * v_x + R[1,1] * v_y + R[1,2] * v_z
#         z = R[2,0] * v_x + R[2,1] * v_y + R[2,2] * v_z

#         img_z = K[2,0] * x + K[2,1] * y + K[2,2] * z
#         img_x = int((K[0,0] * x + K[0,1] * y + K[0,2] * z) // img_z)
#         img_y = int((K[1,0] * x + K[1,1] * y + K[1,2] * z) // img_z)
#         if y >= 0 and y < d_shape[0] \
#             and x >= 0 and x < d_shape[1] \
#             and depth[img_y,img_x] - z*1000 > 20:
#             mask[idx, idy, idz] = 0

def valid_positions(R, vertices, depth, K, mask, lower, grid_size):
    valid_positions_device(((mask.size * len(vertices)) // 512 + 1,), (512,), (
        cp.asarray(K.flatten()),
        cp.asarray(R.flatten()),
        cp.asarray(vertices.flatten()),
        cp.asarray(depth.flatten()),
        cp.array(depth.shape, cp.int), 
        cp.float32(grid_size), 
        cp.asarray(mask.flatten(), cp.int),
        cp.array(mask.shape, cp.int), 
        cp.asarray(lower),
        cp.int(len(vertices))
    ))

class PoseDetector:
    def __init__(self, **config):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            config.get("model_zoo_config","COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        ))
        cfg.MODEL.WEIGHTS = config.get('model.weights')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.get('model.score_threshold',0.3)
        
        classes = [m['name'] for m in config.get('meshes')]

        self.MetaData = MetadataCatalog.get(config.get("model.metadata_catalog")).set(thing_classes = classes)
        cfg.DATASETS.TEST = (config.get("model.metadata_catalog"),)
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.get('model.roi_heads.batch_size_per_image')
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.get('model.roi_heads.num_classes')
        self.Predictor = DefaultPredictor(cfg)

        self.MeshProfiles = [MeshProfile(**v) for v in config.get('meshes')]

    def get_masks(self, rgb):
        instances = self.Predictor(rgb)['instances'].to('cpu')
        return instances.pred_masks
    def get_boxes(self, rgb):
        instances = self.Predictor(rgb)['instances'].to('cpu')
        boxes = instances.pred_boxes
        return list(boxes)

    def detect(self, rgb, depth, K):
        instances = self.Predictor(rgb)['instances'].to('cpu')
        detected_objects = []
        for i in range(len(instances)):
            cloud = self.get_cloud(instances[i], depth, K)
            if len(cloud) == 0:
                continue
            cloud = self.select_largest_cluster(cloud)

            mask = instances[i].pred_masks[0]
            hu = cv2.HuMoments(cv2.moments(np.float32(mask)))
            profile = self.MeshProfiles[instances[i].pred_classes[0]]
            (R, o), cloud_orientation = self.get_best_pose(profile, hu, depth, K, cloud)
            detected_objects.append(DetectedObject(R, o, instances[i].pred_boxes, mask, profile.MeshFile, profile.Name, profile.Trimesh))
        return detected_objects

    def get_best_pose(self, profile, hu, depth, K, cloud):
        cloud_sample = cloud[np.random.choice(range(len(cloud)), min(200, len(cloud)))]
        _, _, cloud_orientation = np.linalg.svd(cloud_sample[:,:2] - np.mean(cloud_sample[:,:2], axis = 0))
        if cloud_orientation[0,0] * cloud_orientation[1,1] - cloud_orientation[0,1] * cloud_orientation[1,0] < 0:
            cloud_orientation[1,:] *= -1
        cloud_mean = np.mean(cloud, axis = 0)
        best_config = (np.inf, None)
        lower = np.min(cloud, axis = 0)
        upper = np.max(cloud, axis = 0)
        h = profile.Trimesh.volume / ((upper[0] - lower[0]) * (upper[1] - lower[1]))
        upper[2] += 0.2
        bbox = BoundingBox(lower, upper)
        for config in profile.HuIndex.query_moment_index(hu):
            for flip in [np.eye(3), np.array(((-1,0,0),(0,-1,0),(0,0,1)))]:
                for zflip in [np.eye(3), np.array(((-1,0,0), (0,1,0), (0,0,-1)))]:
                    R, offset, db_orientation = config
                    R2 = db_orientation.T @ cloud_orientation
                    R3 = np.eye(3)
                    R3[:2,:2] = R2
                    R = zflip @ flip @ R3.T @ R

                    import time
                    s = time.time()
                    o, val = profile.HoughTransform.vote(
                        cloud_sample @ R, *bbox.rotated(R.T), 
                        functools.partial(valid_positions, R, profile.Trimesh.vertices, depth, K)
                    )

                    tmp_config = (-val, (R, o))
                    if tmp_config[0] < best_config[0]:
                        best_config = tmp_config
        return best_config[1], cloud_orientation
    
    def get_cloud(self, instance, depth, K):
        mask = np.array(instance.pred_masks[0]) & (depth > 0)
        
        depth_values = depth[mask]
        y, x = np.where(mask)
        img_coords = np.column_stack((x, y, np.ones_like(depth_values)))
        camera_coords = np.linalg.inv(K) @ (img_coords.T * depth_values)
        return camera_coords.T / 1000.0

    def select_largest_cluster(self, cloud):
        clustering = DBSCAN(eps=0.01).fit(cloud)
        counter = Counter(clustering.labels_)
        label, count = counter.most_common(1)[0]
        return cloud[clustering.labels_ == label]




            
        