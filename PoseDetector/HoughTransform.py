import trimesh
from scipy.ndimage import morphology
from scipy.ndimage import gaussian_filter
import scipy.ndimage
from scipy.spatial.transform import Rotation
import time
from numba import cuda
from PoseDetector.BoundingBox import BoundingBox
import numpy as np
from appdirs import user_cache_dir


@cuda.jit('(f4[:,:,:], i4[:], f4[:], f4, f4[:], i4[:], f4[:], f4, f4[:,:])')
def accumulate(
        accumulator, a_shape, a_lower, a_grid_size,
        profile, p_shape, p_lower, p_grid_size,
        points
    ):
    id = cuda.blockIdx.x * 1024 + cuda.threadIdx.x
    if id >= accumulator.size:
        return
    idx = id // (a_shape[2] * a_shape[1])
    idy = id // a_shape[2]
    idy = idy - (idy // a_shape[1]) * a_shape[1]
    idz = id - idx * (a_shape[2] * a_shape[1]) - idy * a_shape[2]
    pos_x = a_lower[0] + (0.5 + idx) * a_grid_size
    pos_y = a_lower[1] + (0.5 + idy) * a_grid_size
    pos_z = a_lower[2] + (0.5 + idz) * a_grid_size
    
    
    for i in range(len(points)):
        p_ind_i = int((pos_x - p_lower[0] - points[i,0]) / p_grid_size)
        p_ind_j = int((pos_y - p_lower[1] - points[i,1]) / p_grid_size)
        p_ind_k = int((pos_z - p_lower[2] - points[i,2]) / p_grid_size)
        if p_ind_i >= 0 and p_ind_i < p_shape[0] \
            and p_ind_j >= 0 and p_ind_j < p_shape[1] \
            and p_ind_k >= 0 and p_ind_k < p_shape[2]: 
            p_ind = int(p_ind_i * (p_shape[1] * p_shape[2]) + p_ind_j * (p_shape[2]) + p_ind_k)
            accumulator[idx, idy, idz] += profile[p_ind]

@cuda.jit
def filter(
        mask, mask_lower, m_grid_size,
        R, vertices,
        depth, K
    ):

    id = cuda.blockIdx.x * 1024 + cuda.threadIdx.x
    if id >= mask:
        return
    idx = id // (mask.shape[2] * mask.shape[1])
    idy = id // mask.shape[2]
    idy = idy - (idy // mask.shape[1]) * mask.shape[1]
    idz = id - idx * (mask.shape[2] * mask.shape[1]) - idy * mask.shape[2]
    pos_x = mask_lower[0] + (0.5 + idx) * m_grid_size
    pos_y = mask_lower[1] + (0.5 + idy) * m_grid_size
    pos_z = mask_lower[2] + (0.5 + idz) * m_grid_size
    o = np.array((pos_x, pos_y, pos_z))
    for i in range(len(vertices)):
        oriented = R @ (vertices[i] + o)
        img_coords = K @ oriented
        img_coords /= img_coords[2]
        if depth[int(img_coords[1]), int(img_coords[0])] > oriented[2]:
            mask[idx, idy, idz] = False
    

def accumulator_profile(mesh, name, grid_size = 0.0005, radius = 0.001):
    import os
    cache_dir = user_cache_dir("Mesh-Pose-Detector")
    profile_dir = os.path.join(cache_dir, 'HoughTransform')
    if not os.path.isdir(profile_dir):
        os.mkdir(profile_dir)
    bottom_left = -np.max(mesh.vertices, axis = 0) - 2*radius
    profile_cache = os.path.join(profile_dir, f'{name}-{grid_size}-{radius}.npy')
    if os.path.isfile(profile_cache):
        return np.load(profile_cache), bottom_left
    print(f'Building profile for {name}')
    shape = tuple(np.array((-np.min(mesh.vertices, axis = 0) - bottom_left + grid_size + 2*radius) // grid_size, np.int32))
    
    X, Y, Z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]), 
        indexing = 'ij'
    )
    xyz = (np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))) * grid_size + bottom_left
    xyz *= -1

    dist = mesh.nearest.signed_distance(xyz)
    profile = np.exp(-(dist.reshape(shape)/(2*radius))**2).astype(np.float32)

    np.save(profile_cache, profile)
    return profile, bottom_left

class HoughTransform:
    def __init__(self, mesh, name, profile_grid_size, accumulation_radius, accumulator_grid_size):
        self.Mesh = mesh
        self.Profile, self.ProfileOrigin = accumulator_profile(mesh, name, profile_grid_size, accumulation_radius)
        self.ProfileResolution = profile_grid_size
        self.PShape = np.array(self.Profile.shape, np.int32)
        self.Profile = self.Profile.flatten()
        self.ProfGPU = cuda.to_device(self.Profile)
        self.AccumulatorGridSize = accumulator_grid_size


    def fit(self, R, o, points):
        lowers = points @ R.T  + self.ProfileOrigin
        indices = ((o - lowers) // self.ProfileResolution).astype(np.int)
        valid = np.all((indices >= 0) & (indices <= self.PShape), axis = 1)
        return np.sum(self.Profile[indices[valid]])

    def vote(self, points, lower, upper, mask_func = None):
        points = points.astype(np.float32)
        lower = lower.astype(np.float32)
        upper = upper.astype(np.float32)
        shape = np.array((upper - lower) // self.AccumulatorGridSize + 1, np.int32)
        accumulator = np.zeros(shape, np.float32)

        s = cuda.stream()
        accumulate[accumulator.size // 1024 + 1, 1024, s](
            accumulator, shape, lower, self.AccumulatorGridSize, 
            self.ProfGPU, self.PShape, self.ProfileOrigin.astype(np.float32), self.ProfileResolution,
            points
        )

        mask = np.full(shape, 1, np.float32)
        if mask_func is not None:
            mask_func(mask, lower, self.AccumulatorGridSize)
        s.synchronize()
        accumulator = accumulator * mask
        max_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)

        return (np.array(max_idx) + 0.5) * self.AccumulatorGridSize + lower, accumulator[max_idx]

    def best_rotation(self, points, bbox, start, mask_func = None):
        from scipy.optimize import minimize
        def f(x):
            R = Rotation.from_euler('XYZ', x)
            oriented = R.apply(points)
            val = -self.vote(R.as_matrix(), oriented, *bbox.rotated(R.as_matrix()), mask_func)[1]
            return val
        m = minimize(f, start, method = 'powell')
        x = m['x']
        R = Rotation.from_euler('XYZ', x)
        oriented = R.apply(points)
        pos, val = self.vote(R.as_matrix(), oriented, *bbox.rotated(R.as_matrix()), mask_func)
        return pos, x