import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from scipy.ndimage import morphology
from scipy.ndimage import gaussian_filter
import scipy.ndimage
from scipy.spatial.transform import Rotation
import time
from PoseDetector.BoundingBox import BoundingBox
import cupy as cp

accumulate = cp.RawKernel(
r'''
extern "C" __global__
void accumulate(
        float* accumulator, const int* a_shape, const float* a_lower, const float a_grid_size,
        const float* profile, const int* p_shape, const float* p_lower, const float p_grid_size,
        const float* points, const int n
    ) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int a_size = a_shape[0]*a_shape[1]*a_shape[2];
    if (id >= (a_size * n)) {
        return;
    }
    int i = id / a_size;
    id = id % a_size;
    int idx = id / (a_shape[2] * a_shape[1]);
    int idy = id / a_shape[2];
    idy = idy - (idy / a_shape[1]) * a_shape[1];
    int idz = id - idx * (a_shape[2] * a_shape[1]) - idy * a_shape[2];
    float pos_x = a_lower[0] + (0.5 + idx) * a_grid_size;
    float pos_y = a_lower[1] + (0.5 + idy) * a_grid_size;
    float pos_z = a_lower[2] + (0.5 + idz) * a_grid_size; 

    int p_ind_i = (pos_x - p_lower[0] - points[i]) / p_grid_size;
    int p_ind_j = (pos_y - p_lower[1] - points[i + n]) / p_grid_size;
    int p_ind_k = (pos_z - p_lower[2] - points[i + 2 * n]) / p_grid_size;
    if (p_ind_i >= 0 && p_ind_i < p_shape[0] 
        && p_ind_j >= 0 && p_ind_j < p_shape[1] 
        && p_ind_k >= 0 && p_ind_k < p_shape[2]) {
        int p_ind = p_ind_i * (p_shape[1] * p_shape[2]) + p_ind_j * (p_shape[2]) + p_ind_k;
        atomicAdd(accumulator + id, profile[p_ind]);
    }
}
''', 'accumulate')


def accumulator_profile(mesh, name, grid_size = 0.0005, radius = 0.001):
    import os
    profile_dir = 'profiles'
    if not os.path.isdir(profile_dir):
        os.mkdir(profile_dir)
    bottom_left = -np.max(mesh.vertices, axis = 0) - 2*radius
    profile_cache = os.path.join(profile_dir, f'{name}-{grid_size}-{radius}.npy')
    if os.path.isfile(profile_cache):
        return np.load(profile_cache), bottom_left
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
    plt.imshow(profile[:,:,0])
    plt.show()

    np.save(profile_cache, profile)
    return profile, bottom_left


class HoughTransform:
    def __init__(self, mesh, name, profile_grid_size, accumulation_radius, resolution):
        self.Mesh = mesh
        self.Profile, self.ProfileOrigin = accumulator_profile(mesh, name, profile_grid_size, accumulation_radius)
        self.ProfileResolution = profile_grid_size
        self.PShape = cp.array(self.Profile.shape, np.int32)
        self.Profile = self.Profile.flatten()
        self.Resolution = resolution


    def fit(self, R, o, points):
        lowers = points @ R.T  + self.ProfileOrigin
        indices = ((o - lowers) // self.ProfileResolution).astype(np.int)
        valid = np.all((indices >= 0) & (indices <= self.PShape), axis = 1)
        return np.sum(self.Profile[indices[valid]])

    def vote(self, points, lower, upper, mask_func = None):
        points = np.array(points.astype(np.float32).flatten(order='F'))
        lower = np.array(lower, np.float32)
        upper = upper.astype(np.float32)
        shape = np.array((upper - lower) // self.Resolution + 1, np.int32)
        accumulator = cp.zeros((shape[0] * shape[1] * shape[2],), cp.float32)

        
        n_threads = accumulator.size * (len(points) // 3)
        accumulate((n_threads // 512 + 1,), (512,), (
            accumulator, cp.asarray(shape, cp.int32), cp.asarray(lower, cp.float32), cp.float32(self.Resolution), 
            cp.asarray(self.Profile, cp.float32), self.PShape, cp.asarray(self.ProfileOrigin, cp.float32), cp.float32(self.ProfileResolution),
            cp.asarray(points, cp.float32), cp.int32(len(points) // 3)
        ))


        accumulator = cp.asnumpy(accumulator).reshape(shape)
        if mask_func is not None:
            mask = np.ones(accumulator.shape)
            mask_func(mask, lower, self.Resolution)
            accumulator *= mask
        max_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        return (np.array((*max_idx,)) + 0.5) * self.Resolution + lower, accumulator[max_idx]