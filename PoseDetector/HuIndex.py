import rtree
from rtree import index
import numpy as np
import trimesh
import os
from scipy.stats import special_ortho_group
import cv2

class HuIndex:
    def __init__(self, name, mesh, grid_size = 0.0005):

        if not os.path.isdir('Indices'):
            os.mkdir('Indices')
        index_name = os.path.join('Indices', name)

        p = index.Property()
        p.dimension = 7
        if not os.path.isfile(index_name + '.idx'):
            self.Index = index.Index(index_name, properties = p, interleaved = True)
            self.build_moment_index(index_name, mesh, grid_size)
        else: 
            self.Index = index.Index(index_name, properties = p, interleaved = True)
    
    def query_moment_index(self, hu):
        matches = list(n.object for n in self.Index.intersection(np.concatenate((hu, hu), axis = 0), objects = True))
        if len(matches) > 0:
            return matches
        return [list(self.Index.nearest(np.concatenate((hu, hu), axis = 0), objects = True))[0].object]


    def build_moment_index(self, name, mesh, grid_size):
        print(f'Building HuIndex for {name}')
        n_samples = 1000
        for i in range(n_samples):
            print(f'{i}/{n_samples}')
            R = special_ortho_group(3).rvs()

            mask, offset, vt = self.mesh_to_mask(mesh, R, grid_size)
            hu = cv2.HuMoments(cv2.moments(np.float32(mask)))

            lower, upper = hu - abs(hu * 0.3), hu + abs(hu * 0.3)
            self.Index.insert(i, np.concatenate((lower, upper), axis = 0), obj = (R, offset, vt))

    @staticmethod
    def mesh_to_mask(mesh: trimesh.Trimesh, R, grid_size):
        mesh.vertices = mesh.vertices @ R.T
        bottom_left = np.min(mesh.vertices[:,:2], axis = 0)
        top_right = np.max(mesh.vertices[:,:2], axis = 0)

        x = np.arange(bottom_left[0], top_right[0], grid_size)
        y = np.arange(bottom_left[1], top_right[1], grid_size)
        X, Y = np.meshgrid(x, y)
        z = np.full_like(X, -5.0)
        xyz = np.stack((X, Y, z), axis = 2).reshape((-1, 3))
        d = np.zeros_like(xyz)
        d[:,2] = 1
        mask = mesh.ray.intersects_any(xyz, d)
        mesh.vertices = mesh.vertices @ R

        intersection_points, *_ = mesh.ray.intersects_location(xyz, d)
        offset = np.mean(intersection_points, axis = 0)

        u, s, vt = np.linalg.svd(xyz[mask,:2] - np.mean(xyz[mask,:2], axis = 0))
        if vt[0,0] * vt[1,1] - vt[0,1] * vt[1,0] < 0:
            vt[1,:] *= -1

        if np.sum((xyz[mask,:2] - np.mean(xyz[mask,:2])) @ vt[0,:] < 0) > np.sum(mask) // 2:
            vt = -vt
        return mask.reshape(X.shape).T, offset, vt