import numpy as np
class DetectedObject:
    def __init__(self, R, o, bounding_box, mask, mesh_file, name, mesh = None):
        self.R = R
        self.o = o
        self.BoundingBox = np.array(bounding_box.tensor[0])
        self.Mask = np.array(mask)
        self.MeshFile = mesh_file
        self.Name = name
        self.Mesh = mesh