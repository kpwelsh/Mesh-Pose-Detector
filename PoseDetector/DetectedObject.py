class DetectedObject:
    def __init__(self, R, o, bounding_box_2d, mask, mesh_file, mesh = None):
        self.R = R
        self.o = o
        self.BoundingBox2d = bounding_box_2d
        self.Mask = mask
        self.MeshFile = mesh_file
        self.Mesh = mesh