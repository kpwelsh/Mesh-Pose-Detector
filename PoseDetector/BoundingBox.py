import numpy as np

class BoundingBox:
    def __init__(self, lower, upper):
        self.Lower = lower
        self.Upper = upper
        dimension = len(self.Lower)
        self.Vertices = np.empty((2**dimension, len(self.Lower)))
        bounds = [self.Lower, self.Upper]
        for i in range(2**dimension):
            for j in range(dimension):
                self.Vertices[i,j] = bounds[int(bool(i & (1<<j)))][j]

    def rotated(self, R):
        vertices = self.Vertices @ R.T
        return np.min(vertices, axis = 0), np.max(vertices, axis = 0)
        