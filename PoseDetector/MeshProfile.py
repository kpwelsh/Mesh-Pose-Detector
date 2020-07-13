import trimesh
from PoseDetector.HuIndex import HuIndex
from PoseDetector.HoughTransform import HoughTransform
class MeshProfile:
    def __init__(self, name, mesh_file, **config):
        self.MeshFile = mesh_file
        self.Name = name
        self.Trimesh = trimesh.load_mesh(mesh_file)
        self.HuIndex = HuIndex(name, self.Trimesh, config.get("hu_index.grid_size"))
        self.HoughTransform = HoughTransform(
            self.Trimesh, 
            name, 
            config.get("hough_transform.profile.grid_size"),
            config.get("hough_transform.profile.accumulation_radius"),
            config.get("hough_transform.accumulator.grid_size")
        )