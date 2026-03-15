import torch
import numpy as np
from plyfile import PlyData

# temporary version of 3DGS GaussianModel for testing
class GaussianModel:
    def __init__(self, ply_path: str):
        self._load_ply(ply_path)

    def _load_ply(self, path: str) -> None:
        plydata = PlyData.read(path)
        v = plydata["vertex"]

        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
        rots = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
        scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
        f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
        opac = np.array(v["opacity"])

        self._xyz = torch.tensor(xyz, dtype=torch.float32, device="cuda")
        self._rotation = torch.tensor(rots, dtype=torch.float32, device="cuda")
        self._scaling = torch.tensor(scales, dtype=torch.float32, device="cuda")
        self._features_dc = torch.tensor(f_dc, dtype=torch.float32, device="cuda").unsqueeze(1)
        self._opacity = torch.tensor(opac, dtype=torch.float32, device="cuda").unsqueeze(1)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)
