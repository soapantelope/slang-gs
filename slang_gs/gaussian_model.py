import torch
import pandas as pd
from pyntcloud import PyntCloud

# temporary version of 3DGS GaussianModel for testing
class GaussianModel:
    def __init__(self, ply_path: str):
        self._load_ply(ply_path)

    def _load_ply(self, path: str) -> None:
        points: pd.DataFrame = PyntCloud.from_file(path).points

        self._xyz = torch.tensor(points[["x", "y", "z"]].values, dtype=torch.float32, device="cuda")
        self._rotation = torch.tensor(points[["rot_0", "rot_1", "rot_2", "rot_3"]].values, dtype=torch.float32, device="cuda")
        self._scaling = torch.exp(torch.tensor(points[["scale_0", "scale_1", "scale_2"]].values, dtype=torch.float32, device="cuda"))
        self._features_dc = torch.tensor(points[["f_dc_0", "f_dc_1", "f_dc_2"]].values, dtype=torch.float32, device="cuda").unsqueeze(1)  # (N, 1, 3)
        self._opacity = torch.sigmoid(torch.tensor(points["opacity"].values, dtype=torch.float32, device="cuda")).unsqueeze(1)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_opacity(self):
        return self._opacity
