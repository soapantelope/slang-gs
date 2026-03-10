from dataclasses import dataclass, field
import pathlib
from pyntcloud import PyntCloud
import numpy as np
import pandas as pd
import logging
import torch

logger = logging.getLogger(__name__)

@dataclass
class GaussianSplat():
    
    def __init__(self, path: pathlib.Path, **kwargs) -> None:
        print("parsing ply")
        point_cloud = PyntCloud.from_file(str(path.resolve()))
        points: pd.DataFrame = point_cloud.points
        self.num_gaussians = len(points)
        logger.info(f"Point cloud loaded from {path} with {self.num_gaussians} points.")

        self.positions = torch.tensor(
            points[["x", "y", "z"]].values, dtype=torch.float32, device="cuda"
        )
        rotations = points[["rot_1", "rot_2", "rot_3", "rot_0"]].values.astype(np.float32)
        rotations = rotations / np.linalg.norm(rotations, axis=-1, keepdims=True)
        self.rotations = torch.tensor(rotations, dtype=torch.float32, device="cuda")
        self.scales = torch.tensor(
            np.exp(points[["scale_0", "scale_1", "scale_2"]].values),
            dtype=torch.float32, device="cuda",
        )
        self.colors = torch.tensor(
            points[["f_dc_0", "f_dc_1", "f_dc_2"]].values,
            dtype=torch.float32, device="cuda",
        )
        raw_opacities = points["opacity"].values
        self.opacities = torch.tensor(
            1.0 / (1.0 + np.exp(-raw_opacities)),
            dtype=torch.float32, device="cuda",
        )