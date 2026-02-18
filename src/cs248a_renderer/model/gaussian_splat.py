from dataclasses import field
import slangpy as spy
import pathlib
from pyntcloud import PyntCloud
import numpy as np
import pandas as pd
import logging

from cs248a_renderer.model.transforms import Transform3D


logger = logging.getLogger(__name__)


class GaussianSplat:
    transform: Transform3D = field(default_factory=Transform3D)

    def __init__(self, device: spy.Device, path: pathlib.Path) -> None:
        # Load modules.
        self._math_module = spy.Module.load_from_file(device=device, path="math.slang")
        self._model_module = spy.Module.load_from_file(
            device=device, path="model.slang", link=[self._math_module]
        )
        point_cloud = PyntCloud.from_file(str(path.resolve()))
        points: pd.DataFrame = point_cloud.points
        self.num_gaussians = len(points)
        logger.info(f"Point cloud loaded from {path} with {self.num_gaussians} points.")

        # Extract Gaussian features.
        positions = np.ascontiguousarray(points[["x", "y", "z"]].to_numpy())
        position_buf = spy.NDBuffer(
            device=device,
            dtype=self._model_module.float3,
            shape=(self.num_gaussians,),
        )
        position_buf.copy_from_numpy(positions)
        # Convert quaternion order from scalar first to scalar last.
        rotations = np.ascontiguousarray(
            points[["rot_1", "rot_2", "rot_3", "rot_0"]].to_numpy()
        )
        rotations = rotations / np.linalg.norm(rotations, axis=-1)[:, np.newaxis]
        rotation_buf = spy.NDBuffer(
            device=device,
            dtype=self._model_module.float4,
            shape=(self.num_gaussians,),
        )
        rotation_buf.copy_from_numpy(rotations)
        scales = np.ascontiguousarray(
            points[["scale_0", "scale_1", "scale_2"]].to_numpy()
        )
        scale_buf = spy.NDBuffer(
            device=device,
            dtype=self._model_module.float3,
            shape=(self.num_gaussians,),
        )
        scale_buf.copy_from_numpy(scales)
        colors = np.ascontiguousarray(points[["f_dc_0", "f_dc_1", "f_dc_2"]].to_numpy())
        color_buf = spy.NDBuffer(
            device=device,
            dtype=self._model_module.float3,
            shape=(self.num_gaussians,),
        )
        color_buf.copy_from_numpy(colors)
        opacities = np.ascontiguousarray(points["opacity"].to_numpy())
        opacity_buf = spy.NDBuffer(
            device=device,
            dtype=self._model_module.float,
            shape=(self.num_gaussians,),
        )
        opacity_buf.copy_from_numpy(opacities)
        # Load point data to Gaussian buffer.
        self.gaussians = spy.InstanceBuffer(
            struct=self._model_module.Gaussian.as_struct(), shape=(self.num_gaussians,)
        )
        self._model_module.unpackGaussianSplat(
            tid=spy.grid(shape=(self.num_gaussians,)),
            soaSplat={
                "position": position_buf,
                "rotation": rotation_buf,
                "scale": scale_buf,
                "color": color_buf,
                "opacity": opacity_buf,
            },
            gaussians=self.gaussians,
        )
