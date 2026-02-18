"""
Scene manager for volumetric rendering application.
"""

import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
from pyglm import glm
import slangpy as spy
import open3d as o3d

from cs248a_renderer.model.cameras import PerspectiveCamera
from cs248a_renderer.model.scene import SingleVolumeScene, NeRFScene, Scene
from cs248a_renderer.model.volumes import DenseVolume, VolumeProperties
from cs248a_renderer.model.nerf import NeRF, NeRFProperties
from cs248a_renderer.model.transforms import Transform3D
from cs248a_renderer.model.ray_marcher_config import RayMarcherConfig
from cs248a_renderer.model.scene_object import get_next_scene_object_index
from cs248a_renderer.model.mesh import Mesh


logger = logging.getLogger(__name__)


DEFAULT_CAM_TRANSFORM = Transform3D(
    position=glm.vec3(0.0, 0.0, 2.5),
    rotation=glm.quat(1.0, 0.0, 0.0, 0.0),
    scale=glm.vec3(1.0, 1.0, 1.0),
)


class SceneManager:
    scene: Scene

    volume_scene: SingleVolumeScene | None = None
    nerf_scene: NeRFScene | None = None

    def __init__(self):
        self.scene = Scene()

    def load_mesh(self, mesh_path: Path, name: str | None = None) -> None:
        """
        Load a mesh from the given path and add it to the scene root.

        :param mesh_path: Path to the mesh file.
        :type mesh_path: Path
        :param name: Name of the mesh object. If None, a default name will be assigned.
        :type name: str | None
        """
        logger.info(f"Loading mesh from {mesh_path}")
        o3d_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if name is None:
            name = f"mesh_{get_next_scene_object_index()}"
        mesh = Mesh(o3d_mesh=o3d_mesh, name=name)
        self.scene.add_object(mesh)
        logger.info(f"Added mesh '{mesh.name}' to scene")

    def load_volume(self, volume_path: Path) -> None:
        """Load a volume from the given path and add it to the scene.

        :param volume_path: Path to the volume numpy file.
        :param properties: Properties for the volume.
        """
        logger.info(f"Loading volume from {volume_path}")
        volume_data: np.ndarray = np.load(volume_path)
        if volume_data.ndim == 3:
            volume_data = np.zeros(volume_data.shape + (4,), dtype=np.float32)
            volume_data[..., 3] = volume_data  # Set alpha channel
            volume_data[..., 0:3] = 0.0  # Set RGB channels
        elif volume_data.ndim != 4:
            raise ValueError(
                f"Volume data must be 3D or 4D (w, h, d, c), got shape {volume_data.shape}"
            )
        # Check that volume has 4 channels
        if volume_data.shape[3] != 4:
            raise ValueError(
                f"Volume data must have 4 channels (RGBA), got shape {volume_data.shape}"
            )
        logger.info(f"Loaded volume data with shape {volume_data.shape}")
        volume = DenseVolume(
            data=volume_data,
            properties=VolumeProperties(pivot=(0.5, 0.5, 0.5), voxel_size=0.01),
        )
        self.scene.single_volume = volume
        logger.info(f"Added volume to scene")

    def create_nerf_from_numpy(
        self, module: spy.Module, nerf_path: Path, properties: NeRFProperties
    ) -> None:
        """Create a new scene with the given NeRF.

        :param nerf_path: Path to the numpy file containing NeRF data.
        :param properties: Properties for the NeRF.
        """
        logger.info(f"Creating scene with NeRF from {nerf_path}")
        # Load NeRF data from file.
        nerf_np_file = np.load(nerf_path)
        mlp_weights: List[np.ndarray] = list(nerf_np_file.values())
        logger.info(f"Loaded NeRF data with keys {list(nerf_np_file.keys())}")
        nerf = NeRF(module=module, mlp_weights=mlp_weights, properties=properties)
        # Placeholder for scene creation logic
        self.nerf_scene = NeRFScene(
            nerf=nerf,
            camera=PerspectiveCamera(
                transform=DEFAULT_CAM_TRANSFORM,
            ),
            ray_marcher_config=RayMarcherConfig(),
        )
        self.volume_scene = None

    def create_empty_nerf(
        self,
        module: spy.Module,
        properties: NeRFProperties,
    ) -> None:
        """Create an empty scene with a NeRF with no weights.

        :param properties: Properties for the NeRF.
        """
        logger.info(f"Creating empty scene with NeRF")
        nerf = NeRF(module=module, properties=properties)
        self.nerf_scene = NeRFScene(
            nerf=nerf,
            camera=PerspectiveCamera(
                transform=DEFAULT_CAM_TRANSFORM,
            ),
            ray_marcher_config=RayMarcherConfig(),
        )
        self.volume_scene = None

    def create_volume_from_numpy(
        self, volume_path: Path, properties: VolumeProperties
    ) -> None:
        """Create a new scene with the given volume.

        :param volume: The dense volume to include in the scene.
        :param volume_properties: Properties for the volume.
        """
        logger.info(f"Creating scene with volume from {volume_path}")
        # Load volume data from file.
        volume_data: np.ndarray = np.load(volume_path)
        if volume_data.ndim == 3:
            volume_data = volume_data[..., np.newaxis]
        elif volume_data.ndim != 4:
            raise ValueError(
                f"Volume data must be 3D or 4D (w, h, d, c), got shape {volume_data.shape}"
            )
        logger.info(f"Loaded volume data with shape {volume_data.shape}")
        volume = DenseVolume(data=volume_data, properties=properties)
        # Placeholder for scene creation logic
        self.volume_scene = SingleVolumeScene(
            volume=volume,
            camera=PerspectiveCamera(
                transform=DEFAULT_CAM_TRANSFORM,
            ),
            ray_marcher_config=RayMarcherConfig(),
        )
        self.nerf_scene = None

    def create_empty_volume(
        self,
        size: Tuple[int, int, int, int],
        properties: VolumeProperties,
    ) -> None:
        """Create an empty scene with a volume of given size.

        :param size: The size of the empty volume (w, h, d, c).
        """
        logger.info(f"Creating empty scene with volume size {size}")
        volume_data = np.zeros(size, dtype=np.float32)
        volume = DenseVolume(data=volume_data, properties=properties)
        self.volume_scene = SingleVolumeScene(
            volume=volume,
            camera=PerspectiveCamera(
                transform=DEFAULT_CAM_TRANSFORM,
            ),
            ray_marcher_config=RayMarcherConfig(),
        )
        self.nerf_scene = None
