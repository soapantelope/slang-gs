"""
A module defining volume data structures.
"""

from dataclasses import dataclass, field
from typing import Tuple, TypedDict, List
import numpy as np
from pyglm import glm
import slangpy as spy

from cs248a_renderer.model.transforms import Transform3D
from cs248a_renderer.model.scene_object import SceneObject
from cs248a_renderer.model.bounding_box import BoundingBox3D


class VolumeProperties(TypedDict):
    """TypedDict for volume properties."""

    # Size of each voxel in world units.
    voxel_size: float
    # Pivot point for transformations, in normalized coordinates (x, y, z).
    pivot: tuple[float, float, float]


@dataclass
class DenseVolume(SceneObject):
    """A dense volume data structure represented as a 4D numpy array.

    :param data: Volme data as a 4D numpy array.
    :param properties: Properties of the volume, including voxel size and pivot.
    :param transform: Transform for the volume.
    """

    # Volume data stored as a 4D numpy array, with shape (depth, height, width, channels).
    data: np.ndarray = field(
        default_factory=lambda: np.empty((1, 1, 1, 4), dtype=np.float32)
    )
    # Size of each voxel in world units.
    properties: VolumeProperties = field(
        default_factory=lambda: {
            "voxel_size": 0.01,
            "pivot": (0.5, 0.5, 0.5),
        }
    )

    def __post_init__(self):
        # Check that data is a 4D numpy array.
        if self.data.ndim != 4:
            raise ValueError(
                "Data must be a 4D numpy array (depth, height, width, channels)."
            )
        # Check that voxel_size is positive.
        if self.properties["voxel_size"] <= 0:
            raise ValueError("voxel_size must be a positive float.")
        # Check pivot values are in [0, 1].
        if not all(0.0 <= p <= 1.0 for p in self.properties["pivot"]):
            raise ValueError("Pivot values must be in the range [0, 1].")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape[:3]

    @property
    def channels(self) -> int:
        return self.data.shape[3]

    @property
    def bounding_box(self) -> BoundingBox3D:
        """Model space axis-aligned bounding box of the volume.

        :return: A tuple containing the minimum and maximum corners of the bounding box.
        """
        depth, height, width = self.shape
        voxel_size = self.properties["voxel_size"]
        pivot = self.properties["pivot"]

        size = glm.vec3(width * voxel_size, height * voxel_size, depth * voxel_size)
        min_corner = glm.vec3(
            -pivot[0] * size.x, -pivot[1] * size.y, -pivot[2] * size.z
        )
        max_corner = min_corner + size

        return BoundingBox3D(min=min_corner, max=max_corner)


def create_volume_buf(
    module: spy.Module, dense_volumes: List[DenseVolume]
) -> Tuple[spy.NDBuffer, spy.NDBuffer]:
    np_volumes = []
    sl_volumes = []
    offset = 0

    for volume in dense_volumes:
        volume_shape = volume.shape
        volume_bound = volume.bounding_box
        sl_volume = {
            "bound": volume_bound.get_this(),
            "tex": {
                "size": [volume_shape[2], volume_shape[1], volume_shape[0]],
                "offset": offset,
            },
            "modelMatrix": np.ascontiguousarray(
                volume.get_transform_matrix(), dtype=np.float32
            ),
            "invModelMatrix": np.ascontiguousarray(
                glm.inverse(volume.get_transform_matrix()), dtype=np.float32
            ),
        }
        sl_volumes.append(sl_volume)
        offset += volume_shape[2] * volume_shape[1] * volume_shape[0]
        np_volume = volume.data.reshape(-1, 4)
        np_volumes.append(np_volume)

    if len(np_volumes) == 0:
        np_volumes = [np.zeros((1, 4), dtype=np.float32)]

    np_volume_concat = np.concatenate(np_volumes, axis=0)
    volume_tex_buf = spy.NDBuffer(
        device=module.device,
        dtype=module.float4,
        shape=(max(np_volume_concat.shape[0], 1),),
    )
    volume_tex_buf.copy_from_numpy(np_volume_concat)
    volume_buf = spy.NDBuffer(
        device=module.device,
        dtype=module.Volume.as_struct(),
        shape=(max(len(sl_volumes), 1),),
    )
    cursor = volume_buf.cursor()
    for idx, sl_volume in enumerate(sl_volumes):
        cursor[idx].write(sl_volume)
    cursor.apply()
    return volume_buf, volume_tex_buf
