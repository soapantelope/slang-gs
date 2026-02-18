"""
A module defining nerf data structures.
"""

from dataclasses import dataclass, field
from typing import Tuple, TypedDict, List
import numpy as np
from pyglm import glm
import slangpy as spy
import slangpy_nn as nn

from cs248a_renderer.model.transforms import Transform3D


class NeRFProperties(TypedDict):
    """TypedDict for NeRF properties."""

    # Bounding box size.
    bounding_box_size: tuple[float, float, float]
    # Pivot point for transformations, in normalized coordinates (x, y, z).
    pivot: tuple[float, float, float]


class NeRF:
    """A NeRF data structure represented as a 4D numpy array.

    :param data: Volme data as a 4D numpy array.
    :param voxel_size: Size of each voxel in world units.
    :param pivot: Pivot point for transformations, in normalized coordinates (x, y, z).
    :param transform: Transform for the volume.
    """

    # Transform for the volume.
    transform: Transform3D = field(default_factory=Transform3D)
    # Size of each voxel in world units.
    properties: NeRFProperties = field(
        default_factory=lambda: {
            "bounding_box_size": (1.0, 1.0, 1.0),
            "pivot": (0.5, 0.5, 0.5),
        }
    )

    # Check if device supports cooperative vector
    use_coopvec: bool = False

    # NeRF MLP weights.
    mlp: nn.IModel

    def __init__(
        self,
        module: spy.Module,
        transform: Transform3D | None = None,
        properties: NeRFProperties | None = None,
        use_coopvec: bool = False,
        mlp_weights: List[np.ndarray] | None = None,
    ) -> None:
        self.transform = transform if transform is not None else Transform3D()
        self.properties = (
            properties
            if properties is not None
            else {
                "bounding_box_size": (1.0, 1.0, 1.0),
                "pivot": (0.5, 0.5, 0.5),
            }
        )
        self.use_coopvec = use_coopvec

        # Check pivot values are in [0, 1].
        if not all(0.0 <= p <= 1.0 for p in self.properties["pivot"]):
            raise ValueError("Pivot values must be in the range [0, 1].")

        if self.use_coopvec:
            mlp_input = nn.ArrayKind.coopvec
            mlp_precision = nn.Real.half
        else:
            mlp_input = nn.ArrayKind.array
            mlp_precision = nn.Real.float

        self.mlp = nn.ModelChain(
            nn.Convert.to_precision(mlp_precision),
            nn.Convert.to_array_kind(mlp_input),
            nn.LinearLayer(num_inputs=nn.Auto, num_outputs=64),
            nn.ReLU(),
            nn.LinearLayer(num_inputs=nn.Auto, num_outputs=64),
            nn.ReLU(),
            nn.LinearLayer(num_inputs=nn.Auto, num_outputs=4),
            nn.Convert.to_vector(),
        )

        self.mlp.initialize(module=module, input_type="float[45]")

        if mlp_weights is not None:
            for i, t in enumerate(self.mlp.parameters()):
                t.copy_from_numpy(mlp_weights[i])

    @property
    def bounding_box(self) -> Tuple[glm.vec3, glm.vec3]:
        """Model space axis-aligned bounding box of the volume.

        :return: A tuple containing the minimum and maximum corners of the bounding box.
        """
        bounding_box_size = self.properties["bounding_box_size"]
        pivot = self.properties["pivot"]

        size = glm.vec3(*bounding_box_size)
        min_corner = glm.vec3(
            -pivot[0] * size.x, -pivot[1] * size.y, -pivot[2] * size.z
        )
        max_corner = min_corner + size

        return (min_corner, max_corner)
