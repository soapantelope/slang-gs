"""
Camera parameters.
"""

from dataclasses import dataclass, field
from pyglm import glm
import numpy as np

from cs248a_renderer.model.scene_object import SceneObject
from cs248a_renderer.model.transforms import Transform3D


@dataclass
class PerspectiveCamera(SceneObject):
    """Perspective camera parameters.

    :param fov: Field of view in degrees.
    :param near: Near clipping plane distance.
    :param far: Far clipping plane distance.
    """

    # Camera intrinsics.
    fov: float = 45.0
    near: float = 0.1
    far: float = 100.0
    # Camera transform.
    transform: Transform3D = field(
        default_factory=lambda: Transform3D(
            position=glm.vec3(0.0, 0.0, 1.0),
            rotation=glm.quat(1.0, 0.0, 0.0, 0.0),
            scale=glm.vec3(1.0, 1.0, 1.0),
        )
    )

    def view_matrix(self) -> glm.mat4:
        """Calculate the view matrix.

        :return: View matrix as glm.mat4.
        """
        return glm.inverse(self.transform.get_matrix())

    def projection_matrix(self, w: int, h: int) -> glm.mat4:
        """Calculate the projection matrix.

        :param w: Width of the viewport.
        :param h: Height of the viewport.
        :return: Projection matrix as glm.mat4.
        """
        return glm.perspectiveFov(
            glm.radians(self.fov),
            w,
            h,
            self.near,
            self.far,
        )

    def focal_length(self, h: int) -> float:
        """Calculate the focal length in pixels.

        :param h: Height of the viewport.
        :return: Focal length in pixels.
        """
        return (0.5 * float(h)) / np.tan(np.radians(self.fov) / 2.0)
