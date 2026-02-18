"""
A module defining transform data structures.
"""

from dataclasses import dataclass

from pyglm import glm


@dataclass(eq=False)
class Transform3D:
    """A simple 3D transform with position, rotation (quaternion), and scale.

    :param position: Position of the transform in 3D space.
    :param rotation: Rotation of the transform as a quaternion (x, y, z, w).
    :param scale: Scale of the transform in 3D space.
    """

    position: glm.vec3 = glm.vec3(0.0, 0.0, 0.0)
    rotation: glm.quat = glm.quat(1.0, 0.0, 0.0, 0.0)
    scale: glm.vec3 = glm.vec3(1.0, 1.0, 1.0)

    def get_matrix(self) -> glm.mat4:
        """Compute the model matrix to transform points from local to world space.

        :return: The 4x4 model matrix.
        """
        translation_matrix = glm.translate(glm.mat4(1.0), self.position)
        rotation_matrix = glm.mat4_cast(self.rotation)
        scale_matrix = glm.scale(glm.mat4(1.0), self.scale)
        return glm.mat4(translation_matrix * rotation_matrix * scale_matrix)
