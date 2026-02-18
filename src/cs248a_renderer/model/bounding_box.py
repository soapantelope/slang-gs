from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from pyglm import glm


@dataclass
class BoundingBox3D:
    """A 3D axis-aligned bounding box defined by minimum and maximum corners.

    :param min: Minimum corner of the bounding box.
    :param max: Maximum corner of the bounding box.
    """

    min: glm.vec3 = field(default_factory=lambda: glm.vec3(float("inf")))
    max: glm.vec3 = field(default_factory=lambda: glm.vec3(float("-inf")))

    def get_this(self) -> Dict:
        return {
            "pMin": self.min.to_list(),
            "pMax": self.max.to_list(),
        }

    @property
    def center(self) -> glm.vec3:
        """Computes the center point of the bounding box."""
        return (self.min + self.max) * 0.5

    @property
    def area(self) -> float:
        """Computes the surface area of the bounding box."""
        d = self.max - self.min
        return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)

    @staticmethod
    def union(box1: BoundingBox3D, box2: BoundingBox3D) -> BoundingBox3D:
        """Computes the union of two bounding boxes.

        :param box1: The first bounding box.
        :param box2: The second bounding box.
        :return: The union bounding box that encompasses both input boxes.
        """
        min_corner = glm.min(box1.min, box2.min)
        max_corner = glm.max(box1.max, box2.max)
        return BoundingBox3D(min=min_corner, max=max_corner)
