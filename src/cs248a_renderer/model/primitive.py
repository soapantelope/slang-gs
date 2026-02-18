from abc import ABC

from cs248a_renderer.model.bounding_box import BoundingBox3D


class Primitive(ABC):
    @property
    def bounding_box(self) -> BoundingBox3D:
        raise NotImplementedError
