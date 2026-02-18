"""
A module defining the scene data structure.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Tuple, List
import numpy as np
from pyglm import glm

from cs248a_renderer.model.cameras import PerspectiveCamera
from cs248a_renderer.model.volumes import DenseVolume
from cs248a_renderer.model.nerf import NeRF
from cs248a_renderer.model.ray_marcher_config import RayMarcherConfig
from cs248a_renderer.model.scene_object import SceneObject
from cs248a_renderer.model.mesh import Triangle, Mesh
from cs248a_renderer.model.material import PhysicsBasedMaterial


@dataclass
class SingleVolumeScene:
    """A scene containing a single dense volume.

    :param volume: The volume in the scene.
    :param camera: The perspective camera for the scene.
    """

    volume: DenseVolume
    camera: PerspectiveCamera
    ray_marcher_config: RayMarcherConfig
    ambient_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass
class NeRFScene:
    """A scene containing a single NeRF volume.

    :param volume: The NeRF volume in the scene.
    :param camera: The perspective camera for the scene.
    """

    nerf: NeRF
    camera: PerspectiveCamera
    ray_marcher_config: RayMarcherConfig
    ambient_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass
class Scene:
    """A general scene that can hold different types of primitives."""

    root: SceneObject = field(default_factory=lambda: SceneObject(name="root"))
    lookup: dict[str, SceneObject] = field(default_factory=dict)
    camera: PerspectiveCamera = field(default_factory=PerspectiveCamera)
    ambient_color: Tuple[float, float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0, 1.0)
    )

    # Primitives.
    _triangles: List[Triangle] = field(default_factory=list)
    _materials: List[PhysicsBasedMaterial] = field(default_factory=list)
    _volumes: List[DenseVolume] = field(default_factory=list)
    # Volume
    single_volume: DenseVolume | None = None

    def __post_init__(self):
        self.lookup[self.root.name] = self.root

    def __getitem__(self, name: str) -> SceneObject | None:
        """Lookup a SceneObject by name.

        :param name: The name of the SceneObject to lookup.
        :return: The SceneObject if found, else None.
        """
        return self.lookup.get(name, None)

    def add_object(self, obj: SceneObject, parent_name: str = "root"):
        """Add a SceneObject to the scene under the specified parent.

        :param obj: The SceneObject to add.
        :param parent_name: The name of the parent SceneObject.
        """
        parent = self.lookup.get(parent_name)
        if parent is None:
            raise ValueError(f"Parent object '{parent_name}' not found in scene.")
        parent.children.append(obj)
        obj.parent = parent
        self.lookup[obj.name] = obj

    def remove_object(self, obj_name: str):
        """Remove a SceneObject from the scene.

        :param obj_name: The name of the SceneObject to remove.
        """
        obj = self.lookup.get(obj_name)
        if obj is None:
            raise ValueError(f"Object '{obj_name}' not found in scene.")
        # Remove from parent's children list.
        if obj.parent is not None:
            obj.parent.children.remove(obj)

        # Recursively remove all children from lookup.
        def _remove_recursive(o: SceneObject):
            for child in o.children:
                _remove_recursive(child)
            del self.lookup[o.name]

        _remove_recursive(obj)

    def reparent(self, obj_name: str, new_parent_name: str):
        """Reparent a SceneObject to a new parent.

        :param obj_name: The name of the SceneObject to reparent.
        :param new_parent_name: The name of the new parent SceneObject.
        """
        obj = self.lookup.get(obj_name)
        new_parent = self.lookup.get(new_parent_name)

        # Check cyclic parenting.
        current = new_parent
        while current is not None:
            if current == obj:
                return  # No-op to avoid cycle.
            current = current.parent

        if obj is None:
            raise ValueError(f"Object '{obj_name}' not found in scene.")
        if new_parent is None:
            raise ValueError(
                f"New parent object '{new_parent_name}' not found in scene."
            )
        if obj.parent is not None:
            obj.parent.children.remove(obj)
        new_parent.children.append(obj)
        obj.parent = new_parent

    def rename_object(self, old_name: str, new_name: str):
        """Rename a SceneObject in the scene.

        :param old_name: The current name of the SceneObject.
        :param new_name: The new name for the SceneObject.
        """
        obj = self.lookup.get(old_name)
        if obj is None:
            raise ValueError(f"Object '{old_name}' not found in scene.")
        if new_name in self.lookup:
            raise ValueError(f"An object with name '{new_name}' already exists.")
        del self.lookup[old_name]
        obj.name = new_name
        self.lookup[new_name] = obj

    def extract_triangles_with_material(self):
        """Extract all triangles in the scene from SceneObject."""
        self._triangles = []
        self._materials = []
        # Depth-first traversal to extract triangles.
        stack = [self.root]
        while stack:
            current = stack.pop(-1)
            if current is None:
                continue
            if type(current) is Mesh:
                transform = current.get_transform_matrix()
                for triangle in current.triangles:
                    triangle = replace(triangle, material_id=len(self._materials))
                    transformed_triangle = triangle.transform(transform)
                    self._triangles.append(transformed_triangle)
                self._materials.append(current.material)
            stack.extend(current.children)
        return self._triangles, self._materials

    def extract_volumes(self):
        """Extract all dense volumes in the scene from SceneObject."""
        self._volumes = []

        # Depth-first traversal to extract volumes.
        stack = [self.root]
        while stack:
            current = stack.pop(-1)
            if current is None:
                continue
            if type(current) is DenseVolume:
                self._volumes.append(current)
            stack.extend(current.children)
        return self._volumes

    def __repr__(self):
        return self.root.desc()
