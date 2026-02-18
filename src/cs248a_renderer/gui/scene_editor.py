from typing import Unpack
import asyncio
from pathlib import Path
from imgui_bundle import ImVec4, imgui, imgui_ctx, portable_file_dialogs as pfd
from pyglm import glm
from reactivex.subject import BehaviorSubject
from slangpy_imgui_bundle.render_targets.window import Window, WindowArgs
from slangpy_imgui_bundle.utils.file_dialog import async_file_dialog

from cs248a_renderer.model.transforms import Transform3D
from cs248a_renderer.model.scene import NeRFScene, Scene, SingleVolumeScene
from cs248a_renderer.model.scene_object import SceneObject
from cs248a_renderer.model.cameras import PerspectiveCamera
from cs248a_renderer.model.ray_marcher_config import RayMarcherConfig
from cs248a_renderer.model.volumes import DenseVolume
from cs248a_renderer.model.mesh import Mesh
from cs248a_renderer.model.material import FilteringMethod, MaterialField
from cs248a_renderer.view_model.scene_manager import SceneManager


class SceneEditorArgs(WindowArgs):
    scene_manager: SceneManager
    editing_object: BehaviorSubject[SceneObject | None]
    mesh_outdated: BehaviorSubject[bool]


class SceneEditorWindow(Window):
    _scene_manager: SceneManager
    _editing_object: BehaviorSubject[SceneObject | None]
    _dnd_store: dict[int, str]
    _mesh_outdated: BehaviorSubject[bool]

    def __init__(self, **kwargs: Unpack[SceneEditorArgs]) -> None:
        super().__init__(**kwargs)
        self._scene_manager = kwargs["scene_manager"]
        self._editing_object = kwargs["editing_object"]
        self._mesh_outdated = kwargs["mesh_outdated"]
        self._dnd_store = {}

    def render_window(self, time: float, delta_time: float, open: bool | None) -> bool:
        with imgui_ctx.begin("Scene Editor", p_open=open) as window:
            with imgui_ctx.push_item_width(-150):
                # has_volume_scene = self._scene_manager.volume_scene is not None
                # has_nerf_scene = self._scene_manager.nerf_scene is not None

                self._render_scene_camera(self._scene_manager.scene)
                self._render_scene_graph(self._scene_manager.scene.root)
                self._render_single_volume(self._scene_manager.scene)

                # Add new empty SceneObject
                if imgui.button("Add Empty SceneObject"):
                    new_object = SceneObject()
                    self._scene_manager.scene.add_object(new_object)
                    self._mesh_outdated.on_next(True)
                # if not has_volume_scene and not has_nerf_scene:
                #     imgui.text_colored(ImVec4(1.0, 0.0, 0.0, 1.0), "NO SCENE LOADED")
                # else:
                #     if has_volume_scene:
                #         self._render_volume_scene(self._scene_manager.volume_scene)
                #     elif has_nerf_scene:
                #         self._render_nerf_scene(self._scene_manager.nerf_scene)
            return window.opened

    def _render_camera_section(self, camera: PerspectiveCamera, suffix: str) -> None:
        imgui.separator_text(f"{suffix} Camera")
        self._render_transform(camera, f"{suffix} Camera")
        changed, fov = imgui.drag_float(
            f"FOV##{suffix}",
            camera.fov,
            v_speed=0.1,
            v_min=1.0,
            v_max=179.0,
        )
        if changed:
            camera.fov = fov

    def _render_scene_config(
        self, scene: SingleVolumeScene | NeRFScene, suffix: str
    ) -> None:
        imgui.separator_text(f"{suffix} Scene Config")
        changed, ambient_color = imgui.color_edit4(
            f"Ambient Color##{suffix}",
            list(scene.ambient_color),
        )
        if changed:
            scene.ambient_color = (
                ambient_color[0],
                ambient_color[1],
                ambient_color[2],
                ambient_color[3],
            )

    def _render_ray_marcher_config(self, config: RayMarcherConfig, suffix: str) -> None:
        imgui.separator_text(f"{suffix} Ray Marcher Config")
        changed, max_steps = imgui.drag_int(
            f"Max Steps##{suffix}",
            config.max_steps,
            v_speed=1,
            v_min=1,
            v_max=10000,
        )
        if changed:
            config.max_steps = max_steps

        changed, step_size = imgui.drag_float(
            f"Step Size##{suffix}",
            config.step_size,
            v_speed=0.001,
            v_min=0.001,
            v_max=1.0,
        )
        if changed:
            config.step_size = step_size

        changed, density_scale = imgui.drag_float(
            f"Density Scale##{suffix}",
            config.density_scale,
            v_speed=1.0,
            v_min=1.0,
            v_max=1000.0,
        )
        if changed:
            config.density_scale = density_scale

    def _render_scene_camera(self, scene: Scene) -> None:
        self._render_camera_section(scene.camera, "Scene")

    def _render_scene_graph(self, root: SceneObject) -> None:
        imgui.separator_text("Scene Graph")
        self._render_scene_graph_node(root, is_root=True)

    def _render_scene_graph_node(
        self, node: SceneObject, is_root: bool = False
    ) -> None:
        label = f"{node.name}##SceneGraphNode{node.name}"
        imgui.push_id(node.name)
        if imgui.tree_node(label):
            # Handle drag and drop for reparenting
            if imgui.begin_drag_drop_source():
                payload = node.name
                payload_id = id(payload)
                self._dnd_store[payload_id] = payload
                imgui.set_drag_drop_payload_py_id(
                    "SCENE_OBJECT",
                    payload_id,
                )
                imgui.text(f"Reparent {node.name}")
                imgui.end_drag_drop_source()

            if imgui.begin_drag_drop_target():
                payload_id = imgui.accept_drag_drop_payload_py_id("SCENE_OBJECT")
                if payload_id is not None:
                    print(f"Dropping payload id: {payload_id.data_id}")
                    payload = self._dnd_store.get(payload_id.data_id, None)
                    print(f"Dropping on {node.name}")
                    if payload != node.name:
                        self._scene_manager.scene.reparent(payload, node.name)
                        self._mesh_outdated.on_next(True)
                imgui.end_drag_drop_target()

            if not is_root:
                self._render_transform(node, node.name)
                # Render material editor if this is a Mesh
                if isinstance(node, Mesh):
                    self._render_material(node, node.name)
            for child in node.children:
                self._render_scene_graph_node(child)

            if not is_root:
                if imgui.button(f"Delete {node.name}"):
                    self._scene_manager.scene.remove_object(node.name)
                    self._mesh_outdated.on_next(True)
                    imgui.tree_pop()
                    imgui.pop_id()
                    return
            imgui.tree_pop()
        imgui.pop_id()

    def _render_transform(self, node: SceneObject, name: str = "") -> None:
        transform = node.transform
        # Position
        changed, position = imgui.drag_float3(
            f"Position##{name}", transform.position.to_list(), v_speed=0.01
        )
        if changed:
            transform.position = glm.vec3(*position)
            self._mesh_outdated.on_next(True)
        # Rotation
        changed, rotation = imgui.input_float4(
            f"Rotation##{name}",
            [
                transform.rotation.w,
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
            ],
        )
        if changed:
            transform.rotation = glm.quat(
                rotation[0], rotation[1], rotation[2], rotation[3]
            )
            self._mesh_outdated.on_next(True)

        # Scale
        changed, scale = imgui.drag_float3(
            f"Scale##{name}", transform.scale.to_list(), v_speed=0.01
        )
        if changed:
            transform.scale = glm.vec3(*scale)
            self._mesh_outdated.on_next(True)
        # Edit Button
        if imgui.button(f"Edit Transform##{name}"):
            if self._editing_object.value == node:
                self._editing_object.on_next(None)
            else:
                self._editing_object.on_next(node)
                self._mesh_outdated.on_next(True)

    def _render_single_volume(self, scene: Scene) -> None:
        """Render controls for the single_volume in the scene."""
        if scene.single_volume is None:
            return

        imgui.separator_text("Single Volume Properties")
        volume = scene.single_volume

        # Display volume shape
        imgui.text(f"Shape: {volume.shape[0]}x{volume.shape[1]}x{volume.shape[2]}")
        imgui.text(f"Channels: {volume.channels}")

        # Voxel size
        changed, voxel_size = imgui.drag_float(
            "Voxel Size##SingleVolume",
            volume.properties["voxel_size"],
            v_speed=0.001,
            v_min=0.001,
            v_max=1.0,
        )
        if changed:
            volume.properties["voxel_size"] = voxel_size
            self._mesh_outdated.on_next(True)

        # Pivot
        changed, pivot = imgui.drag_float3(
            "Pivot##SingleVolume",
            list(volume.properties["pivot"]),
            v_speed=0.01,
            v_min=0.0,
            v_max=1.0,
        )
        if changed:
            volume.properties["pivot"] = (pivot[0], pivot[1], pivot[2])
            self._mesh_outdated.on_next(True)

        # Display bounding box
        bbox = volume.bounding_box
        imgui.text(f"Bounding Box:")
        imgui.text(f"  Min: ({bbox.min.x:.3f}, {bbox.min.y:.3f}, {bbox.min.z:.3f})")
        imgui.text(f"  Max: ({bbox.max.x:.3f}, {bbox.max.y:.3f}, {bbox.max.z:.3f})")

        # Transform
        self._render_transform(volume, "SingleVolume")

    def _render_material(self, mesh: Mesh, name: str = "") -> None:
        """Render material editing controls for a Mesh."""
        imgui.separator_text(f"Material##{name}")
        material = mesh.material

        # Render albedo field
        self._render_material_field(material.albedo, "Albedo", name, is_color=True)

    def _render_material_field(
        self,
        field: MaterialField,
        field_name: str,
        mesh_name: str,
        is_color: bool = True,
    ) -> None:
        """Render a reusable material field editor component.

        :param field: The MaterialField to edit
        :param field_name: The display name of the field (e.g., "Albedo", "Smoothness")
        :param mesh_name: The name of the mesh (for unique ID generation)
        :param is_color: Whether the uniform value should be displayed as a color picker
        """
        if imgui.tree_node(f"{field_name}##{mesh_name}_{field_name}"):
            # Toggle between uniform and texture
            changed, use_texture = imgui.checkbox(
                f"Use Texture##{mesh_name}_{field_name}", field.use_texture
            )
            if changed:
                field.use_texture = use_texture
                self._mesh_outdated.on_next(True)

            if field.use_texture:
                # Texture settings
                imgui.text(f"Texture Levels: {len(field.textures)}")

                # Filtering method dropdown
                filtering_methods = [
                    "NEAREST",
                    "BILINEAR",
                    "BILINEAR_DISCRETIZED_LEVEL",
                    "TRILINEAR",
                ]
                current_method = field.filtering_method.value
                changed, selected_method = imgui.combo(
                    f"Filtering Method##{mesh_name}_{field_name}",
                    current_method,
                    filtering_methods,
                )
                if changed:
                    field.filtering_method = FilteringMethod(selected_method)
                    self._mesh_outdated.on_next(True)

                # Display texture info
                for i, texture in enumerate(field.textures):
                    imgui.text(f"  Level {i}: {texture.shape[1]}x{texture.shape[0]}")

                # Load texture button
                if imgui.button(f"Load Image##{mesh_name}_{field_name}"):
                    asyncio.create_task(
                        self._load_texture_image(field, field_name, mesh_name)
                    )
            else:
                # Uniform value editor
                if field.uniform_value is not None:
                    if is_color and isinstance(field.uniform_value, glm.vec3):
                        # Color picker for vec3
                        changed, color = imgui.color_edit3(
                            f"{field_name} Color##{mesh_name}_{field_name}",
                            [
                                field.uniform_value.x,
                                field.uniform_value.y,
                                field.uniform_value.z,
                            ],
                        )
                        if changed:
                            field.uniform_value = glm.vec3(color[0], color[1], color[2])
                            self._mesh_outdated.on_next(True)
                    else:
                        # Generic text display for other types
                        imgui.text(f"{field_name}: {field.uniform_value}")

            imgui.tree_pop()

    async def _load_texture_image(
        self, field: MaterialField, field_name: str, mesh_name: str
    ) -> None:
        """Load a texture image from file dialog.

        :param field: The MaterialField to load texture into
        :param field_name: The display name of the field
        :param mesh_name: The name of the mesh
        """
        files = await async_file_dialog(
            title=f"Load {field_name} Texture",
            default_path=str(Path.cwd()),
            filters=["Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff"],
            options=pfd.opt.none,
        )
        if files:
            image_path = Path(files[0])
            try:
                field.load_texture_from_image(image_path)
                self._mesh_outdated.on_next(True)
            except Exception as e:
                print(f"Error loading texture: {e}")
