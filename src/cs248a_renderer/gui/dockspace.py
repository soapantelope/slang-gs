"""
Volumetric renderer dockspace.
"""

from typing import TypedDict, Unpack, Callable
from imgui_bundle import imgui
from reactivex import Observable
from reactivex.subject import BehaviorSubject, Subject
from slangpy_imgui_bundle.render_targets.dockspace import Dockspace, DockspaceArgs
from slangpy_imgui_bundle.render_targets.menu import Menu, MenuItem, SimpleMenuItem
from slangpy_imgui_bundle.utils.fps_counter import FPSCounter
from slangpy_imgui_bundle.render_targets.render_target import RenderTarget, RenderArgs


class BVHBuildItemArgs(RenderArgs):
    bvh_build_progress: BehaviorSubject[tuple[int, int]]
    on_clicked: Callable[[], None]


class BVHMenuItem(RenderTarget):
    def __init__(self, **kwargs: Unpack[BVHBuildItemArgs]) -> None:
        super().__init__(**kwargs)
        self._bvh_build_progress = kwargs["bvh_build_progress"]
        self._on_clicked = kwargs["on_clicked"]

    def render(self, time: float, delta_time: float) -> None:
        building = self._bvh_build_progress.value[1] != 0
        if imgui.menu_item_simple("Build BVH", enabled=not building):
            self._on_clicked()
        if building:
            current, total = self._bvh_build_progress.value
            imgui.progress_bar(
                fraction=float(current) / float(total),
                overlay=f"Building BVH: {current}/{total}",
            )


class BVHBuildProgressArgs(TypedDict):
    bvh_build_progress: BehaviorSubject[tuple[int, int]]


class BVHBuildProgress(RenderTarget):
    def __init__(self, **kwargs: Unpack[BVHBuildProgressArgs]) -> None:
        super().__init__(**kwargs)
        self._bvh_build_progress = kwargs["bvh_build_progress"]

    def render(self, time: float, delta_time: float) -> None:
        building = self._bvh_build_progress.value[1] != 0
        if building:
            current, total = self._bvh_build_progress.value
            imgui.progress_bar(
                fraction=float(current) / float(total),
                size_arg=(300, -1),
                overlay=f"Building BVH: {current}/{total}",
            )
        else:
            imgui.text("BVH Build Idle")


class SceneStatusArgs(RenderArgs):
    mesh_outdated: BehaviorSubject[bool]


class SceneStatus(RenderTarget):
    def __init__(self, **kwargs: Unpack[SceneStatusArgs]) -> None:
        super().__init__(**kwargs)
        self._mesh_outdated = kwargs["mesh_outdated"]

    def render(self, time: float, delta_time: float) -> None:
        if self._mesh_outdated.value:
            imgui.text_colored((1.0, 0.0, 0.0, 1.0), "Mesh Outdated")
        else:
            imgui.text_colored((0.0, 1.0, 0.0, 1.0), "Mesh Up-to-date")


class FileSubjects(TypedDict):
    on_load_mesh: Subject[None]
    on_load_volume: Subject[None]


class WindowOpenSubjects(TypedDict):
    scene_wizard_open: Subject[None]
    preview_open: BehaviorSubject[bool]
    renderer_open: BehaviorSubject[bool]
    scene_editor_open: BehaviorSubject[bool]


class RendererState(TypedDict):
    render_request: Subject[None]
    mesh_outdated: Observable[bool]
    build_bvh: Subject[None]
    bvh_progress: Observable[tuple[int, int]]


class VolumetricDockspaceArgs(DockspaceArgs):
    file_subjects: FileSubjects
    window_open_subjects: WindowOpenSubjects
    renderer_state: RendererState


class VolumetricDockspace(Dockspace):
    def __init__(self, **kwargs: Unpack[VolumetricDockspaceArgs]) -> None:
        super().__init__(**kwargs)

        self._menu_items = [
            Menu(
                device=self._device,
                adapter=self._adapter,
                name="File",
                children=[
                    Menu(
                        device=self._device,
                        adapter=self._adapter,
                        name="Load",
                        children=[
                            SimpleMenuItem(
                                device=self._device,
                                adapter=self._adapter,
                                name="Mesh",
                                on_clicked=lambda: kwargs["file_subjects"][
                                    "on_load_mesh"
                                ].on_next(None),
                            ),
                            SimpleMenuItem(
                                device=self._device,
                                adapter=self._adapter,
                                name="Volume",
                                on_clicked=lambda: kwargs["file_subjects"][
                                    "on_load_volume"
                                ].on_next(None),
                            ),
                        ],
                    ),
                ],
            ),
            Menu(
                device=self._device,
                adapter=self._adapter,
                name="Views",
                children=[
                    MenuItem(
                        device=self._device,
                        adapter=self._adapter,
                        name="Preview",
                        open=kwargs["window_open_subjects"]["preview_open"],
                        on_open_changed=lambda opened: kwargs["window_open_subjects"][
                            "preview_open"
                        ].on_next(opened),
                    ),
                    MenuItem(
                        device=self._device,
                        adapter=self._adapter,
                        name="Renderer",
                        open=kwargs["window_open_subjects"]["renderer_open"],
                        on_open_changed=lambda opened: kwargs["window_open_subjects"][
                            "renderer_open"
                        ].on_next(opened),
                    ),
                    MenuItem(
                        device=self._device,
                        adapter=self._adapter,
                        name="Scene Editor",
                        open=kwargs["window_open_subjects"]["scene_editor_open"],
                        on_open_changed=lambda opened: kwargs["window_open_subjects"][
                            "scene_editor_open"
                        ].on_next(opened),
                    ),
                ],
            ),
            Menu(
                device=self._device,
                adapter=self._adapter,
                name="Renderer",
                children=[
                    SimpleMenuItem(
                        device=self._device,
                        adapter=self._adapter,
                        name="Render",
                        on_clicked=lambda: kwargs["render_request"].on_next(None),
                    ),
                    BVHMenuItem(
                        device=self._device,
                        adapter=self._adapter,
                        mesh_outdated=kwargs["renderer_state"]["mesh_outdated"],
                        bvh_build_progress=kwargs["renderer_state"]["bvh_progress"],
                        on_clicked=lambda: kwargs["renderer_state"][
                            "build_bvh"
                        ].on_next(None),
                    ),
                ],
            ),
        ]

        self._status_items = [
            FPSCounter(),
            BVHBuildProgress(
                bvh_build_progress=kwargs["renderer_state"]["bvh_progress"]
            ),
            SceneStatus(mesh_outdated=kwargs["renderer_state"]["mesh_outdated"]),
        ]

    def build(self, dockspace_id: int) -> None:
        # Build dock space.
        if not imgui.internal.dock_builder_get_node(dockspace_id):
            imgui.internal.dock_builder_remove_node(dockspace_id)
            main_id = imgui.internal.dock_builder_add_node(dockspace_id)
            # Split the main node into view node, and property node.
            res = imgui.internal.dock_builder_split_node(main_id, imgui.Dir.left, 0.7)
            view_id = res.id_at_dir
            property_id = res.id_at_opposite_dir
            # Split the property node into scene editor and camera controls.
            res = imgui.internal.dock_builder_split_node(property_id, imgui.Dir.up, 0.8)
            scene_editor_id = res.id_at_dir
            camera_controls_id = res.id_at_opposite_dir

            imgui.internal.dock_builder_dock_window("Preview", view_id)
            imgui.internal.dock_builder_dock_window("Renderer", view_id)

            imgui.internal.dock_builder_dock_window("Scene Editor", scene_editor_id)
            imgui.internal.dock_builder_dock_window(
                "Camera Controls", camera_controls_id
            )

            imgui.internal.dock_builder_finish(dockspace_id)
