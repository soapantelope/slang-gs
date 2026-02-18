"""
Preview window.
"""

import logging
from typing import Tuple, Unpack
from imgui_bundle import ImVec2, ImVec4, imgui, imgui_ctx, imguizmo
from pyglm import glm
from reactivex import Observable
from reactivex.subject import BehaviorSubject
import math
import slangpy as spy
import numpy as np
from slangpy_imgui_bundle.render_targets.window import Window, WindowArgs

from cs248a_renderer.model.scene_object import SceneObject
from cs248a_renderer.model.transforms import Transform3D
from cs248a_renderer.renderer.wireframe_renderer import WireframeRenderer
from cs248a_renderer.renderer.mesh_renderer import MeshRenderer
from cs248a_renderer.view_model.scene_manager import SceneManager
from cs248a_renderer.model.cameras import PerspectiveCamera


logger = logging.getLogger(__name__)
gizmo = imguizmo.im_guizmo

GIZMO_OPS = [
    gizmo.OPERATION.translate,
    gizmo.OPERATION.rotate,
    gizmo.OPERATION.scale,
]
GIZMO_MODES = [
    gizmo.MODE.world,
    gizmo.MODE.local,
]


class PreviewWindowArgs(WindowArgs):
    scene_manager: SceneManager
    canvas_size: Observable[Tuple[int, int]]
    editing_object: Observable[SceneObject | None]
    mesh_outdated: BehaviorSubject[bool]


class PreviewWindow(Window):
    """Preview window for rendering the volumetric scene."""

    _viewport_size: ImVec2 = ImVec2(1.0, 1.0)

    _show_camera_controls: bool = True
    # When True, the preview camera will follow the scene's active camera position and rotation.
    _follow_scene_camera: bool = False
    _3d_cursor_pos: glm.vec3 = glm.vec3(0.0, 0.0, 0.0)
    _camera_distance: float = 5.0

    _viewport_fov: float = 50.0
    _viewport_camera: PerspectiveCamera = PerspectiveCamera(
        fov=_viewport_fov,
        transform=Transform3D(
            position=glm.vec3(0.0, 0.0, _camera_distance),
            rotation=glm.quat(1.0, 0.0, 0.0, 0.0),
            scale=glm.vec3(1.0, 1.0, 1.0),
        ),
    )
    _editing_object: SceneObject | None
    _mesh_outdated: BehaviorSubject[bool]
    _gizmo_op: int = 0
    _gizmo_mode: int = 0

    def __init__(self, **kwargs: Unpack[PreviewWindowArgs]) -> None:
        super().__init__(**kwargs)
        self._scene_manager = kwargs["scene_manager"]
        self._mesh_outdated = kwargs["mesh_outdated"]
        # Keep track of the canvas size (width, height) from the observable so
        # we can adapt the camera FOV depending on the viewport vs canvas aspect.
        self._canvas_size: tuple[int, int] | None = None
        # Base vertical FOV (degrees) used when the viewport is wide enough.
        self._base_fov: float = float(self._viewport_fov)

        # Require device.
        device = kwargs.get("device")
        if device is None:
            raise ValueError("PreviewWindow requires a device.")
        self._device = device
        # Require adapater.
        adapter = kwargs.get("adapter")
        if adapter is None:
            raise ValueError("PreviewWindow requires an adapter.")
        self._adapter = adapter

        # Setup volume render texture.
        self._volume_render_tex = self._device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba8_unorm,
            width=int(self._viewport_size.x),
            height=int(self._viewport_size.y),
            usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
            label="preview_window_volume_render_texture",
            data=np.zeros(
                (int(self._viewport_size.y), int(self._viewport_size.x), 4),
                dtype=np.uint8,
            ),
        )
        self._volume_render_tex_id = self._adapter.register_texture(
            self._volume_render_tex
        )

        # Setup render texture.
        self._wireframe_render_tex = self._device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba8_unorm,
            width=int(self._viewport_size.x),
            height=int(self._viewport_size.y),
            usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource,
            label="preview_window_render_texture",
            data=np.zeros(
                (int(self._viewport_size.y), int(self._viewport_size.x), 4),
                dtype=np.uint8,
            ),
        )
        self._wireframe_render_tex_id = self._adapter.register_texture(
            self._wireframe_render_tex
        )

        # Setup renderer.
        self._wireframe_renderer = WireframeRenderer(
            device=self._device, render_target=self._wireframe_render_tex
        )
        self._mesh_renderer = MeshRenderer(
            device=self._device, render_target=self._wireframe_render_tex
        )

        # Update canvas size.
        kwargs["canvas_size"].subscribe(self._wireframe_renderer.update_canvas_size)
        kwargs["canvas_size"].subscribe(self._mesh_renderer.update_canvas_size)
        # Also store the latest canvas size locally so we can compute aspect ratios
        # and adjust the viewport camera FOV when needed.
        kwargs["canvas_size"].subscribe(self._on_canvas_size)

        # Editing object.
        def _on_editing_object(obj: SceneObject | None) -> None:
            self._editing_object = obj

        kwargs["editing_object"].subscribe(_on_editing_object)

        # Interaction state for orbiting/panning/zooming.
        self._last_mouse_pos: tuple[float, float] | None = None
        # Size (in pixels) of the ImGuizmo view manipulator square.
        self._gizmo_size: float = 128.0
        # Initialize spherical coordinates (radius, theta, phi) from current camera.
        self._orbit_radius: float = 0.0
        self._orbit_theta: float = 0.0
        self._orbit_phi: float = 0.0
        self._update_spherical_from_camera()

    def _on_canvas_size(self, size: Tuple[int, int]) -> None:
        """Callback for canvas size observable events.

        Stores a validated (width, height) pair in `self._canvas_size`.
        The observable may emit different numeric types, so coerce to int
        and ignore malformed values.
        """
        # size may be a tuple-like (w, h)
        w = int(size[0])
        h = int(size[1])
        if w > 0 and h > 0:
            self._canvas_size = (w, h)

    def _adjust_vertical_fov(self, viewport_w: float, viewport_h: float) -> None:
        """Adjust the camera's vertical FOV based on the viewport and canvas aspect.

        Rules:
        - If viewport aspect >= canvas aspect: keep base vertical FOV.
        - If viewport aspect < canvas aspect: increase vertical FOV such that
          horizontal coverage matches what the base FOV would cover at the
          canvas aspect ratio.
        """
        try:
            viewport_aspect = float(viewport_w) / float(viewport_h)
        except Exception:
            viewport_aspect = 1.0

        if self._canvas_size is None:
            # No canvas size known yet — use base FOV.
            self._viewport_camera.fov = self._base_fov
            return

        canvas_w, canvas_h = self._canvas_size
        if canvas_h <= 0 or canvas_w <= 0:
            self._viewport_camera.fov = self._base_fov
            return

        canvas_aspect = float(canvas_w) / float(canvas_h)
        if viewport_aspect >= canvas_aspect:
            # Viewport is wider (or equal) than canvas: keep base vertical FOV.
            self._viewport_camera.fov = self._base_fov
            return

        # Viewport is narrower than canvas. Compute the horizontal FOV covered
        # by the base vertical FOV at the canvas aspect, then solve for the
        # vertical FOV required at the viewport aspect to preserve that
        # horizontal coverage.
        half_base_rad = math.radians(self._base_fov) / 2.0
        # horizontal FOV at canvas aspect
        half_horiz = math.atan(math.tan(half_base_rad) * canvas_aspect)
        horiz_fov = 2.0 * half_horiz
        # Solve for new vertical FOV at viewport aspect:
        half_new_vert = math.atan(math.tan(horiz_fov / 2.0) / viewport_aspect)
        new_fov_deg = math.degrees(2.0 * half_new_vert)
        self._viewport_camera.fov = float(new_fov_deg)

    def _set_camera_from_position(
        self, camera_pos: glm.vec3, update_radius: bool = True
    ) -> None:
        """Set camera position and rotation to look at `_3d_cursor_pos` from camera_pos.

        If update_radius is True, update internal orbit radius and camera distance.
        """
        view_mat = glm.lookAt(camera_pos, self._3d_cursor_pos, glm.vec3(0.0, 1.0, 0.0))
        camera_transform = glm.inverse(view_mat)
        rotation_mat = glm.mat3(camera_transform)
        rotation_quat = glm.quat_cast(rotation_mat)
        self._viewport_camera.transform.position = camera_pos
        self._viewport_camera.transform.rotation = rotation_quat
        if update_radius:
            self._orbit_radius = float(glm.length(camera_pos - self._3d_cursor_pos))
            self._camera_distance = float(self._orbit_radius)

    def _camera_pos_from_spherical(
        self, r: float, phi: float, theta: float
    ) -> glm.vec3:
        """Return a glm.vec3 camera offset from the cursor using spherical coords.

        r: radius (distance from cursor)
        phi: inclination (0..pi)
        theta: azimuth
        """
        x = r * math.sin(phi) * math.sin(theta)
        y = r * math.cos(phi)
        z = r * math.sin(phi) * math.cos(theta)
        return glm.vec3(x, y, z)

    def _update_spherical_from_camera(self, camera_pos: glm.vec3 | None = None) -> None:
        """Update self._orbit_radius, _orbit_theta, _orbit_phi from camera position.

        If camera_pos is None, use the current `self._viewport_camera.transform.position`.
        """
        if camera_pos is None:
            camera_pos = self._viewport_camera.transform.position
        dir_vec = camera_pos - self._3d_cursor_pos
        r = float(glm.length(dir_vec))
        self._orbit_radius = r
        # azimuth (theta) and inclination (phi)
        self._orbit_theta = math.atan2(float(dir_vec.x), float(dir_vec.z))
        if r != 0.0:
            self._orbit_phi = math.acos(float(dir_vec.y) / r)
        else:
            self._orbit_phi = math.pi / 2.0

    def render_window(self, time: float, delta_time: float, open: bool | None) -> bool:
        if self._show_camera_controls:
            self._render_camera_controls()

        imgui.set_next_window_size(
            ImVec2(800.0, 600.0),
            imgui.Cond_.first_use_ever.value,
        )
        imgui.set_next_window_size_constraints(
            ImVec2(300, 200),
            ImVec2(imgui.FLT_MAX, imgui.FLT_MAX),
        )
        window_flags = imgui.WindowFlags_.menu_bar.value
        with imgui_ctx.begin("Preview", p_open=open, flags=window_flags) as window:
            # Menu bar.
            self._render_menu_bar()

            # Get available content size.
            avail_size = imgui.get_content_region_avail()
            if avail_size != self._viewport_size:
                self._resize_viewport(avail_size)

            # Check for zero size.
            if avail_size.x <= 0.0 or avail_size.y <= 0.0:
                return window.opened

            # Disable window move when interacting with the viewport.
            internal_window = imgui.internal.get_current_window()
            if imgui.is_window_hovered() and imgui.is_mouse_hovering_rect(
                internal_window.inner_rect.min, internal_window.inner_rect.max
            ):
                internal_window.flags |= imgui.WindowFlags_.no_move.value
            else:
                internal_window.flags &= ~imgui.WindowFlags_.no_move.value

            # Get view and projection matrices from the viewport camera.
            if self._follow_scene_camera:
                view_matrix = self._scene_manager.scene.camera.view_matrix()
                self._base_fov = self._scene_manager.scene.camera.fov + 5.0
                # if self._scene_manager.volume_scene is not None:
                #     view_matrix = self._scene_manager.volume_scene.camera.view_matrix()
                #     self._base_fov = self._scene_manager.volume_scene.camera.fov + 5.0
                # elif self._scene_manager.nerf_scene is not None:
                #     view_matrix = self._scene_manager.nerf_scene.camera.view_matrix()
                #     self._base_fov = self._scene_manager.nerf_scene.camera.fov + 5.0
            else:
                view_matrix = self._viewport_camera.view_matrix()
                self._base_fov = self._viewport_fov

            # Adjust vertical FOV based on viewport vs canvas aspect ratios.
            self._adjust_vertical_fov(avail_size.x, avail_size.y)

            projection_matrix = self._viewport_camera.projection_matrix(
                int(avail_size.x), int(avail_size.y)
            )

            self._wireframe_renderer.clear_render_target()
            self._mesh_renderer.clear_depth_target()
            self._mesh_renderer.render_scene_mesh(
                scene=self._scene_manager.scene,
                view_mat=view_matrix,
                proj_mat=projection_matrix,
            )
            self._wireframe_renderer.render_camera(
                camera=self._scene_manager.scene.camera,
                view_mat=view_matrix,
                proj_mat=projection_matrix,
            )
            self._wireframe_renderer.render_scene_bounding_box(
                scene=self._scene_manager.scene,
                view_mat=view_matrix,
                proj_mat=projection_matrix,
            )
            # Render single_volume bounding box if it exists
            if self._scene_manager.scene.single_volume is not None:
                self._wireframe_renderer.render_bounding_box(
                    bounding_box=(
                        self._scene_manager.scene.single_volume.bounding_box.min,
                        self._scene_manager.scene.single_volume.bounding_box.max,
                    ),
                    model_mat=self._scene_manager.scene.single_volume.get_transform_matrix(),
                    view_mat=view_matrix,
                    proj_mat=projection_matrix,
                )
            # if self._scene_manager.volume_scene is not None:
            #     # Render preview wireframe.
            #     self._wireframe_renderer.clear_render_target()
            #     self._wireframe_renderer.render_bounding_box(
            #         transform=self._scene_manager.volume_scene.volume.transform,
            #         bounding_box=self._scene_manager.volume_scene.volume.bounding_box,
            #         view_mat=view_matrix,
            #         proj_mat=projection_matrix,
            #     )
            #     self._wireframe_renderer.render_camera(
            #         camera=self._scene_manager.volume_scene.camera,
            #         view_mat=view_matrix,
            #         proj_mat=projection_matrix,
            #     )
            #     # Render volume.
            #     self._volume_renderer.render(
            #         scene=self._scene_manager.volume_scene,
            #         view_mat=view_matrix,
            #         fov=self._viewport_camera.fov,
            #     )
            # elif self._scene_manager.nerf_scene is not None:
            #     # Render preview wireframe.
            #     self._wireframe_renderer.clear_render_target()
            #     self._wireframe_renderer.render_bounding_box(
            #         transform=self._scene_manager.nerf_scene.nerf.transform,
            #         bounding_box=self._scene_manager.nerf_scene.nerf.bounding_box,
            #         view_mat=view_matrix,
            #         proj_mat=projection_matrix,
            #     )
            #     self._wireframe_renderer.render_camera(
            #         camera=self._scene_manager.nerf_scene.camera,
            #         view_mat=view_matrix,
            #         proj_mat=projection_matrix,
            #     )

            # Get cursor position before rendering the image (this will be the image's top-left corner)
            image_pos = imgui.get_cursor_screen_pos()

            # Interaction: handle zoom/rotate/pan while avoiding the gizmo
            # view_manipulate region and flipping Y axis for drag behaviour.
            # Handle viewport interaction (zoom / pan / rotate) and return whether
            # the user is interacting so the gizmo can be disabled while dragging.
            if not self._follow_scene_camera:
                interacting = self._process_viewport_interaction(image_pos, avail_size)
            else:
                interacting = False

            cursor_pos = imgui.get_cursor_pos()

            # Render volume.
            # if self._scene_manager.volume_scene is not None:
            #     imgui.set_cursor_pos(cursor_pos)
            #     imgui.image(self._volume_render_tex_id, avail_size, (0, 1), (1, 0))

            # Render viewport.
            imgui.set_cursor_pos(cursor_pos)
            imgui.image(self._wireframe_render_tex_id, avail_size)

            # Render gizmo (after image, so we know the exact position and size).
            # Disable camera updates from gizmo while the user is interacting
            # directly with the viewport (so mouse drag/pan/zoom wins).
            self._render_gizmo(
                view_matrix,
                projection_matrix,
                image_pos,
                avail_size,
                allow_camera_update=not interacting and not self._follow_scene_camera,
            )

            return window.opened

    def _resize_viewport(self, new_size: ImVec2) -> None:
        logger.debug(f"Preview window available size changed to {new_size}")
        # Check for zero size.
        if new_size.x <= 0.0 or new_size.y <= 0.0:
            return
        # Unregister old texture.
        self._adapter.unregister_texture(self._wireframe_render_tex_id)
        self._adapter.unregister_texture(self._volume_render_tex_id)
        # Create new texture.
        self._volume_render_tex = self._device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba8_unorm,
            width=int(new_size.x),
            height=int(new_size.y),
            usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
            label="preview_window_volume_render_texture",
            data=np.zeros((int(new_size.y), int(new_size.x), 4), dtype=np.uint8),
        )
        self._volume_render_tex_id = self._adapter.register_texture(
            self._volume_render_tex
        )
        self._wireframe_render_tex = self._device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba8_unorm,
            width=int(new_size.x),
            height=int(new_size.y),
            usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource,
            label="preview_window_render_texture",
            data=np.zeros((int(new_size.y), int(new_size.x), 4), dtype=np.uint8),
        )
        self._wireframe_render_tex_id = self._adapter.register_texture(
            self._wireframe_render_tex
        )
        self._wireframe_renderer.update_render_target(self._wireframe_render_tex)
        self._mesh_renderer.update_render_target(self._wireframe_render_tex)
        self._viewport_size = new_size

    def _render_menu_bar(self) -> None:
        if imgui.begin_menu_bar():
            # Toggle camera control window.
            _, self._show_camera_controls = imgui.menu_item(
                "Camera Controls",
                "",
                p_selected=self._show_camera_controls,
            )
            # Gizmo operation combo.
            imgui.push_item_width(100)
            _, self._gizmo_op = imgui.combo(
                "Operation",
                self._gizmo_op,
                [op.name for op in GIZMO_OPS],
            )
            # Gizmo mode combo.
            _, self._gizmo_mode = imgui.combo(
                "Mode",
                self._gizmo_mode,
                [mode.name for mode in GIZMO_MODES],
            )
            imgui.pop_item_width()

            imgui.end_menu_bar()

    def _render_gizmo(
        self,
        view_matrix: glm.mat4x4,
        projection_matrix: glm.mat4x4,
        image_pos: ImVec2,
        image_size: ImVec2,
        allow_camera_update: bool = True,
    ) -> None:
        """Render ImGuizmo gizmos including grid and view manipulator."""
        gizmo.begin_frame()
        gizmo.set_drawlist()
        gizmo.set_rect(
            image_pos.x,
            image_pos.y,
            image_size.x,
            image_size.y,
        )
        # Draw grid.
        gizmo_identity = gizmo.Matrix16(np.identity(4).flatten().tolist())
        gizmo_view = gizmo.Matrix16(np.array(view_matrix).T.flatten().tolist())
        gizmo_proj = gizmo.Matrix16(np.array(projection_matrix).T.flatten().tolist())
        gizmo.draw_grid(
            view=gizmo_view,
            projection=gizmo_proj,
            matrix=gizmo_identity,
            grid_size=10.0,
        )

        # Editing object gizmo.
        if self._editing_object is not None:
            # Prepare matrices.
            obj_matrix = gizmo.Matrix16(
                np.array(self._editing_object.get_transform_matrix())
                .T.flatten()
                .tolist()
            )
            # Manipulate.
            modified = gizmo.manipulate(
                view=gizmo_view,
                projection=gizmo_proj,
                operation=GIZMO_OPS[self._gizmo_op],
                mode=GIZMO_MODES[self._gizmo_mode],
                object_matrix=obj_matrix,
            )
            if modified:
                # Update editing transform from manipulated matrix.
                # Matrix16.values is laid out the same way the demo expects (row-major
                # flattened). Read it back as a 4x4 row-major array (no transpose)
                # and extract translation from the last row (indices [3,0..2]).
                manipulated_arr = np.array(obj_matrix.values).reshape((4, 4)).T
                # Build a glm mat4x4 directly from the row-major array. PyGLM will
                # interpret the array correctly when passed as rows.
                gizmo_transform = glm.mat4x4(manipulated_arr)
                if self._editing_object.parent is not None:
                    parent_inv = glm.inverse(
                        glm.mat4x4(self._editing_object.parent.get_transform_matrix())
                    )
                    gizmo_transform = parent_inv @ gizmo_transform
                position = glm.vec3(
                    gizmo_transform[3, 0],
                    gizmo_transform[3, 1],
                    gizmo_transform[3, 2],
                )
                rotation_mat = glm.mat3(gizmo_transform)
                rotation_quat = glm.normalize(glm.quat_cast(rotation_mat))
                self._editing_object.transform.position = position
                self._editing_object.transform.rotation = rotation_quat
                self._editing_object.transform.scale = glm.vec3(
                    glm.length(rotation_mat[0]),
                    glm.length(rotation_mat[1]),
                    glm.length(rotation_mat[2]),
                )
                self._mesh_outdated.on_next(True)

        # Draw view manipulator (positioned at top-right of the image).
        image_right = image_pos.x + image_size.x
        image_top = image_pos.y
        gizmo.view_manipulate(
            view=gizmo_view,
            length=self._camera_distance,
            position=ImVec2(image_right - self._gizmo_size, image_top),
            size=ImVec2(self._gizmo_size, self._gizmo_size),
            background_color=0x10101010,
        )
        # Update viewport camera from gizmo view matrix if allowed.
        if allow_camera_update:
            gizmo_view_glm = glm.mat4x4(gizmo_view.values.reshape((4, 4)).T)
            # Extract camera transform from view matrix.
            camera_transform = glm.inverse(gizmo_view_glm)
            rotation_mat = glm.mat3(camera_transform)
            rotation_quat = glm.quat_cast(rotation_mat)
            camera_pos = self._3d_cursor_pos + (
                rotation_mat @ glm.vec3(0.0, 0.0, self._camera_distance)
            )
            self._viewport_camera.transform.position = camera_pos
            self._viewport_camera.transform.rotation = rotation_quat
            # Keep internal spherical representation in sync with gizmo camera.
            self._update_spherical_from_camera(camera_pos)

    def _process_viewport_interaction(
        self, image_pos: ImVec2, avail_size: ImVec2
    ) -> bool:
        """Handle interactions inside the viewport image region.

        Returns True if the user is interacting (dragging) so callers can
        disable other camera updates like gizmo-driven camera changes.
        """
        io = imgui.get_io()
        mouse_pos = imgui.get_mouse_pos()

        # Image rect (screen coords)
        image_min_x = image_pos.x
        image_min_y = image_pos.y
        image_max_x = image_pos.x + avail_size.x
        image_max_y = image_pos.y + avail_size.y

        # Gizmo view_manipulate rect (top-right corner of image)
        gizmo_size = float(self._gizmo_size)
        gizmo_min_x = image_max_x - gizmo_size
        gizmo_min_y = image_min_y
        gizmo_max_x = gizmo_min_x + gizmo_size
        gizmo_max_y = gizmo_min_y + gizmo_size

        def _point_in_rect(
            px: float, py: float, xmin: float, ymin: float, xmax: float, ymax: float
        ) -> bool:
            return px >= xmin and px <= xmax and py >= ymin and py <= ymax

        mouse_x = float(mouse_pos.x)
        mouse_y = float(mouse_pos.y)
        over_image = _point_in_rect(
            mouse_x, mouse_y, image_min_x, image_min_y, image_max_x, image_max_y
        )
        over_gizmo = _point_in_rect(
            mouse_x, mouse_y, gizmo_min_x, gizmo_min_y, gizmo_max_x, gizmo_max_y
        )

        # Determine if user is interacting (middle button down over image and not over gizmo)
        pan_rotate_down = imgui.is_mouse_down(imgui.MouseButton_.middle.value)
        interacting = pan_rotate_down and over_image and not over_gizmo

        # Only interact when mouse is over the image and not over the gizmo.
        if over_image and not over_gizmo:
            # Zoom (wheel)
            wheel = io.mouse_wheel
            if wheel != 0.0:
                factor = 0.9 ** float(wheel)
                self._orbit_radius = max(0.5, min(50.0, self._orbit_radius * factor))
                # update camera position from spherical coordinates
                r = self._orbit_radius
                camera_pos = self._3d_cursor_pos + self._camera_pos_from_spherical(
                    r, self._orbit_phi, self._orbit_theta
                )
                self._set_camera_from_position(camera_pos)

            if pan_rotate_down:
                if self._last_mouse_pos is None:
                    self._last_mouse_pos = (mouse_x, mouse_y)

                dx = mouse_x - self._last_mouse_pos[0]
                # flip Y axis: invert dy
                dy = -(mouse_y - self._last_mouse_pos[1])
                self._last_mouse_pos = (mouse_x, mouse_y)

                if imgui.is_key_down(imgui.Key.left_shift):
                    # pan
                    forward = glm.normalize(
                        self._3d_cursor_pos - self._viewport_camera.transform.position
                    )
                    right = glm.normalize(glm.cross(forward, glm.vec3(0.0, 1.0, 0.0)))
                    up = glm.normalize(glm.cross(right, forward))
                    pan_sens = 0.002 * float(self._orbit_radius)
                    delta = -right * (dx * pan_sens) - up * (dy * pan_sens)
                    self._3d_cursor_pos = self._3d_cursor_pos + delta
                    new_cam_pos = self._viewport_camera.transform.position + delta
                    # update position and rotation, but keep radius consistent
                    self._set_camera_from_position(new_cam_pos, update_radius=False)
                else:
                    # rotate orbit
                    rot_sens = 0.005
                    self._orbit_theta -= dx * rot_sens
                    self._orbit_phi += dy * rot_sens
                    self._orbit_phi = max(0.01, min(math.pi - 0.01, self._orbit_phi))
                    r = self._orbit_radius
                    camera_pos = self._3d_cursor_pos + self._camera_pos_from_spherical(
                        r, self._orbit_phi, self._orbit_theta
                    )
                    logger.debug(f"Orbit camera pos: {camera_pos}")
                    self._set_camera_from_position(camera_pos)
            else:
                self._last_mouse_pos = None

        return interacting

    def _render_camera_controls(self) -> None:
        imgui.set_next_window_size(
            ImVec2(400.0, 400.0),
            imgui.Cond_.first_use_ever.value,
        )
        imgui.set_next_window_size_constraints(
            ImVec2(400, 200),
            ImVec2(imgui.FLT_MAX, imgui.FLT_MAX),
        )
        with imgui_ctx.begin("Camera Controls", self._show_camera_controls) as window:
            # Reserve with for labels.
            with imgui_ctx.push_item_width(-150):
                _, self._follow_scene_camera = imgui.checkbox(
                    "Follow Scene Camera", self._follow_scene_camera
                )

                if not self._follow_scene_camera:
                    # Camera distance slider.
                    changed, new_distance = imgui.slider_float(
                        "Distance",
                        self._camera_distance,
                        0.5,
                        50.0,
                        flags=imgui.SliderFlags_.logarithmic.value,
                    )
                    if changed:
                        self._camera_distance = new_distance

                    # Viewport FOV.
                    changed, new_fov = imgui.drag_float(
                        "Viewport FOV",
                        self._viewport_fov,
                        v_speed=0.1,
                        v_min=1.0,
                        v_max=179.0,
                    )
                    if changed:
                        self._viewport_fov = new_fov

                    # 3D cursor position inputs.
                    changed, new_cursor_pos = imgui.drag_float3(
                        "3D Cursor Position",
                        self._3d_cursor_pos.to_list(),
                        0.02,
                        -10.0,
                        10.0,
                    )
                    if changed:
                        self._3d_cursor_pos = glm.vec3(new_cursor_pos)

                    # Reset 3d cursor position button.
                    if imgui.button("Reset 3D Cursor Position"):
                        self._3d_cursor_pos = glm.vec3(0.0, 0.0, 0.0)

                    # Update scene camera to follow viewport camera.
                    if imgui.button("Snap Scene Camera to Viewport Camera"):
                        # if self._scene_manager.volume_scene is not None:
                        #     scene_camera = self._scene_manager.volume_scene.camera
                        # elif self._scene_manager.nerf_scene is not None:
                        #     scene_camera = self._scene_manager.nerf_scene.camera
                        scene_camera = self._scene_manager.scene.camera
                        scene_camera.transform.position = glm.vec3(
                            self._viewport_camera.transform.position
                        )
                        scene_camera.transform.rotation = glm.quat(
                            self._viewport_camera.transform.rotation
                        )
                else:
                    imgui.text("Camera is following the scene's active camera.")

            self._show_camera_controls = window.opened
