"""
Preview renderer for the volumetric scene.
"""

from typing import Tuple
import slangpy as spy
import numpy as np
from pyglm import glm

from cs248a_renderer.model.transforms import Transform3D
from cs248a_renderer.model.cameras import PerspectiveCamera
from cs248a_renderer.model.scene import Scene
from cs248a_renderer.model.mesh import Mesh


class WireframeRenderer:
    """Renderer for previewing the volumetric scene."""

    _device: spy.Device
    _render_target: spy.Texture
    _canvas_size: Tuple[int, int]

    def __init__(self, device: spy.Device, render_target: spy.Texture) -> None:
        self._device = device
        self._render_target = render_target

        self._input_layout = self._device.create_input_layout(
            input_elements=[
                {
                    "semantic_name": "POSITION",
                    "semantic_index": 0,
                    "format": spy.Format.rgb32_float,
                }
            ],
            vertex_streams=[{"stride": 12}],
        )
        self._program = self._device.load_program(
            "gui/wireframe.slang", ["vertexMain", "fragmentMain"]
        )
        self._pipeline = self._device.create_render_pipeline(
            program=self._program,
            input_layout=self._input_layout,
            primitive_topology=spy.PrimitiveTopology.line_list,
            targets=[
                {
                    "format": spy.Format.rgba8_unorm,
                }
            ],
        )

    def update_render_target(self, render_target: spy.Texture) -> None:
        """Update the render target texture.

        :param render_target: The new render target texture.
        """
        self._render_target = render_target

    def update_canvas_size(self, size: Tuple[int, int]) -> None:
        """Update the canvas size.

        :param size: The new canvas size (width, height).
        """
        self._canvas_size = size

    def clear_render_target(self) -> None:
        """Clear the render target texture."""
        command_encoder = self._device.create_command_encoder()
        with command_encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": self._render_target.create_view({}),
                        "load_op": spy.LoadOp.clear,
                        "clear_value": (0.0, 0.0, 0.0, 0.0),
                        "store_op": spy.StoreOp.store,
                    }
                ]
            }
        ):
            pass
        self._device.submit_command_buffer(command_encoder.finish())

    def render_scene_bounding_box(
        self,
        scene: Scene,
        view_mat: glm.mat4,
        proj_mat: glm.mat4,
    ):
        # Render the bounding box of all the objects in the scene.
        stack = [scene.root]
        while stack:
            obj = stack.pop()
            if type(obj) == Mesh:
                mesh: Mesh = obj  # type: ignore
                bbox = mesh.bounding_box
                self.render_bounding_box(
                    bounding_box=(bbox.min, bbox.max),
                    model_mat=mesh.get_transform_matrix(),
                    view_mat=view_mat,
                    proj_mat=proj_mat,
                )
            for child in obj.children:
                stack.append(child)

    def render_bounding_box(
        self,
        bounding_box: Tuple[glm.vec3, glm.vec3],
        model_mat: glm.mat4,
        view_mat: glm.mat4,
        proj_mat: glm.mat4,
    ):
        command_encoder = self._device.create_command_encoder()
        with command_encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": self._render_target.create_view({}),
                        "load_op": spy.LoadOp.load,
                        "store_op": spy.StoreOp.store,
                    }
                ]
            }
        ) as pass_encoder:
            # Get the wire frame for the volume bounding box.
            bbox_min, bbox_max = bounding_box
            vertices = []
            for x in [bbox_min.x, bbox_max.x]:
                for y in [bbox_min.y, bbox_max.y]:
                    for z in [bbox_min.z, bbox_max.z]:
                        vertices.append([x, y, z])
            vert_arr = np.array(vertices, dtype=np.float32)
            idx_arr = np.array(
                [
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [1, 3],
                    [1, 5],
                    [2, 3],
                    [2, 6],
                    [3, 7],
                    [4, 5],
                    [4, 6],
                    [5, 7],
                    [6, 7],
                ],
                dtype=np.uint32,
            )

            vbo = self._device.create_buffer(
                usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
                label="wireframe_vbo",
                data=vert_arr,
            )
            ibo = self._device.create_buffer(
                usage=spy.BufferUsage.index_buffer | spy.BufferUsage.shader_resource,
                label="wireframe_ibo",
                data=idx_arr,
            )

            root = pass_encoder.bind_pipeline(self._pipeline)
            root_cursor = spy.ShaderCursor(root)
            root_cursor["uniforms"]["modelMatrix"].write(model_mat)
            root_cursor["uniforms"]["viewMatrix"].write(view_mat)
            root_cursor["uniforms"]["projMatrix"].write(proj_mat)
            root_cursor["uniforms"]["color"].write(
                [255.0 / 255.0, 141.0 / 255.0, 40.0 / 255.0, 1.0]
            )

            pass_encoder.set_render_state(
                {
                    "viewports": [
                        spy.Viewport.from_size(
                            self._render_target.width, self._render_target.height
                        )
                    ],
                    "scissor_rects": [
                        spy.ScissorRect.from_size(
                            self._render_target.width, self._render_target.height
                        )
                    ],
                    "vertex_buffers": [vbo],
                    "index_buffer": ibo,
                    "index_format": spy.IndexFormat.uint32,
                }
            )
            pass_encoder.draw_indexed({"vertex_count": idx_arr.shape[0] * 2})
        self._device.submit_command_buffer(command_encoder.finish())

    def render_camera(
        self,
        camera: PerspectiveCamera,
        view_mat: glm.mat4,
        proj_mat: glm.mat4,
    ):
        command_encoder = self._device.create_command_encoder()
        with command_encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": self._render_target.create_view({}),
                        "load_op": spy.LoadOp.load,
                        "store_op": spy.StoreOp.store,
                    }
                ]
            }
        ) as pass_encoder:
            # Render camera frustum as wireframe.
            fov = camera.fov
            w, h = self._canvas_size
            fp_height = np.tan(np.radians(fov) / 2.0)
            fp_width = fp_height * (w / h)
            vert_arr = np.array(
                [
                    [0.0, 0.0, 0.0],  # Camera position
                    [fp_width, fp_height, -1.0],  # Top-right
                    [-fp_width, fp_height, -1.0],  # Top-left
                    [-fp_width, -fp_height, -1.0],  # Bottom-left
                    [fp_width, -fp_height, -1.0],  # Bottom-right
                ],
                dtype=np.float32,
            )
            idx_arr = np.array(
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [0, 4],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 1],
                ],
                dtype=np.uint32,
            )

            vbo = self._device.create_buffer(
                usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
                label="camera_wireframe_vbo",
                data=vert_arr,
            )
            ibo = self._device.create_buffer(
                usage=spy.BufferUsage.index_buffer | spy.BufferUsage.shader_resource,
                label="camera_wireframe_ibo",
                data=idx_arr,
            )
            model_mat = camera.transform.get_matrix()
            root = pass_encoder.bind_pipeline(self._pipeline)
            root_cursor = spy.ShaderCursor(root)
            root_cursor["uniforms"]["modelMatrix"].write(model_mat)
            root_cursor["uniforms"]["viewMatrix"].write(view_mat)
            root_cursor["uniforms"]["projMatrix"].write(proj_mat)
            root_cursor["uniforms"]["color"].write([1.0, 1.0, 1.0, 1.0])
            pass_encoder.set_render_state(
                {
                    "viewports": [
                        spy.Viewport.from_size(
                            self._render_target.width, self._render_target.height
                        )
                    ],
                    "scissor_rects": [
                        spy.ScissorRect.from_size(
                            self._render_target.width, self._render_target.height
                        )
                    ],
                    "vertex_buffers": [vbo],
                    "index_buffer": ibo,
                    "index_format": spy.IndexFormat.uint32,
                }
            )
            pass_encoder.draw_indexed({"vertex_count": idx_arr.shape[0] * 2})
        self._device.submit_command_buffer(command_encoder.finish())
