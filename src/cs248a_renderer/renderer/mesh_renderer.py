"""
Preview renderer for triangle meshes.
"""

from typing import Tuple
import numpy as np
import slangpy as spy
from pyglm import glm

from cs248a_renderer.model.scene import Scene
from cs248a_renderer.model.mesh import Mesh


class MeshRenderer:
    """Renderer for displaying triangle meshes."""

    _device: spy.Device
    _render_target: spy.Texture
    _depth_target: spy.Texture
    _canvas_size: Tuple[int, int]

    def __init__(self, device: spy.Device, render_target: spy.Texture) -> None:
        self._device = device
        self._render_target = render_target
        self._depth_target = self._device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.d32_float_s8_uint,
            width=render_target.width,
            height=render_target.height,
            usage=spy.TextureUsage.depth_stencil,
        )

        self._input_layout = self._device.create_input_layout(
            input_elements=[
                {
                    "semantic_name": "POSITION",
                    "semantic_index": 0,
                    "format": spy.Format.rgb32_float,
                },
                {
                    "semantic_name": "NORMAL",
                    "semantic_index": 0,
                    "format": spy.Format.rgb32_float,
                    "offset": 12,
                },
            ],
            vertex_streams=[{"stride": 24}],
        )
        self._program = self._device.load_program(
            "gui/mesh.slang", ["vertexMain", "fragmentMain"]
        )
        self._module = spy.Module.load_from_file(
            device=self._device, path="gui/mesh.slang"
        )
        self._pipeline = self._device.create_render_pipeline(
            program=self._program,
            input_layout=self._input_layout,
            primitive_topology=spy.PrimitiveTopology.triangle_list,
            targets=[
                {
                    "format": spy.Format.rgba8_unorm,
                }
            ],
            depth_stencil={
                "depth_test_enable": True,
                "depth_write_enable": True,
                "format": spy.Format.d32_float_s8_uint,
                "depth_func": spy.ComparisonFunc.less,
            },
        )

    def update_render_target(self, render_target: spy.Texture) -> None:
        """Update the render target texture."""

        self._render_target = render_target
        self._depth_target = self._device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.d32_float_s8_uint,
            width=render_target.width,
            height=render_target.height,
            usage=spy.TextureUsage.depth_stencil,
        )

    def update_canvas_size(self, size: Tuple[int, int]) -> None:
        """Update the canvas size (width, height)."""

        self._canvas_size = size

    def clear_depth_target(self) -> None:
        """Clear the depth target texture."""
        command_encoder = self._device.create_command_encoder()
        command_encoder.clear_texture_depth_stencil(
            texture=self._depth_target,
            clear_depth=True,
            depth_value=1.0,
        )
        self._device.submit_command_buffer(command_encoder.finish())

    def render_mesh(
        self,
        mesh: Mesh,
        model_mat: glm.mat4,
        view_mat: glm.mat4,
        proj_mat: glm.mat4,
    ) -> None:
        if mesh._o3d_mesh is None:
            return
        vertices = np.ascontiguousarray(mesh._o3d_mesh.vertices, dtype=np.float32)
        normals = np.ascontiguousarray(mesh._o3d_mesh.vertex_normals, dtype=np.float32)
        vertex_data = np.hstack((vertices, normals))
        indices = np.ascontiguousarray(mesh._o3d_mesh.triangles, dtype=np.uint32)

        vbo = self._device.create_buffer(
            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
            label="mesh_vbo",
            data=vertex_data,
        )
        ibo = self._device.create_buffer(
            usage=spy.BufferUsage.index_buffer | spy.BufferUsage.shader_resource,
            label="mesh_ibo",
            data=indices,
        )

        command_encoder = self._device.create_command_encoder()
        with command_encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": self._render_target.create_view({}),
                        "load_op": spy.LoadOp.load,
                        "store_op": spy.StoreOp.store,
                    }
                ],
                "depth_stencil_attachment": {
                    "view": self._depth_target.create_view({}),
                    "depth_load_op": spy.LoadOp.load,
                    "depth_store_op": spy.StoreOp.store,
                },
            }
        ) as pass_encoder:
            root = pass_encoder.bind_pipeline(self._pipeline)
            root_cursor = spy.ShaderCursor(root)
            root_cursor["uniforms"]["modelMatrix"].write(model_mat)
            root_cursor["uniforms"]["viewMatrix"].write(view_mat)
            root_cursor["uniforms"]["projMatrix"].write(proj_mat)
            root_cursor["uniforms"]["color"].write((1.0, 0.0, 1.0, 1.0))

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
            pass_encoder.draw_indexed({"vertex_count": int(indices.size)})
        self._device.submit_command_buffer(command_encoder.finish())

    def render_scene_mesh(
        self,
        scene: Scene,
        view_mat: glm.mat4,
        proj_mat: glm.mat4,
    ) -> None:
        stack = [scene.root]
        while stack:
            obj = stack.pop()
            if isinstance(obj, Mesh):
                self.render_mesh(
                    mesh=obj,
                    model_mat=obj.get_transform_matrix(),
                    view_mat=view_mat,
                    proj_mat=proj_mat,
                )
            stack.extend(obj.children)
