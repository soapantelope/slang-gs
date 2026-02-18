"""
Volume Renderer Module.
"""

from typing import Tuple
from reactivex.subject import BehaviorSubject
import slangpy as spy
import numpy as np
from pyglm import glm

from cs248a_renderer.model.cameras import PerspectiveCamera
from cs248a_renderer.model.scene import SingleVolumeScene


class VolumeRenderer:
    _device: spy.Device
    _render_target: spy.Texture
    _data: spy.Buffer

    def __init__(
        self,
        device: spy.Device,
        render_texture_sbj: BehaviorSubject[Tuple[spy.Texture, int]] | None = None,
        render_texture: spy.Texture | None = None,
    ) -> None:
        self._device = device

        def update_render_target(texture: Tuple[spy.Texture, int]):
            self._render_target = texture[0]

        if render_texture is not None:
            self._render_target = render_texture
        elif render_texture_sbj is not None:
            render_texture_sbj.subscribe(update_render_target)
        else:
            raise ValueError(
                "Must provide a render_texture or render_texture_sbj for VolumeRenderer."
            )

        # Load shader and create kernel.
        self._forward_program = self._device.load_program(
            "volume_renderer.slang", ["renderForward"]
        )
        self._forward_kernel = self._device.create_compute_kernel(self._forward_program)
        self._backward_program = self._device.load_program(
            "volume_renderer.slang", ["renderBackward"]
        )
        self._backward_kernel = self._device.create_compute_kernel(
            self._backward_program
        )

    def load_volume(
        self,
        scene: SingleVolumeScene,
    ):
        self._volume = scene.volume
        if self._volume.channels == 1:
            shp = self._volume.data.shape
            data_arr = np.zeros((shp[0], shp[1], shp[2], 4), dtype=np.float32)
            data_arr[:, :, :, 3] = self._volume.data[:, :, :, 0]
        elif self._volume.channels >= 4:
            data_arr = np.ascontiguousarray(self._volume.data[:, :, :, 0:4]).astype(
                np.float32
            )
        else:
            raise ValueError(
                f"Unsupported number of channels: {self._volume.channels}. Must be 1 or >=4."
            )
        self._data = self._device.create_buffer(
            format=spy.Format.rgba32_float,
            usage=spy.BufferUsage.shader_resource,
            data=data_arr,
        )

    def reset_volume_d(self):
        d_density_arr = np.zeros((self._data.size,), dtype=np.uint8)
        self._d_data = self._device.create_buffer(
            format=spy.Format.rgba32_float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            data=d_density_arr,
        )

    def get_volume_d(self):
        dim = self._volume.shape
        return self._d_data.to_numpy().reshape(dim[0], dim[1], dim[2], 4)

    def render(
        self,
        scene: SingleVolumeScene,
        view_mat: glm.mat4,
        fov: float,
        use_albedo_volume: bool = False,
    ):
        self.load_volume(scene=scene)
        self.render_with_cache(
            scene=scene, view_mat=view_mat, fov=fov, use_albedo_volume=use_albedo_volume
        )

    def render_with_cache(
        self,
        scene: SingleVolumeScene,
        view_mat: glm.mat4,
        fov: float,
        use_albedo_volume: bool = False,
    ):
        volume = scene.volume

        min, max = volume.bounding_box
        dim = volume.shape

        # Get model and view matrices.
        model_mat = volume.transform.get_matrix()

        # Calculate focal length based on vertical FOV.
        focal_length = (0.5 * float(self._render_target.height)) / np.tan(
            np.radians(fov) / 2.0
        )

        volume = {
            "minBound": (min.x, min.y, min.z),
            "maxBound": (max.x, max.y, max.z),
            "invModelMatrix": np.array(glm.inverse(model_mat)),
            "dimensions": (dim[2], dim[1], dim[0]),
            "data": {
                "tex": self._data,
                "size": (dim[2], dim[1], dim[0]),
            },
            "useAlbedoVolume": use_albedo_volume,
            "albedo": scene.volume.properties["albedo"],
        }

        self._forward_kernel.dispatch(
            thread_count=(self._render_target.width, self._render_target.height, 1),
            uniforms={
                "canvasSize": (self._render_target.width, self._render_target.height),
                "invViewMatrix": np.array(glm.inverse(view_mat)),
                "focalLength": focal_length,
                "ambientColor": scene.ambient_color,
                "rayMarcherConfig": {
                    "maxSteps": scene.ray_marcher_config.max_steps,
                    "stepSize": scene.ray_marcher_config.step_size,
                    "densityScale": scene.ray_marcher_config.density_scale,
                },
                "volume": volume,
                "outputTexture": self._render_target,
            },
        )

    def render_backward(
        self,
        scene: SingleVolumeScene,
        d_output: np.ndarray,
        use_albedo_volume: bool = False,
    ):
        volume = scene.volume
        camera = scene.camera

        min, max = volume.bounding_box
        dim = volume.shape

        d_output_tex = self._device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=self._render_target.width,
            height=self._render_target.height,
            usage=spy.TextureUsage.shader_resource,
            data=d_output,
        )

        # Get model and view matrices.
        model_mat = volume.transform.get_matrix()
        view_mat = camera.view_matrix()

        # Calculate focal length based on vertical FOV.
        focal_length = (0.5 * float(self._render_target.height)) / np.tan(
            np.radians(camera.fov) / 2.0
        )

        volume = {
            "minBound": (min.x, min.y, min.z),
            "maxBound": (max.x, max.y, max.z),
            "dimensions": (dim[2], dim[1], dim[0]),
            "invModelMatrix": np.array(glm.inverse(model_mat)),
            "data": {
                "tex": self._data,
                "size": (dim[2], dim[1], dim[0]),
            },
            "dData": {
                "dTex": self._d_data,
            },
            "useAlbedoVolume": use_albedo_volume,
            "albedo": scene.volume.properties["albedo"],
        }

        self._backward_kernel.dispatch(
            thread_count=(self._render_target.width, self._render_target.height, 1),
            uniforms={
                "canvasSize": (self._render_target.width, self._render_target.height),
                "invViewMatrix": np.array(glm.inverse(view_mat)),
                "focalLength": focal_length,
                "ambientColor": scene.ambient_color,
                "rayMarcherConfig": {
                    "maxSteps": scene.ray_marcher_config.max_steps,
                    "stepSize": scene.ray_marcher_config.step_size,
                    "densityScale": scene.ray_marcher_config.density_scale,
                },
                "volume": volume,
                "dOutputTexture": d_output_tex,
            },
        )
