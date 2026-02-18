"""Renderer for NeRF scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import slangpy as spy
from pyglm import glm
from reactivex.subject import BehaviorSubject
import slangpy_nn as nn

from cs248a_renderer.model.nerf import NeRF
from cs248a_renderer.model.scene import NeRFScene


@dataclass(slots=True)
class _LayerBuffers:
    """Container describing buffers for a single MLP layer."""

    weights: spy.Buffer
    biases: spy.Buffer
    weights_shape: Tuple[int, ...]
    biases_shape: Tuple[int, ...]


@dataclass(slots=True)
class _LayerGradientBuffers:
    """Container with gradient buffers for a single MLP layer."""

    d_weights: spy.Buffer
    d_biases: spy.Buffer


class NeRFRenderer:
    """Renderer for NeRF-based scenes using Slang kernels."""

    _device: spy.Device
    _render_target: spy.Texture
    _nerf: NeRF | None
    _mlp: nn.IModel | None

    def __init__(
        self,
        device: spy.Device,
        render_texture_sbj: BehaviorSubject[Tuple[spy.Texture, int]] | None = None,
        render_texture: spy.Texture | None = None,
    ) -> None:
        self._device = device
        self._nerf = None

        def update_render_target(texture: Tuple[spy.Texture, int]) -> None:
            self._render_target = texture[0]

        if render_texture is not None:
            self._render_target = render_texture
        elif render_texture_sbj is not None:
            render_texture_sbj.subscribe(update_render_target)
        else:
            raise ValueError(
                "Must provide a render_texture or render_texture_sbj for NeRFRenderer."
            )

        self.module = spy.Module.load_from_file(
            device=device, path="nerf_renderer.slang"
        )

    def load_nerf(self, scene: NeRFScene) -> None:
        """Upload NeRF parameters to the device."""
        self._nerf = scene.nerf
        self._mlp = self._nerf.mlp

    def reset_nerf_d(self) -> None:
        """Create fresh gradient buffers for the current NeRF."""
        if self._nerf is None:
            raise ValueError("No NeRF loaded. Cannot reset gradient buffers.")

        params = self._nerf.mlp.parameters()
        for p in params:
            p.grad.clear()

    def render(
        self,
        scene: NeRFScene,
        view_mat: glm.mat4,
        fov: float,
    ) -> None:
        self.load_nerf(scene=scene)
        self.render_with_cache(scene=scene, view_mat=view_mat, fov=fov)

    def render_with_cache(
        self,
        scene: NeRFScene,
        view_mat: glm.mat4,
        fov: float,
    ) -> None:
        if self._nerf is None:
            self.load_nerf(scene)

        model_mat = scene.nerf.transform.get_matrix()
        focal_length = (0.5 * float(self._render_target.height)) / np.tan(
            np.radians(fov) / 2.0
        )

        nerf_aabb = self._build_nerf_aabb(scene, include_grad=False)

        self.module.renderForward(
            tid=spy.grid(shape=(self._render_target.height, self._render_target.width)),
            uniforms={
                "canvasSize": [self._render_target.width, self._render_target.height],
                "invModelMatrix": np.ascontiguousarray(
                    glm.inverse(model_mat), dtype=np.float32
                ),
                "invViewMatrix": np.ascontiguousarray(
                    glm.inverse(view_mat), dtype=np.float32
                ),
                "focalLength": float(focal_length),
                "ambientColor": list(scene.ambient_color),
                "rayMarcherConfig": {
                    "maxSteps": scene.ray_marcher_config.max_steps,
                    "stepSize": scene.ray_marcher_config.step_size,
                    "densityScale": scene.ray_marcher_config.density_scale,
                },
                "nerfAABB": nerf_aabb,
            },
            mlp=self._mlp,
            _result=self._render_target,
        )

    def render_backward(self, scene: NeRFScene, d_output: np.ndarray) -> None:
        if self._nerf is None:
            self.load_nerf(scene)

        camera = scene.camera
        model_mat = scene.nerf.transform.get_matrix()
        view_mat = camera.view_matrix()

        focal_length = (0.5 * float(self._render_target.height)) / np.tan(
            np.radians(camera.fov) / 2.0
        )

        d_output_tex = self._device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=self._render_target.width,
            height=self._render_target.height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            data=d_output,
        )

        nerf_aabb = self._build_nerf_aabb(scene, include_grad=True)

        self.module.renderBackward(
            tid=spy.grid(shape=(self._render_target.width, self._render_target.height)),
            uniforms={
                "canvasSize": [self._render_target.width, self._render_target.height],
                "invModelMatrix": np.ascontiguousarray(
                    glm.inverse(model_mat), dtype=np.float32
                ),
                "invViewMatrix": np.ascontiguousarray(
                    glm.inverse(view_mat), dtype=np.float32
                ),
                "focalLength": float(focal_length),
                "ambientColor": list(scene.ambient_color),
                "rayMarcherConfig": {
                    "maxSteps": scene.ray_marcher_config.max_steps,
                    "stepSize": scene.ray_marcher_config.step_size,
                    "densityScale": scene.ray_marcher_config.density_scale,
                },
                "nerfAABB": nerf_aabb,
                "dOutputTexture": d_output_tex,
            },
            mlp=self._mlp,
        )

    def _build_nerf_aabb(
        self, scene: NeRFScene, *, include_grad: bool
    ) -> Dict[str, object]:
        nerf = scene.nerf
        min_corner, max_corner = nerf.bounding_box

        uniform: Dict[str, object] = {
            "minBound": [min_corner.x, min_corner.y, min_corner.z],
            "maxBound": [max_corner.x, max_corner.y, max_corner.z],
        }

        return uniform
