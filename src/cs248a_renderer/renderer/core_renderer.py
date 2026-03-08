"""
Core rendering module
"""
import time

import slangpy as spy
from typing import Tuple, List, Dict
from pyglm import glm
import numpy as np
from reactivex.subject import BehaviorSubject
from enum import Enum
import torch

from cs248a_renderer import RendererModules
from cs248a_renderer.model.scene import Scene
from cs248a_renderer.model.mesh import Triangle, create_triangle_buf
from cs248a_renderer.model.volumes import create_volume_buf
from cs248a_renderer.model.bvh import BVH, create_bvh_node_buf
from cs248a_renderer.model.material import create_material_buf
from cs248a_renderer.model.volumes import DenseVolume
from cs248a_renderer.model.bounding_box import BoundingBox3D
from cs248a_renderer.model.material import PhysicsBasedMaterialTextureBuf
from cs248a_renderer.model.gaussian_splat import GaussianSplat

class FilteringMethod(Enum):
    NEAREST = 0
    BILINEAR = 1
    TRILINEAR = 2


class Renderer:
    _device: spy.Device
    _render_target: spy.Texture

    sqrt_spp: int = 1

    # Primitive buffers.
    _physics_based_material_texture_buf: PhysicsBasedMaterialTextureBuf | None
    _physics_based_material_buf: spy.NDBuffer | None
    _material_count: int | None
    _triangle_buf: spy.NDBuffer | None
    _triangle_count: int | None

    _surface_volume_tex_buf: spy.NDBuffer | None
    _surface_volume_buf: spy.NDBuffer | None
    _surface_volume_count: int | None

    _volume: Dict | None
    _volume_tex_buf: spy.NDBuffer | None
    _volume_d_tex_buf: spy.NDBuffer | None

    _gaussian_buf: spy.InstanceBuffer | None
    _gaussian_count: int | None

    _bvh_node_buf: spy.NDBuffer | None
    _use_bvh: bool = False
    _max_nodes: int = 0

    _sphere_sdf_buf: spy.NDBuffer | None
    _sphere_sdf_count: int | None

    _custom_sdf: Dict
    _render_custom_sdf: bool = False

    _ambientColor: np.array = np.array([0.0, 0.0, 0.0, 1.0])

    def __init__(
        self,
        device: spy.Device,
        render_texture_sbj: BehaviorSubject[Tuple[spy.Texture, int]] | None = None,
        render_texture: spy.Texture | None = None,
        render_modules: RendererModules | None = None,
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

        # Load renderer module.
        if render_modules is None:
            render_modules = RendererModules(device=device)
        self.primitive_module = render_modules.primitive_module
        self.texture_module = render_modules.texture_module
        self.model_module = render_modules.model_module
        self.renderer_module = render_modules.renderer_module
        self.material_module = render_modules.material_module

        # Initialize primitive buffers.
        self._physics_based_material_texture_buf = PhysicsBasedMaterialTextureBuf(
            albedo=spy.NDBuffer(
                device=device, dtype=self.material_module.float3.as_struct(), shape=(1,)
            )
        )
        self._physics_based_material_buf = spy.NDBuffer(
            device=device,
            dtype=self.material_module.PhysicsBasedMaterial.as_struct(),
            shape=(1,),
        )
        self._material_count = 0
        self._triangle_buf = spy.NDBuffer(
            device=device, dtype=self.primitive_module.Triangle.as_struct(), shape=(1,)
        )
        self._triangle_count = 0
        self._surface_volume_tex_buf = spy.NDBuffer(
            device=device, dtype=self.texture_module.float4.as_struct(), shape=(1,)
        )
        self._surface_volume_buf = spy.NDBuffer(
            device=device, dtype=self.primitive_module.Volume.as_struct(), shape=(1,)
        )
        self._surface_volume_count = 0
        self._bvh_node_buf = spy.NDBuffer(
            device=device, dtype=self.model_module.BVHNode.as_struct(), shape=(1,)
        )
        self._max_nodes = 0
        self._sphere_sdf_buf = spy.NDBuffer(
            device=device, dtype=self.primitive_module.SphereSDF.as_struct(), shape=(1,)
        )
        self._sphere_sdf_count = 0
        self._cube_sdf_buf = spy.NDBuffer(
            device=device, dtype=self.primitive_module.CubeSDF.as_struct(), shape=(1,)
        )
        self._cube_sdf_count = 0
        self._custom_sdf = {
            "cubeSize": [1.0, 1.0, 1.0],
            "sphereRadius": 0.5,
            "invModelMatrix": np.identity(4, dtype=np.float32),
        }
        self._filtering_method = 0
        self._ambientColor = np.array([0.0, 0.0, 0.0, 1.0])
        self._volume = {
            "bound": BoundingBox3D(min=glm.vec3(0.0), max=glm.vec3(0.0)).get_this(),
            "tex": {
                "tex": spy.NDBuffer(
                    device=self._device,
                    dtype=self.primitive_module.float4,
                    shape=(1,),
                ),
                "size": [1, 1, 1],
            },
            "dTex": {
                "dTex": spy.NDBuffer(
                    device=self._device,
                    dtype=self.primitive_module.find_struct("Atomic<float>[4]"),
                    shape=(1,),
                ),
            },
            "modelMatrix": spy.math.float4x4(
                np.ascontiguousarray(glm.mat4(1.0), dtype=np.float32)
            ),
            "invModelMatrix": spy.math.float4x4(
                np.ascontiguousarray(glm.mat4(1.0), dtype=np.float32)
            ),
        }

    def load_gaussian(self, gaussian: GaussianSplat) -> None:
        # load a single gaussian splat into renderer
        self._gaussian_buf = gaussian.gaussians
        self._gaussian_count = gaussian.num_gaussians

    def load_triangles(self, scene: Scene) -> None:
        """Load a scene into the renderer."""
        triangles, materials = scene.extract_triangles_with_material()
        self._triangle_buf = create_triangle_buf(self.primitive_module, triangles)
        self._physics_based_material_buf, self._physics_based_material_texture_buf = (
            create_material_buf(self.material_module, materials)
        )
        self._triangle_count = len(triangles)
        # Clear BVH when loading new triangles.
        self._bvh_node_buf = spy.NDBuffer(
            device=self._device, dtype=self.model_module.BVHNode.as_struct(), shape=(1,)
        )
        self._max_nodes = 0
        self._use_bvh = False

    def load_surface_volumes(self, scene: Scene) -> None:
        """Load volumes into the renderer."""
        volumes = scene.extract_volumes()
        self._surface_volume_buf, self._surface_volume_tex_buf = create_volume_buf(
            self.primitive_module, volumes
        )
        self._surface_volume_count = len(volumes)

    def load_volume(self, volume: DenseVolume) -> None:
        """Load a single volume into the renderer."""
        np_volume = volume.data.reshape(-1, 4)
        volume_tex_buf = spy.NDBuffer(
            device=self._device,
            dtype=self.primitive_module.float4,
            shape=(max(np_volume.shape[0], 1),),
        )
        volume_tex_buf.copy_from_numpy(np_volume)
        self._volume_tex_buf = volume_tex_buf
        volume_d_tex_buf = spy.NDBuffer(
            device=self._device,
            dtype=self.primitive_module.find_struct("Atomic<float>[4]"),
            shape=(max(np_volume.shape[0], 1),),
        )
        self._volume_d_tex_buf = volume_d_tex_buf
        self._volume = {
            "bound": volume.bounding_box.get_this(),
            "tex": {
                "tex": volume_tex_buf,
                "size": [volume.shape[2], volume.shape[1], volume.shape[0]],
            },
            "dTex": {
                "dTex": volume_d_tex_buf,
            },
            "modelMatrix": spy.math.float4x4(
                np.ascontiguousarray(volume.get_transform_matrix(), dtype=np.float32)
            ),
            "invModelMatrix": spy.math.float4x4(
                np.ascontiguousarray(
                    glm.inverse(volume.get_transform_matrix()), dtype=np.float32
                )
            ),
        }

    def get_d_volume(self):
        """Get the volume density gradient buffer."""
        return self._volume_d_tex_buf.to_numpy()

    def load_bvh(self, triangles: List[Triangle], bvh: BVH) -> None:
        self._triangle_buf = create_triangle_buf(self.primitive_module, triangles)
        self._triangle_count = len(triangles)
        self._bvh_node_buf = create_bvh_node_buf(self.model_module, bvh.nodes)
        self._max_nodes = len(bvh.nodes)
        self._use_bvh = True

    def load_sdf_spheres(self, sphere_buffer: spy.NDBuffer, sphere_count: int) -> None:
        """Load SDF spheres into the renderer."""
        self._sphere_sdf_buf = sphere_buffer
        self._sphere_sdf_count = sphere_count

    def load_sdf_cubes(self, cube_buffer: spy.NDBuffer, cube_count: int) -> None:
        """Load SDF cubes into the renderer."""
        self._cube_sdf_buf = cube_buffer
        self._cube_sdf_count = cube_count

    def set_custom_sdf(self, custom_sdf: Dict, render_custom_sdf: bool = False) -> None:
        """Load custom SDF into the renderer."""
        self._custom_sdf = custom_sdf
        self._render_custom_sdf = render_custom_sdf

    def _build_render_uniforms(
        self,
        view_mat: glm.mat4,
        proj_mat: glm.mat4,
        num_tiles: glm.ivec2,
        fov: float,
        render_depth: bool = False,
        render_normal: bool = False,
        visualize_barycentric_coords: bool = False,
        visualize_tex_uv: bool = False,
        visualize_level_of_detail: bool = False,
        visualize_albedo: bool = False,
    ) -> Dict:
        """Build the uniforms dictionary for rendering."""
        focal_length = (0.5 * float(self._render_target.height)) / np.tan(
            np.radians(fov) / 2.0
        )
        uniforms = {
            "camera": {
                "invViewMatrix": np.ascontiguousarray(
                    glm.inverse(view_mat), dtype=np.float32
                ),
                "viewMatrix": np.ascontiguousarray(
                    view_mat, dtype=np.float32
                ),
                "canvasSize": [
                    self._render_target.width,
                    self._render_target.height,
                ],
                "focalLength": float(focal_length),
            },
            "ambientColor": np.ascontiguousarray(self._ambientColor, dtype=np.float32),
            "sqrtSpp": self.sqrt_spp,
            "materialCount": self._material_count,
            "triangleCount": self._triangle_count,
            "surfaceVolumeCount": self._surface_volume_count,
            "volume": self._volume,
            "useBVH": self._use_bvh,
            "renderDepth": render_depth,
            "renderNormal": render_normal,
            "visualizeBarycentricCoords": visualize_barycentric_coords,
            "visualizeTexUV": visualize_tex_uv,
            "visualizeLevelOfDetail": visualize_level_of_detail,
            "visualizeAlbedo": visualize_albedo,
        }
        if proj_mat is not None:
            uniforms["camera"]["projMatrix"] = np.ascontiguousarray(
                proj_mat, dtype=np.float32
            )
        if num_tiles is not None:
            uniforms["camera"]["num_tiles"] = np.ascontiguousarray(
                num_tiles, dtype=np.int32
            )
        if self._physics_based_material_texture_buf is not None:
            uniforms["physicsBasedMaterialTextureBuf"] = {
                "albedoTexBuf": {
                    "buffer": self._physics_based_material_texture_buf.albedo,
                },
            }
        if self._physics_based_material_buf is not None:
            uniforms["physicsBasedMaterialBuf"] = self._physics_based_material_buf
        if self._triangle_buf is not None:
            uniforms["triangleBuf"] = self._triangle_buf
        if self._surface_volume_tex_buf is not None:
            uniforms["surfaceVolumeTexBuf"] = {
                "buffer": self._surface_volume_tex_buf,
            }
        if self._surface_volume_buf is not None:
            uniforms["surfaceVolumeBuf"] = self._surface_volume_buf
        if self._gaussian_buf is not None:
            uniforms["gaussianBuf"] = self._gaussian_buf
            uniforms["gaussianCount"] = self._gaussian_count
        if self._bvh_node_buf is not None:
            uniforms["bvh"] = {
                "nodes": self._bvh_node_buf,
                "maxNodes": self._max_nodes,
                "primitives": self._triangle_buf,
                "numPrimitives": self._triangle_count,
            }
        sdf_uniforms = {
            "sphereCount": self._sphere_sdf_count,
            "cubeCount": self._cube_sdf_count,
            "customSDF": self._custom_sdf,
            "renderCustomSDF": self._render_custom_sdf,
        }
        if self._sphere_sdf_buf is not None:
            sdf_uniforms["spheres"] = self._sphere_sdf_buf
        if self._cube_sdf_buf is not None:
            sdf_uniforms["cubes"] = self._cube_sdf_buf
        uniforms["sdfBuf"] = sdf_uniforms
        return uniforms

    def render_gaussians(
        self,
        view_mat: glm.mat4,
        proj_mat: glm.mat4,
        num_tiles: glm.ivec2,
        fov: float
    ) -> None:
        
        # TODO: use tiles instead of tiles, figure out thread groups(?)

        print("building uniforms")
        uniforms = self._build_render_uniforms(
            view_mat=view_mat,
            proj_mat=proj_mat,
            num_tiles=num_tiles,
            fov=fov
        )

        start_time = time.perf_counter()
        print("time started")

        tiles_touched = spy.NDBuffer(device=self._device, dtype=self.model_module.int, shape=(self._gaussian_count,))
        tile_ranges_touched = spy.NDBuffer(device=self._device, dtype=self.model_module.int4, shape=(self._gaussian_count,))
        radii = spy.NDBuffer(device=self._device, dtype=self.model_module.int, shape=(self._gaussian_count,))
        opacities = spy.NDBuffer(device=self._device, dtype=self.model_module.float, shape=(self._gaussian_count,))
        centers = spy.NDBuffer(device=self._device, dtype=self.model_module.float3, shape=(self._gaussian_count,))
        inv_cov2Ds = spy.NDBuffer(device=self._device, dtype=self.model_module.float2x2, shape=(self._gaussian_count,))
        rgbs = spy.NDBuffer(device=self._device, dtype=self.model_module.float3, shape=(self._gaussian_count,))

        print("about to preprocess gaussians, timestamp: " + str(time.perf_counter() - start_time))
        self.renderer_module.preprocessGaussians(
            tid=spy.grid(shape=(self._gaussian_count,)),
            uniforms=uniforms,
            tiles_touched=tiles_touched,
            tile_ranges_touched=tile_ranges_touched,
            radii=radii,
            opacities=opacities,
            centers=centers,
            inv_cov2Ds=inv_cov2Ds,
            rgbs=rgbs
        )
        print("preprocessed gaussians, timestamp: " + str(time.perf_counter() - start_time))

        total_num_tiles = num_tiles.x * num_tiles.y

        tiles_touched_np = tiles_touched.to_numpy()
        total_tiles_touched = int(np.sum(tiles_touched_np))
        print("total tile-gaussian intersections: " + str(total_tiles_touched))

        tiles_touched_prefix_sum_np = np.zeros(self._gaussian_count + 1, dtype=np.int32)
        tiles_touched_prefix_sum_np[1:] = np.cumsum(tiles_touched_np)
        tiles_touched_prefix_sum = spy.NDBuffer(device=self._device, dtype=self.model_module.int, shape=(self._gaussian_count + 1,))
        tiles_touched_prefix_sum.copy_from_numpy(tiles_touched_prefix_sum_np)

        tile_and_depth_keys_buf = spy.NDBuffer(device=self._device, dtype=self.model_module.uint64_t, shape=(total_tiles_touched,))
        gauss_idx_vals_buf = spy.NDBuffer(device=self._device, dtype=self.model_module.int, shape=(total_tiles_touched,))

        print("about to make keys, timestamp: " + str(time.perf_counter() - start_time))
        self.renderer_module.makeDict(
            tid=spy.grid(shape=(self._gaussian_count,)),
            uniforms=uniforms,
            tiles_touched_prefix_sum=tiles_touched_prefix_sum,
            tile_ranges_touched=tile_ranges_touched,
            centers=centers,
            tile_and_depth_keys_buf=tile_and_depth_keys_buf,
            gauss_idx_vals_buf=gauss_idx_vals_buf
        )
        print("finished making keys, timestamp: " + str(time.perf_counter() - start_time))

        # start part to parallelize TODO: radix sort them instead?

        print("starting to transfer and sort, timestamp: " + str(time.perf_counter() - start_time))
        tile_and_depth_keys_buf_np = tile_and_depth_keys_buf.to_numpy()
        gauss_idx_vals_buf = gauss_idx_vals_buf.to_numpy()

        idxs = np.argsort(tile_and_depth_keys_buf_np)
        sorted_tile_and_depth_keys_buf_np = tile_and_depth_keys_buf_np[idxs]
        sorted_gauss_idx_vals_buf_np = gauss_idx_vals_buf[idxs]

        sorted_tile_and_depth_keys_buf = spy.NDBuffer(device=self._device, dtype=self.model_module.uint64_t, shape=(total_tiles_touched,))
        sorted_tile_and_depth_keys_buf.copy_from_numpy(sorted_tile_and_depth_keys_buf_np)
        sorted_gauss_idx_vals_buf = spy.NDBuffer(device=self._device, dtype=self.model_module.int, shape=(total_tiles_touched,))
        sorted_gauss_idx_vals_buf.copy_from_numpy(sorted_gauss_idx_vals_buf_np)

        tile_range_starts = spy.NDBuffer(device=self._device, dtype=self.model_module.int, shape=(total_num_tiles,))
        tile_range_starts.copy_from_numpy(np.zeros(total_num_tiles, dtype=np.int32))
        tile_range_ends = spy.NDBuffer(device=self._device, dtype=self.model_module.int, shape=(total_num_tiles,))
        tile_range_ends.copy_from_numpy(np.zeros(total_num_tiles, dtype=np.int32))

        print("sorted and transfered back, timestamp: " + str(time.perf_counter() - start_time))

        self.renderer_module.prefixSumTiles(
            tid=spy.grid(shape=(total_tiles_touched,)),
            total_tiles_touched=total_tiles_touched,
            sorted_tile_and_depth_keys_buf=sorted_tile_and_depth_keys_buf,
            tile_range_starts=tile_range_starts,
            tile_range_ends=tile_range_ends,
        )
        print("calculated tile ranges, timestamp: " + str(time.perf_counter() - start_time))

        print("about to render gaussians, timestamp: " + str(time.perf_counter() - start_time))
        tile_height = self._render_target.height / num_tiles.y
        tile_width = self._render_target.width / num_tiles.x
        self.renderer_module.renderGaussians(
            uniforms=uniforms,
            _result=self._render_target,
            gaussian_idxs=sorted_gauss_idx_vals_buf,
            tile_range_starts=tile_range_starts,
            tile_range_ends=tile_range_ends,
            inv_cov2Ds=inv_cov2Ds,
            centers=centers,
            rgbs=rgbs,
            opacities=opacities
        )
        print("rendered gaussians, timestamp: " + str(time.perf_counter() - start_time))
        
    def render(
        self,
        view_mat: glm.mat4,
        fov: float,
        render_depth: bool = False,
        render_normal: bool = False,
        visualize_barycentric_coords: bool = False,
        visualize_tex_uv: bool = False,
        visualize_level_of_detail: bool = False,
        visualize_albedo: bool = False,
    ) -> None:
        """Render the loaded scene."""
        uniforms = self._build_render_uniforms(
            view_mat=view_mat,
            fov=fov,
            render_depth=render_depth,
            render_normal=render_normal,
            visualize_barycentric_coords=visualize_barycentric_coords,
            visualize_tex_uv=visualize_tex_uv,
            visualize_level_of_detail=visualize_level_of_detail,
            visualize_albedo=visualize_albedo,
        )
        self.renderer_module.render(
            tid=spy.grid(shape=(self._render_target.height, self._render_target.width)),
            uniforms=uniforms,
            _result=self._render_target,
        )

    def render_volume_backward(
        self,
        view_mat: glm.mat4,
        fov: float,
        out_grad: torch.Tensor,
    ) -> None:
        """Backward pass for volume rendering."""
        # Build output gradient texture.
        out_grad_texture = self._device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=out_grad.shape[1],
            height=out_grad.shape[0],
            usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
        )
        out_grad_texture.copy_from_numpy(out_grad.cpu().numpy().astype(np.float32))
        # Build uniforms.
        uniforms = self._build_render_uniforms(
            view_mat=view_mat,
            fov=fov,
        )
        self.renderer_module.renderVolumeBwd(
            tid=spy.grid(shape=(self._render_target.height, self._render_target.width)),
            uniforms=uniforms,
            outGrad=out_grad_texture,
        )
