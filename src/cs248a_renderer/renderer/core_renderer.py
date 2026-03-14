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
from cs248a_renderer.model.cameras import PerspectiveCamera

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

    _gaussian_positions: torch.Tensor | None
    _gaussian_rotations: torch.Tensor | None
    _gaussian_scales: torch.Tensor | None
    _gaussian_colors: torch.Tensor | None
    _gaussian_opacities: torch.Tensor | None
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
        self.renderer_cuda_module = render_modules.renderer_cuda_module
         
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
        self._gaussian_positions = gaussian.positions
        self._gaussian_rotations = gaussian.rotations
        self._gaussian_scales = gaussian.scales
        self._gaussian_colors = gaussian.colors
        self._gaussian_opacities = gaussian.opacities
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
        if self._gaussian_positions is not None:
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

    def divide_ceil(self, a: int, b: int) -> int:
        return (a + b - 1) // b

    def render_gaussians(
        self,
        cam: PerspectiveCamera,
        num_tiles: glm.ivec2,
    ) -> None:

        # print("time started")
        start_time = time.perf_counter()

        block_size = 256

        view_mat_tensor = torch.as_tensor(
            np.ascontiguousarray(cam.view_matrix()), dtype=torch.float32, device="cuda",
        )
        proj_mat_tensor = torch.as_tensor(
            np.ascontiguousarray(cam.projection_matrix(self._render_target.width, self._render_target.height)),
            dtype=torch.float32, device="cuda",
        )

        tiles_touched = torch.zeros(self._gaussian_count, dtype=torch.int32, device="cuda")
        tile_ranges_touched = torch.zeros((self._gaussian_count, 4), dtype=torch.int32, device="cuda")
        radii = torch.zeros(self._gaussian_count, dtype=torch.int32, device="cuda")
        opacities = torch.zeros(self._gaussian_count, dtype=torch.float32, device="cuda")
        centers = torch.zeros((self._gaussian_count, 3), dtype=torch.float32, device="cuda")
        inv_cov2Ds = torch.zeros((self._gaussian_count, 2, 2), dtype=torch.float32, device="cuda")
        rgbs = torch.zeros((self._gaussian_count, 3), dtype=torch.float32, device="cuda")

        # print("about to preprocess gaussians, timestamp: " + str(time.perf_counter() - start_time))
        self.renderer_cuda_module.preprocessGaussians(
            positions=self._gaussian_positions,
            rotations=self._gaussian_rotations,
            scales=self._gaussian_scales,
            colors=self._gaussian_colors,
            gaussian_opacities=self._gaussian_opacities,
            viewMatrix_t=view_mat_tensor,
            projMatrix_t=proj_mat_tensor,
            canvas_width=self._render_target.width,
            canvas_height=self._render_target.height,
            focalLength=cam.focal_length(self._render_target.height),
            tiles_touched=tiles_touched,
            tile_ranges_touched=tile_ranges_touched,
            radii=radii,
            opacities=opacities,
            centers=centers,
            inv_cov2Ds=inv_cov2Ds,
            rgbs=rgbs,
        ).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=(self.divide_ceil(self._gaussian_count, block_size), 1, 1),
        )
        # print("preprocessed gaussians, timestamp: " + str(time.perf_counter() - start_time))

        total_num_tiles = int(num_tiles.x) * int(num_tiles.y)

        tiles_touched_cumsum = torch.cumsum(tiles_touched, dim=0, dtype=torch.int32)
        total_tiles_touched = tiles_touched_cumsum[-1].item()
        # print("total tile-gaussian intersections: " + str(total_tiles_touched))

        tiles_touched_prefix_sum = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), tiles_touched_cumsum])

        tile_and_depth_keys_buf = torch.zeros(total_tiles_touched, dtype=torch.int64, device="cuda")
        gauss_idx_vals_buf = torch.zeros(total_tiles_touched, dtype=torch.int32, device="cuda")

        # print("about to make keys, timestamp: " + str(time.perf_counter() - start_time))
        self.renderer_cuda_module.makeDict(
            tiles_touched_prefix_sum=tiles_touched_prefix_sum,
            tile_ranges_touched=tile_ranges_touched,
            centers=centers,
            tile_and_depth_keys_buf=tile_and_depth_keys_buf,
            gauss_idx_vals_buf=gauss_idx_vals_buf,
        ).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=(self.divide_ceil(self._gaussian_count, block_size), 1, 1),
        )
        # print("finished making keys, timestamp: " + str(time.perf_counter() - start_time))

        # print("starting sort, timestamp: " + str(time.perf_counter() - start_time))
        sorted_tile_and_depth_keys_buf, sort_indices = torch.sort(tile_and_depth_keys_buf)
        sorted_gauss_idx_vals_buf = gauss_idx_vals_buf[sort_indices]

        tile_range_starts = torch.zeros(total_num_tiles, dtype=torch.int32, device="cuda")
        tile_range_ends = torch.zeros(total_num_tiles, dtype=torch.int32, device="cuda")

        # print("sorted, timestamp: " + str(time.perf_counter() - start_time))

        self.renderer_cuda_module.prefixSumTiles(
            total_tiles_touched=total_tiles_touched,
            sorted_tile_and_depth_keys_buf=sorted_tile_and_depth_keys_buf,
            tile_range_starts=tile_range_starts,
            tile_range_ends=tile_range_ends,
        ).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=(self.divide_ceil(total_tiles_touched, block_size), 1, 1),
        )
        # print("calculated tile ranges, timestamp: " + str(time.perf_counter() - start_time))

        # print("about to render gaussians, timestamp: " + str(time.perf_counter() - start_time))
        tile_size = glm.ivec2(self._render_target.width // num_tiles.x, self._render_target.height // num_tiles.y)

        result = torch.zeros(
            (self._render_target.height, self._render_target.width, 4),
            dtype=torch.float32,
            device="cuda",
        )
        num_gaussians_used = torch.zeros(
            (self._render_target.height, self._render_target.width),
            dtype=torch.int32,
            device="cuda",
        )

        d_inv_cov2Ds = torch.zeros_like(inv_cov2Ds)
        d_centers = torch.zeros_like(centers)
        d_rgbs = torch.zeros_like(rgbs)
        d_opacities = torch.zeros_like(opacities)

        self.renderer_cuda_module.renderGaussians(
            gaussian_idxs=sorted_gauss_idx_vals_buf,
            tile_range_starts=tile_range_starts,
            tile_range_ends=tile_range_ends,
            inv_cov2Ds=(inv_cov2Ds, d_inv_cov2Ds),
            centers=(centers, d_centers),
            rgbs=(rgbs, d_rgbs),
            opacities=(opacities, d_opacities),
            result=result,
            num_gaussians_used=num_gaussians_used
        ).launchRaw(
            gridSize=(num_tiles.x, num_tiles.y, 1),
            blockSize=(tile_size.x, tile_size.y, 1),
        )
        print("rendered gaussians, timestamp: " + str(time.perf_counter() - start_time))
        self._render_target.copy_from_numpy(result.cpu().numpy())


    def render_gaussians_bwd(
        self,
        cam: PerspectiveCamera,
        num_tiles: glm.ivec2
    ) -> None:
        

        
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
