import time
from pyglm import glm
import torch
import slangtorch
from pathlib import Path

SHADER_PATH = Path(__file__).parent / "slang_shaders"

# simplified the renderer just for gaussians 
# for use with externally defined cameras like in 3DGS
class Renderer:
    _gaussian_positions: torch.Tensor | None
    _gaussian_rotations: torch.Tensor | None
    _gaussian_scales: torch.Tensor | None
    _gaussian_colors: torch.Tensor | None
    _gaussian_opacities: torch.Tensor | None
    _gaussian_count: int | None

    def __init__(
        self,
    ) -> None:
        self.renderer_cuda_module = slangtorch.loadModule(
            str(SHADER_PATH / "renderer.slang")
        )

    # loading directly from external source now
    def load_gaussians(self, positions, rotations, scales, colors, opacities, num_gaussians) -> None:
        self._gaussian_positions = positions
        self._gaussian_rotations = rotations
        self._gaussian_scales = scales
        self._gaussian_colors = colors
        self._gaussian_opacities = opacities
        self._gaussian_count = num_gaussians

    def divide_ceil(self, a: int, b: int) -> int:
        return (a + b - 1) // b

    def render_gaussians(
        self,
        view_matrix,
        proj_matrix,
        focal_length_x,
        focal_length_y,
        image_height,
        image_width,
        num_tiles: glm.ivec2,
    ) -> None:

        # print("time started")
        start_time = time.perf_counter()

        block_size = 256

        view_mat_tensor = view_matrix.contiguous()
        proj_mat_tensor = proj_matrix.contiguous()

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
            canvas_width=image_width,
            canvas_height=image_height,
            focal_length_x=focal_length_x,
            focal_length_y=focal_length_y,
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
        tile_size = glm.ivec2(image_width // num_tiles.x, image_height // num_tiles.y)

        result = torch.zeros(
            (image_height, image_width, 4),
            dtype=torch.float32,
            device="cuda",
        )
        num_gaussians_used = torch.zeros(
            (image_height, image_width),
            dtype=torch.int32,
            device="cuda",
        )

        d_inv_cov2Ds = torch.zeros_like(inv_cov2Ds)
        d_centers = torch.zeros_like(centers)
        d_rgbs = torch.zeros_like(rgbs)
        d_opacities = torch.zeros_like(opacities)

        self.renderer_cuda_module.renderGaussiansCUDAKernel(
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

        return result, centers, radii