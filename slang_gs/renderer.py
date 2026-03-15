import time
from pyglm import glm
import torch
import slangtorch
from pathlib import Path

def divide_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b

# simplified the renderer just for gaussians 
# for use with externally defined cameras like in 3DGS
class Renderer(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        cuda_module,
        positions,
        rotations,
        scales,
        colors,
        opacities,
        num_gaussians,
        view_matrix,
        proj_matrix,
        focal_length_x,
        focal_length_y,
        image_height,
        image_width,
        num_tiles: glm.ivec2,
    ) -> None:
        start_time = time.perf_counter()
        block_size = 256

        view_mat_tensor = view_matrix.contiguous()
        proj_mat_tensor = proj_matrix.contiguous()

        tiles_touched = torch.zeros(num_gaussians, dtype=torch.int32, device="cuda")
        tile_ranges_touched = torch.zeros((num_gaussians, 4), dtype=torch.int32, device="cuda")
        radii = torch.zeros(num_gaussians, dtype=torch.int32, device="cuda")
        centers = torch.zeros((num_gaussians, 3), dtype=torch.float32, device="cuda")
        inv_cov2Ds = torch.zeros((num_gaussians, 2, 2), dtype=torch.float32, device="cuda")
        rgbs = torch.zeros((num_gaussians, 3), dtype=torch.float32, device="cuda")

        cuda_module.preprocessGaussians(
            positions=positions,
            rotations=rotations,
            scales=scales,
            colors=colors,
            viewMatrix_t=view_mat_tensor,
            projMatrix_t=proj_mat_tensor,
            canvas_width=image_width,
            canvas_height=image_height,
            focal_length_x=focal_length_x,
            focal_length_y=focal_length_y,
            tiles_touched=tiles_touched,
            tile_ranges_touched=tile_ranges_touched,
            radii=radii,
            centers=centers,
            inv_cov2Ds=inv_cov2Ds,
            rgbs=rgbs,
        ).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=(divide_ceil(num_gaussians, block_size), 1, 1),
        )

        total_num_tiles = int(num_tiles.x) * int(num_tiles.y)

        tiles_touched_cumsum = torch.cumsum(tiles_touched, dim=0, dtype=torch.int32)
        total_tiles_touched = tiles_touched_cumsum[-1].item()

        tiles_touched_prefix_sum = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), tiles_touched_cumsum])

        tile_and_depth_keys_buf = torch.zeros(total_tiles_touched, dtype=torch.int64, device="cuda")
        gauss_idx_vals_buf = torch.zeros(total_tiles_touched, dtype=torch.int32, device="cuda")

        cuda_module.makeDict(
            tiles_touched_prefix_sum=tiles_touched_prefix_sum,
            tile_ranges_touched=tile_ranges_touched,
            centers=centers,
            tile_and_depth_keys_buf=tile_and_depth_keys_buf,
            gauss_idx_vals_buf=gauss_idx_vals_buf,
        ).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=(divide_ceil(num_gaussians, block_size), 1, 1),
        )

        sorted_tile_and_depth_keys_buf, sort_indices = torch.sort(tile_and_depth_keys_buf)
        sorted_gauss_idx_vals_buf = gauss_idx_vals_buf[sort_indices]

        tile_range_starts = torch.zeros(total_num_tiles, dtype=torch.int32, device="cuda")
        tile_range_ends = torch.zeros(total_num_tiles, dtype=torch.int32, device="cuda")

        cuda_module.prefixSumTiles(
            total_tiles_touched=total_tiles_touched,
            sorted_tile_and_depth_keys_buf=sorted_tile_and_depth_keys_buf,
            tile_range_starts=tile_range_starts,
            tile_range_ends=tile_range_ends,
        ).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=(divide_ceil(total_tiles_touched, block_size), 1, 1),
        )
        
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

        cuda_module.renderGaussiansCUDAKernel(
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
        print("rendered gaussians in " + str(time.perf_counter() - start_time) + " seconds")

        ctx.save_for_backward(positions, rotations, scales, colors, view_mat_tensor, proj_mat_tensor,
         tiles_touched, tile_ranges_touched, radii,sorted_gauss_idx_vals_buf, 
         tile_range_starts, tile_range_ends, inv_cov2Ds, centers, rgbs, opacities, 
         result, num_gaussians_used)
        ctx.cuda_module = cuda_module
        ctx.block_size = block_size
        ctx.num_gaussians = num_gaussians
        ctx.focal_length_x = focal_length_x
        ctx.focal_length_y = focal_length_y
        ctx.image_height = image_height
        ctx.image_width = image_width
        ctx.num_tiles = num_tiles
        return result, centers, radii
    
    @staticmethod
    def backward(ctx, d_result, centers_grad, radii_grad):
        (positions, rotations, scales, colors, view_mat_tensor, proj_mat_tensor,
         tiles_touched, tile_ranges_touched, radii,sorted_gauss_idx_vals_buf, 
         tile_range_starts, tile_range_ends, inv_cov2Ds, centers, rgbs, opacities, 
         result, num_gaussians_used) = ctx.saved_tensors
        cuda_module = ctx.cuda_module
        block_size = ctx.block_size
        num_gaussians = ctx.num_gaussians
        focal_length_x = ctx.focal_length_x
        focal_length_y = ctx.focal_length_y
        image_height = ctx.image_height
        image_width = ctx.image_width
        num_tiles = ctx.num_tiles

        d_inv_cov2Ds = torch.zeros_like(inv_cov2Ds)
        d_centers = torch.zeros_like(centers)
        d_rgbs = torch.zeros_like(rgbs)
        d_opacities = torch.zeros_like(opacities)

        tile_size = glm.ivec2(image_width // num_tiles.x, image_height // num_tiles.y)

        cuda_module.renderGaussiansCUDAKernel.bwd(
            gaussian_idxs=sorted_gauss_idx_vals_buf,
            tile_range_starts=tile_range_starts,
            tile_range_ends=tile_range_ends,
            inv_cov2Ds=(inv_cov2Ds, d_inv_cov2Ds),
            centers=(centers, d_centers),
            rgbs=(rgbs, d_rgbs),
            opacities=(opacities, d_opacities),
            result=(result, d_result),
            num_gaussians_used=num_gaussians_used
        ).launchRaw(
            gridSize=(num_tiles.x, num_tiles.y, 1),
            blockSize=(tile_size.x, tile_size.y, 1),
        )

        d_positions = torch.zeros_like(positions)
        d_rotations = torch.zeros_like(rotations)
        d_scales = torch.zeros_like(scales)
        d_colors = torch.zeros_like(colors)

        cuda_module.preprocessGaussians.bwd(
            positions=(positions, d_positions),
            rotations=(rotations, d_rotations),
            scales=(scales, d_scales),
            colors=(colors, d_colors),
            viewMatrix_t=view_mat_tensor,
            projMatrix_t=proj_mat_tensor,
            canvas_width=image_width,
            canvas_height=image_height,
            focal_length_x=focal_length_x,
            focal_length_y=focal_length_y,
            tiles_touched=tiles_touched,
            tile_ranges_touched=tile_ranges_touched,
            radii=radii,
            centers=(centers, d_centers),
            inv_cov2Ds=(inv_cov2Ds, d_inv_cov2Ds),
            rgbs=(rgbs, d_rgbs),
        ).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=(divide_ceil(num_gaussians, block_size), 1, 1),
        )

        return None, d_positions, d_rotations, d_scales, d_colors, d_opacities, None, None, None, None, None, None, None, None
        