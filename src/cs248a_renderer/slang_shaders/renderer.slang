module renderer;

import math;
import texture;
import material;
import model;
import primitive;

public struct RendererUniform
{
    // Camera parameters.
    public Camera camera;
    // Ambient color, used when no ray hits a primitive
    public float4 ambientColor;
    // Ray samples per pixel
    public uint sqrtSpp;
    // Primitives.
    public StructuredBuffer<Triangle> triangleBuf;
    public uint triangleCount;
    public SharedTexture3DBuffer<float4> surfaceVolumeTexBuf;
    public StructuredBuffer<Volume> surfaceVolumeBuf;
    public uint surfaceVolumeCount;
    public SDFBuffer sdfBuf;
    // Volumetric data.
    public DiffVolume volume;
    // Acceleration structure.
    public bool useBVH;
    public BVH<Triangle> bvh;
    // Materials.
    public PhysicsBasedMaterialTextureBuf physicsBasedMaterialTextureBuf;
    public StructuredBuffer<PhysicsBasedMaterial> physicsBasedMaterialBuf;
    public uint materialCount;
    // Filtering
    public uint filteringMethod;
    // Debug flags.
    public bool renderDepth;
    public bool renderNormal;
    public bool visualizeBarycentricCoords;
    public bool visualizeTexUV;
    public bool visualizeLevelOfDetail;
    public bool visualizeAlbedo;
}

static const int NUM_TILES_PER_DIM = 32;
groupshared ProjectedGaussian gaussians_in_threadgroup[NUM_TILES_PER_DIM * NUM_TILES_PER_DIM];
groupshared int gaussian_idxs_in_threadgroup[NUM_TILES_PER_DIM * NUM_TILES_PER_DIM];

float4x4 loadFloat4x4FromTensor(TensorView<float> t) {
    return float4x4(
        t[uint2(0,0)], t[uint2(0,1)], t[uint2(0,2)], t[uint2(0,3)],
        t[uint2(1,0)], t[uint2(1,1)], t[uint2(1,2)], t[uint2(1,3)],
        t[uint2(2,0)], t[uint2(2,1)], t[uint2(2,2)], t[uint2(2,3)],
        t[uint2(3,0)], t[uint2(3,1)], t[uint2(3,2)], t[uint2(3,3)]
    );
}

Camera loadCameraFromTensors(
    TensorView<float> viewMatrix_t,
    TensorView<float> projMatrix_t,
    uint canvas_width, uint canvas_height,
    float focalLength
) {
    Camera cam;
    cam.viewMatrix = loadFloat4x4FromTensor(viewMatrix_t);
    cam.invViewMatrix = float4x4(0);
    cam.projMatrix = loadFloat4x4FromTensor(projMatrix_t);
    cam.canvasSize = uint2(canvas_width, canvas_height);
    cam.focalLength = focalLength;
    cam.num_tiles = int2(NUM_TILES_PER_DIM, NUM_TILES_PER_DIM);
    return cam;
}

/**
 * Sample the scene at the given uv coordinate.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @return The color of the ray sample at the given uv coordinate.
 */
public float4 sample(float2 uv, RendererUniform uniforms)
{
    return float4(0.0, 0.0, 0.0, 0.0);
}

/**
 * The entry point of the renderer.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @return The color of the pixel at the given thread coordinate.
 */
public float4 render(uint2 tid, RendererUniform uniforms)
{
    float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < uniforms.sqrtSpp; i++) {
        for (int j = 0; j < uniforms.sqrtSpp; j++) {
            float offset = 0.5 * (1.0 / uniforms.sqrtSpp);
            float2 uv = float2(tid[0] + (1.0 / uniforms.sqrtSpp) * i + offset, tid[1] + (1.0 / uniforms.sqrtSpp) * j + offset) / float2(uniforms.camera.canvasSize);
            sum += sample(uv, uniforms);
        }
    }

    return sum / pow(uniforms.sqrtSpp, 2);
}

[AutoPyBindCUDA]
[CUDAKernel]
public void preprocessGaussians(
    TensorView<float> positions,
    TensorView<float> rotations,
    TensorView<float> scales,
    TensorView<float> colors,
    TensorView<float> gaussian_opacities,
    TensorView<float> viewMatrix_t,
    TensorView<float> projMatrix_t,
    uint canvas_width,
    uint canvas_height,
    float focalLength,
    TensorView<int> tiles_touched,
    TensorView<int> tile_ranges_touched,
    TensorView<int> radii,
    TensorView<float> opacities,
    TensorView<float> centers,
    TensorView<float> inv_cov2Ds,
    TensorView<float> rgbs
)
{
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= positions.size(0))
        return;

    Gaussian gaussian = loadGaussianFromTensors(tid, positions, rotations, scales, colors, gaussian_opacities);
    Camera cam = loadCameraFromTensors(viewMatrix_t, projMatrix_t, canvas_width, canvas_height, focalLength);

    tiles_touched[tid] = 0;
    tile_ranges_touched[uint2(tid, 0)] = 0;
    tile_ranges_touched[uint2(tid, 1)] = 0;
    tile_ranges_touched[uint2(tid, 2)] = 0;
    tile_ranges_touched[uint2(tid, 3)] = 0;

    // project center to screen space (NDC)
    float3 screen_center = gaussian.projectPoint(cam);
    if (screen_center.z >= -0.1) return;

    // compute spherical harmonic color
    float3 rgb = gaussian.computeSHColor(cam);

    // compute cov3D -> cov2D
    float3x3 cov3D = gaussian.getCov3D();
    float2x2 cov2D = gaussian.cov3Dto2D(cov3D, cam);
    if (gaussian.getDet2D(cov2D) <= 0.0) return;

    // estimate pixel radius
    float pixel_radius = gaussian.getPixelRadius(cov2D);

    // compute inverse cov2D
    float2x2 invCov2D = gaussian.invertCov2D(cov2D);

    // 0.5 to dim + 0.5
    float2 pixel_center = cam.ndcToPixel(screen_center);
    RectBounds bounds = gaussian.getRectTileBounds(cam, pixel_center, pixel_radius);

    // store this data for rasterization
    tiles_touched[tid] = (bounds.maxX - bounds.minX) * (bounds.maxY - bounds.minY);
    tile_ranges_touched[uint2(tid, 0)] = bounds.minX;
    tile_ranges_touched[uint2(tid, 1)] = bounds.maxX;
    tile_ranges_touched[uint2(tid, 2)] = bounds.minY;
    tile_ranges_touched[uint2(tid, 3)] = bounds.maxY;
    radii[tid] = (uint32_t) pixel_radius;
    opacities[tid] = gaussian.opacity;
    centers[uint2(tid, 0)] = pixel_center.x;
    centers[uint2(tid, 1)] = pixel_center.y;
    centers[uint2(tid, 2)] = screen_center.z;
    inv_cov2Ds[uint3(tid, 0, 0)] = invCov2D[0][0];
    inv_cov2Ds[uint3(tid, 0, 1)] = invCov2D[0][1];
    inv_cov2Ds[uint3(tid, 1, 0)] = invCov2D[1][0];
    inv_cov2Ds[uint3(tid, 1, 1)] = invCov2D[1][1];
    rgbs[uint2(tid, 0)] = rgb.x;
    rgbs[uint2(tid, 1)] = rgb.y;
    rgbs[uint2(tid, 2)] = rgb.z;
}

[AutoPyBindCUDA]
[CUDAKernel]
[BackwardDerivative(bwdRenderGaussians)]
public void renderGaussians(
    TensorView<int> gaussian_idxs, 
    TensorView<int> tile_range_starts,
    TensorView<int> tile_range_ends,
    DiffTensorView<float> inv_cov2Ds,
    DiffTensorView<float> centers,
    DiffTensorView<float> rgbs,
    DiffTensorView<float> opacities,
    TensorView<float> result,
    TensorView<int> num_gaussians_used
)
{
    int2 pixel_idx = (cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx()).xy;
    int tile_idx = cudaBlockIdx().y * NUM_TILES_PER_DIM + cudaBlockIdx().x;
    int tile_start_idx = tile_range_starts[tile_idx];
    int tile_end_idx = tile_range_ends[tile_idx];
    int num_threads_in_block = cudaBlockDim().x * cudaBlockDim().y;

    int num_gaussians_left = tile_end_idx - tile_start_idx;
    int num_memory_loops = (num_gaussians_left + num_threads_in_block -  1) / num_threads_in_block;
    int thread_idx = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

    float4 color = float4(0.0, 0.0, 0.0, 1.0);

    int thread_gaussians_used = 0;

    for (int i = 0; i < num_memory_loops; i++) {
        AllMemoryBarrierWithGroupSync();
        int offset = num_threads_in_block * i + thread_idx;
        if (tile_start_idx + offset < tile_end_idx) {
            int gaussian_idx = gaussian_idxs[tile_start_idx + offset];
            float3 g_rgb = float3(rgbs[uint2(gaussian_idx, 0)], rgbs[uint2(gaussian_idx, 1)], rgbs[uint2(gaussian_idx, 2)]);
            float2 g_center = float2(centers[uint2(gaussian_idx, 0)], centers[uint2(gaussian_idx, 1)]);
            float g_opacity = opacities[gaussian_idx];
            float2x2 g_inv_cov = float2x2(
                inv_cov2Ds[uint3(gaussian_idx, 0, 0)], inv_cov2Ds[uint3(gaussian_idx, 0, 1)],
                inv_cov2Ds[uint3(gaussian_idx, 1, 0)], inv_cov2Ds[uint3(gaussian_idx, 1, 1)]
            );
            gaussian_idxs_in_threadgroup[thread_idx] = gaussian_idx;
            gaussians_in_threadgroup[thread_idx] = initProjectedGaussian(gaussian_idx, inv_cov2Ds, centers, rgbs, opacities);
        }
        AllMemoryBarrierWithGroupSync();

        int num_gaussians_left_in_threadgroup = min(num_threads_in_block, num_gaussians_left);
        for (int i = 0; i < num_gaussians_left_in_threadgroup; i++) {
            if (color.w < 0.001) {
                break;
            }
            thread_gaussians_used++;

            float4 pix_gaussian_color = computePixelGaussianColor(gaussians_in_threadgroup[i], pixel_idx);
            color = computeNewPixelColor(pix_gaussian_color, color);
        }
        num_gaussians_left -= num_gaussians_left_in_threadgroup;
    }
    result[uint3(pixel_idx.y, pixel_idx.x, 0)] = color.r;
    result[uint3(pixel_idx.y, pixel_idx.x, 1)] = color.g;
    result[uint3(pixel_idx.y, pixel_idx.x, 2)] = color.b;
    result[uint3(pixel_idx.y, pixel_idx.x, 3)] = 1.0 - color.w;
    num_gaussians_used[uint2(pixel_idx.y, pixel_idx.x)] = thread_gaussians_used;
}

[AutoPyBindCUDA]
[CUDAKernel]
public void bwdRenderGaussians(
    TensorView<int> gaussian_idxs,
    TensorView<int> tile_range_starts,
    TensorView<int> tile_range_ends,
    DiffTensorView<float> inv_cov2Ds,
    DiffTensorView<float> centers,
    DiffTensorView<float> rgbs,
    DiffTensorView<float> opacities,
    TensorView<float> result,
    TensorView<int> num_gaussians_used,
    float4 d_cur_pix,
) {
    int2 pixel_idx = (cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx()).xy;
    int tile_idx = cudaBlockIdx().y * NUM_TILES_PER_DIM + cudaBlockIdx().x;
    int tile_start_idx = tile_range_starts[tile_idx];
    int tile_end_idx = tile_range_ends[tile_idx];
    int num_threads_in_block = cudaBlockDim().x * cudaBlockDim().y;

    int num_gaussians_left = tile_end_idx - tile_start_idx;
    int num_memory_loops = (num_gaussians_left + num_threads_in_block -  1) / num_threads_in_block;
    int thread_idx = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

    // get current pixel color and opacity
    float4 cur_pixel_color = float4(
        result[uint3(pixel_idx.y, pixel_idx.x, 0)],
        result[uint3(pixel_idx.y, pixel_idx.x, 1)],
        result[uint3(pixel_idx.y, pixel_idx.x, 2)],
        result[uint3(pixel_idx.y, pixel_idx.x, 3)]
    );
    int thread_gaussians_used = num_gaussians_used[uint2(pixel_idx.y, pixel_idx.x)];
    
    for (int i = 0; i < num_memory_loops; i++) {
        AllMemoryBarrierWithGroupSync();
        int offset = num_threads_in_block * i + thread_idx;
        if (tile_start_idx + offset < tile_end_idx) {
            int gaussian_idx = gaussian_idxs[tile_end_idx - 1 - offset];
            gaussians_in_threadgroup[thread_idx] = initProjectedGaussian(gaussian_idx, inv_cov2Ds, centers, rgbs, opacities);
        }
        AllMemoryBarrierWithGroupSync();

        int num_gaussians_left_in_threadgroup = min(num_threads_in_block, num_gaussians_left);
        for (int i = 0; i < num_gaussians_left_in_threadgroup; i++) {
            num_gaussians_left--;
            if (num_gaussians_left >= thread_gaussians_used) continue;

            ProjectedGaussian gaussian = gaussians_in_threadgroup[i];
            int gaussian_idx = gaussian_idxs_in_threadgroup[i];

            float4 pix_gaussian_color = computePixelGaussianColor(gaussian, pixel_idx);
            float4 prev_pix_color = computePrevPixelColor(pix_gaussian_color, cur_pixel_color);

            // DL/D_cur_pix_color -> DL/D_prev_pix_color, DL/D_gaussian_color
            DifferentialPair<float4> d_prev_pix = diffPair(cur_pixel_color);
            DifferentialPair<float4> d_gaussian_color = diffPair(pix_gaussian_color);
            bwd_diff(computeNewPixelColor)(d_gaussian_color, d_prev_pix, d_cur_pix);

            // DL/D_gaussian_color -> DL/D_proj_gauss, DL/D_pix_coord
            DifferentialPair<ProjectedGaussian> d_proj_gaussian = diffPair(gaussian);
            DifferentialPair<float2> d_pix_coord = diffPair((float2) pixel_idx);
            bwd_diff(computePixelGaussianColor)(d_proj_gaussian, d_pix_coord, d_gaussian_color.d);

            // DL/D_proj_gauss -> DL/D_proj_gauss_params
            bwd_diff(initProjectedGaussian)(gaussian_idx, inv_cov2Ds, centers, rgbs, opacities, d_proj_gaussian.d);

            // update DL/D_cur_pix_color to DL/D_prev_pix_color
            d_cur_pix = d_prev_pix.d;
        }
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
public void makeDict(
    TensorView<int> tiles_touched_prefix_sum,
    TensorView<int> tile_ranges_touched,
    TensorView<float> centers,
    TensorView<int64_t> tile_and_depth_keys_buf,
    TensorView<int> gauss_idx_vals_buf
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= centers.size(0))
        return;

    int start_idx = tiles_touched_prefix_sum[tid];
    int4 tile_range_touched = int4(
        tile_ranges_touched[uint2(tid, 0)],
        tile_ranges_touched[uint2(tid, 1)],
        tile_ranges_touched[uint2(tid, 2)],
        tile_ranges_touched[uint2(tid, 3)]
    );

    for (int i = tile_range_touched.x; i < tile_range_touched.y; i++) {
        for (int j = tile_range_touched.z; j < tile_range_touched.w; j++) {
            uint64_t tile_idx = j * NUM_TILES_PER_DIM + i;
            tile_idx <<= 32;
            uint32_t depth = reinterpret<uint32_t>(-centers[uint2(tid, 2)]);
            uint64_t key = tile_idx | depth;
            tile_and_depth_keys_buf[start_idx] = reinterpret<int64_t>(key);
            gauss_idx_vals_buf[start_idx] = tid;

            start_idx++;
        }
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
public void prefixSumTiles(
    int total_tiles_touched,
    TensorView<int64_t> sorted_tile_and_depth_keys_buf,
    TensorView<int> tile_range_starts,
    TensorView<int> tile_range_ends
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= total_tiles_touched)
        return;

    uint32_t tile_idx_touched = (uint32_t) (reinterpret<uint64_t>(sorted_tile_and_depth_keys_buf[tid]) >> 32);
    if (tid == 0) {
        tile_range_starts[tile_idx_touched] = 0;
    } else {
        uint32_t prev_tile_idx_touched = (uint32_t) (reinterpret<uint64_t>(sorted_tile_and_depth_keys_buf[tid - 1]) >> 32);
        if (tile_idx_touched != prev_tile_idx_touched) {
            tile_range_ends[prev_tile_idx_touched] = tid;
            tile_range_starts[tile_idx_touched] = tid;
        }
    }
    if (tid == total_tiles_touched - 1) {
        tile_range_ends[tile_idx_touched] = tid + 1;
    }
}
