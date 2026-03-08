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

    // gaussian stuff
    public StructuredBuffer<Gaussian> gaussianBuf;
    public uint gaussianCount;

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

// tid is gaussian index
public void preprocessGaussians(
    uint tid, 
    RendererUniform uniforms,
    RWStructuredBuffer<int> tiles_touched,
    RWStructuredBuffer<int4> tile_ranges_touched,
    RWStructuredBuffer<int> radii,
    RWStructuredBuffer<float> opacities,
    RWStructuredBuffer<float3> centers,
    RWStructuredBuffer<float2x2> inv_cov2Ds,
    RWStructuredBuffer<float3> rgbs
)
{
    Gaussian gaussian = uniforms.gaussianBuf[tid];

    tiles_touched[tid] = 0;
    tile_ranges_touched[tid] = int4(0, 0, 0, 0);

    // project center to screen space (NDC)
    float3 screen_center = gaussian.projectPoint(uniforms.camera);
    if (screen_center.z >= -0.1) return;

    // compute spherical harmonic color
    float3 rgb = gaussian.computeSHColor(uniforms.camera);

    // compute cov3D -> cov2D
    float3x3 cov3D = gaussian.getCov3D();
    float2x2 cov2D = gaussian.cov3Dto2D(cov3D, uniforms.camera);
    if (gaussian.getDet2D(cov2D) <= 0.0) return;

    // estimate pixel radius
    float pixel_radius = gaussian.getPixelRadius(cov2D);

    // compute inverse cov2D
    float2x2 invCov2D = gaussian.invertCov2D(cov2D);

    // 0.5 to dim + 0.5
    float2 pixel_center = uniforms.camera.ndcToPixel(screen_center);
    RectBounds bounds = gaussian.getRectTileBounds(uniforms.camera, pixel_center, pixel_radius);

    // store this data for rasterization
    tiles_touched[tid] = (bounds.maxX - bounds.minX) * (bounds.maxY - bounds.minY);
    tile_ranges_touched[tid] = int4(bounds.minX, bounds.maxX, bounds.minY, bounds.maxY);
    radii[tid] = (uint32_t) pixel_radius;
    opacities[tid] = gaussian.opacity;
    centers[tid] = float3(pixel_center, screen_center.z);
    inv_cov2Ds[tid] = invCov2D;
    rgbs[tid] = rgb;
}

[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
public float4 renderGaussians(
    uint2 tid, 
    RendererUniform uniforms, 
    StructuredBuffer<int> gaussian_idxs, 
    StructuredBuffer<int> tile_range_starts,
    StructuredBuffer<int> tile_range_ends,
    StructuredBuffer<float2x2> inv_cov2Ds,
    StructuredBuffer<float3> centers,
    StructuredBuffer<float3> rgbs,
    StructuredBuffer<float> opacities
)
{
    float transmittance = 1.0;
    float3 color = float3(0.0, 0.0, 0.0);
    uint2 tile = tid / (uint2) uniforms.camera.getTileDims(); 
    uint tile_idx = tile.y * uniforms.camera.num_tiles.x + tile.x;

    for (int i = tile_range_ends[tile_idx] -  1; i >= tile_range_starts[tile_idx]; i--) {
        if (transmittance < 0.001) break;

        int gaussian_idx = gaussian_idxs[i];
        float3 gaussian_color = rgbs[gaussian_idx];
        float gaussian_opacity = opacities[gaussian_idx];

        float2 gaussian_screen_center = centers[gaussian_idx].xy;
        float2 x = float2(tid.x - gaussian_screen_center.x, tid.y - gaussian_screen_center.y);

        float alpha = exp(-0.5 * dot(x, mul(inv_cov2Ds[gaussian_idx], x))) * gaussian_opacity;
        color += transmittance * alpha * gaussian_color;
        transmittance *= (1.0 - alpha);
    }
    return float4(color, 1.0 - transmittance);
}

public void makeDict(
    uint tid,
    RendererUniform uniforms,
    StructuredBuffer<int> tiles_touched_prefix_sum,
    StructuredBuffer<int4> tile_ranges_touched,
    StructuredBuffer<float3> centers,
    RWStructuredBuffer<uint64_t> tile_and_depth_keys_buf,
    RWStructuredBuffer<int> gauss_idx_vals_buf
) {
    int start_idx = tiles_touched_prefix_sum[tid];
    int4 tile_range_touched = tile_ranges_touched[tid];

    for (int i = tile_range_touched.x; i < tile_range_touched.y; i++) {
        for (int j = tile_range_touched.z; j < tile_range_touched.w; j++) {
            uint64_t tile_idx = j * uniforms.camera.num_tiles.x + i;
            tile_idx <<= 32;
            uint32_t depth = ~reinterpret<uint32_t>(-centers[tid].z);
            uint64_t key = tile_idx | depth;
            tile_and_depth_keys_buf[start_idx] = key;
            gauss_idx_vals_buf[start_idx] = tid;

            start_idx++;
        }
    }
}

public void prefixSumTiles(
    uint tid, // index of the key
    int total_tiles_touched,
    StructuredBuffer<uint64_t> sorted_tile_and_depth_keys_buf,
    RWStructuredBuffer<int> tile_range_starts,
    RWStructuredBuffer<int> tile_range_ends,
) {
    uint32_t tile_idx_touched = (uint32_t) (sorted_tile_and_depth_keys_buf[tid] >> 32);
    if (tid == 0) {
        tile_range_starts[tile_idx_touched] = 0;
    } else {
        uint32_t prev_tile_idx_touched = (uint32_t) (sorted_tile_and_depth_keys_buf[tid - 1] >> 32);
        if (tile_idx_touched != prev_tile_idx_touched) {
            tile_range_ends[prev_tile_idx_touched] = tid;
            tile_range_starts[tile_idx_touched] = tid;
        }
    }
    if (tid == total_tiles_touched - 1) {
        tile_range_ends[tile_idx_touched] = tid + 1;
    }
}