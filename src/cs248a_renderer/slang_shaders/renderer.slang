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
    RWStructuredBuffer<int> pixels_touched,
    RWStructuredBuffer<int4> pixel_ranges_touched,
    RWStructuredBuffer<int> radii,
    RWStructuredBuffer<float3> centers,
    RWStructuredBuffer<float2x2> inv_cov2Ds,
    RWStructuredBuffer<float3> rgbs
)
{
    Gaussian gaussian = uniforms.gaussianBuf[tid];

    pixels_touched[tid] = 0;
    pixel_ranges_touched[tid] = int4(0, 0, 0, 0);

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
    RectBounds bounds = gaussian.getRectBounds(uniforms.camera, pixel_center, pixel_radius);

    // store this data for rasterization
    pixels_touched[tid] = (bounds.maxX - bounds.minX) * (bounds.maxY - bounds.minY);
    pixel_ranges_touched[tid] = int4(bounds.minX, bounds.maxX, bounds.minY, bounds.maxY);
    radii[tid] = (uint32_t) pixel_radius;
    centers[tid] = screen_center;
    inv_cov2Ds[tid] = invCov2D;
    rgbs[tid] = rgb;
}

// tid is pixel coordinate
public float4 renderGaussians(
    uint2 tid, 
    RendererUniform uniforms, 
    StructuredBuffer<int> gaussian_idxs, 
    StructuredBuffer<int> pixel_range_starts,
    StructuredBuffer<int> pixel_range_ends,
    StructuredBuffer<float2x2> inv_cov2Ds,
    StructuredBuffer<float3> centers,
    StructuredBuffer<float3> rgbs
)
{
    float opacity = 1.0;
    float4 color = float4(0.0, 0.0, 0.0, opacity);
    uint pixel_idx = tid.y * uniforms.camera.canvasSize.x + tid.x;
    for (int i = pixel_range_starts[pixel_idx]; i < pixel_range_ends[pixel_idx]; i++) {
        int gaussian_idx = gaussian_idxs[i];
        Gaussian gaussian = uniforms.gaussianBuf[gaussian_idx];
        float3 gaussian_color = rgbs[gaussian_idx];

        float2 gaussian_screen_center = uniforms.camera.ndcToPixel(centers[gaussian_idx]);
        float2 x = float2(tid.x - gaussian_screen_center.x, tid.y - gaussian_screen_center.y);
        float gaussian_opacity = exp(-0.5 * dot(x, mul(inv_cov2Ds[gaussian_idx], x))) * gaussian.opacity;
        float4 premultiplied_gauss_color = float4(gaussian_color * gaussian_opacity, gaussian_opacity);
        color = premultiplied_gauss_color + (1 - gaussian_opacity) * color;
    }
    return color;
}

public void makeDict(
    uint tid,
    RendererUniform uniforms,
    StructuredBuffer<int> pixels_touched_prefix_sum,
    StructuredBuffer<int4> pixel_ranges_touched,
    StructuredBuffer<float3> centers,
    RWStructuredBuffer<uint64_t> pix_and_depth_keys_buf,
    RWStructuredBuffer<int> gauss_idx_vals_buf
) {
    int start_idx = pixels_touched_prefix_sum[tid];
    int4 pixel_range_touched = pixel_ranges_touched[tid];

    for (int i = pixel_range_touched.x; i < pixel_range_touched.y; i++) {
        for (int j = pixel_range_touched.z; j < pixel_range_touched.w; j++) {
            uint64_t pixel_idx = j * uniforms.camera.canvasSize.x + i;
            pixel_idx <<= 32;
            uint32_t depth = ~reinterpret<uint32_t>(-centers[tid].z);
            uint64_t key = pixel_idx | depth;
            pix_and_depth_keys_buf[start_idx] = key;
            gauss_idx_vals_buf[start_idx] = tid;

            start_idx++;
        }
    }
}

public void prefixSumPixels(
    uint tid, // index of the key
    int total_pixels_touched,
    StructuredBuffer<uint64_t> sorted_pix_and_depth_keys_buf,
    RWStructuredBuffer<int> pixel_range_starts,
    RWStructuredBuffer<int> pixel_range_ends,
) {
    uint32_t pixel_idx_touched = (uint32_t) (sorted_pix_and_depth_keys_buf[tid] >> 32);
    if (tid == 0) {
        pixel_range_starts[pixel_idx_touched] = 0;
    } else {
        uint32_t prev_pix_idx_touched = (uint32_t) (sorted_pix_and_depth_keys_buf[tid - 1] >> 32);
        if (pixel_idx_touched != prev_pix_idx_touched) {
            pixel_range_ends[prev_pix_idx_touched] = tid;
            pixel_range_starts[pixel_idx_touched] = tid;
        }
    }
    if (tid == total_pixels_touched - 1) {
        pixel_range_ends[pixel_idx_touched] = tid + 1;
    }
}