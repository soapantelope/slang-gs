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
    TensorView<int32_t> pixels_touched,
    TensorView<int32_t> pixel_ranges_touched,
    TensorView<int32_t> radii,
    TensorView<float> centers,
    TensorView<float> inv_cov2Ds,
    TensorView<float> rgbs
)
{
    Gaussian gaussian = uniforms.gaussianBuf[tid];

    // compute spherical harmonic color
    float3 rgb = gaussian.computeSHColor(uniforms.camera);

    // project center to screen
    float3 pixel_center = gaussian.projectPoint(uniforms.camera);

    // compute cov3D -> cov2D
    float3x3 cov3D = gaussian.getCov3D();
    float2x2 cov2D = gaussian.cov3Dto2D(cov3D, uniforms.camera);

    // estimate pixel radius
    int pixel_radius = gaussian.getPixelRadius(cov2D);

    // compute inverse cov2D
    float2x2 invCov2D = gaussian.invertCov2D(cov2D);

    RectBounds bounds = gaussian.getRectBounds(uniforms.camera, pixel_center, pixel_radius);

    // store this data for rasterization
    pixels_touched[tid] = (bounds.maxX - bounds.minX) * (bounds.maxY - bounds.minY);

    pixel_ranges_touched[uint2(tid, 0)] = bounds.minX;
    pixel_ranges_touched[uint2(tid, 1)] = bounds.maxX;
    pixel_ranges_touched[uint2(tid, 2)] = bounds.minY;
    pixel_ranges_touched[uint2(tid, 3)] = bounds.maxY;

    radii[tid] = pixel_radius;

    centers[uint2(tid, 0)] = pixel_center.x;
    centers[uint2(tid, 1)] = pixel_center.y;
    centers[uint2(tid, 2)] = pixel_center.z;

    inv_cov2Ds[uint3(tid, 0, 0)] = invCov2D[0][0];
    inv_cov2Ds[uint3(tid, 0, 1)] = invCov2D[0][1];
    inv_cov2Ds[uint3(tid, 1, 0)] = invCov2D[1][0];
    inv_cov2Ds[uint3(tid, 1, 1)] = invCov2D[1][1];

    rgbs[uint2(tid, 0)] = rgb.x;
    rgbs[uint2(tid, 1)] = rgb.y;
    rgbs[uint2(tid, 2)] = rgb.z;
}

// tid is pixel coordinate
public float4 renderGaussians(
    uint2 tid, 
    RendererUniform uniforms, 
    TensorView<int32_t> gaussian_idxs, 
    TensorView<int32_t> pixel_prefix_sums,
    TensorView<float> rgbs
)
{
    // get gaussians that affect pixel
    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    uint pixel_idx = tid.x * uniforms.camera.canvasSize.x + tid.y;
    for (int i = pixel_prefix_sums[pixel_idx]; i < pixel_prefix_sums[pixel_idx + 1]; i++) {
        // evaluate gaussian density based on inverse cov2D
        // alpha composite front-to-back
        // exit early if opaque
        color = float4(rgbs[(i, 0)], rgbs[(i, 1)], rgbs[(i, 2)], uniforms.gaussianBuf[i].opacity);
    }
    return color;
}