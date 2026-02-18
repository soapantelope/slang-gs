module renderer;

import math;
import texture;
import material;
import model;
import primitive;

__include "./renderer/volume_renderer.slang";

public struct RendererUniform
{
    // Camera parameters.
    public Camera camera;
    // Ambient color, used when no ray hits a primitive
    public float4 ambientColor;
    // Ray samples per pixel
    public uint sqrtSpp;
    // Primitives.
    public SharedTexture3DBuffer<float4> surfaceVolumeTexBuf;
    public StructuredBuffer<Volume> surfaceVolumeBuf;
    public uint surfaceVolumeCount;
    public SDFBuffer sdfBuf;
    // Volumetric data.
    public DiffVolume volume;
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
    float4 primColor = uniforms.ambientColor;
    float4 volumeColor = volumeSample(uv, uniforms, primColor);
    return volumeColor;
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

/**
 * Backward pass for volume sampling at a single uv.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the sampled color.
 */
public void sampleVolumeBwd(float2 uv, RendererUniform uniforms, float4 outGrad)
{
    volumeSampleBwd(uv, uniforms, outGrad);
}

/**
 * Backward pass for volume rendering at a pixel.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the rendered pixel color.
 */
public void renderVolumeBwd(uint2 tid, RendererUniform uniforms, float4 outGrad)
{
    // For simplicity, we're not implementing supersampling in the backward pass.
    float2 uv = (float2(tid) + 0.5) / float2(uniforms.camera.canvasSize);
    sampleVolumeBwd(uv, uniforms, outGrad);
}
