implementing "../renderer.slang";

import math;
import texture;

public static uint MAX_STEPS = 1024;
public static float EPSILON = 1e-6;

public static float STEP_SIZE = 0.01;
public static float DENSITY_SCALE = 5.0;

/**
 * Perform ray marching through the volume and accumulate color.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param tMax The maximum t value for ray marching (usually the nearest primitive intersection t).
 * @param primColor The color of the nearest primitive intersection.
 * @return The accumulated color after ray marching through the volume.
 */
public float4 volumeSample(float2 uv, RendererUniform uniforms, float4 primColor)
{
    Ray ray = uniforms.camera.generateRay(uv);
    ray = ray.transform(uniforms.volume.invModelMatrix);
    return accumulateColor<4>(ray, uniforms.volume.bound, MAX_STEPS, STEP_SIZE, DENSITY_SCALE, primColor, uniforms.volume.tex);
}

/**
 * Backward pass for volumetric ray marching at a single uv.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the sampled color.
 */
public void volumeSampleBwd(float2 uv, RendererUniform uniforms, float4 outGrad)
{
    // TODO: Student implementation starts here.
    Ray ray = uniforms.camera.generateRay(uv);
    ray = ray.transform(uniforms.volume.invModelMatrix);

    DifferentialPtrPair<DiffTexture3D<float, 4>> diffPtrPair = DifferentialPtrPair<DiffTexture3D<float, 4>>(uniforms.volume.tex, uniforms.volume.dTex);

    bwd_diff(accumulateColor<4>)(ray, uniforms.volume.bound, MAX_STEPS, STEP_SIZE, DENSITY_SCALE, float4(0.0, 0.0, 0.0, 0.0), diffPtrPair, outGrad);

    // TODO: Student implementation ends here.
}

/**
 * Ray march through the volume and accumulate color.
 * @param ray The ray in local space.
 * @param volumeBound The bounding box of the volume in local space.
 * @param maxSteps The maximum number of marching steps.
 * @param stepSize The size of each marching step.
 * @param densityScale The scale factor for the volume density (alpha channel).
 * @param backgroundColor The background color behind the volume
                          (this can be the ambient color if there's no primitive behind the volume, or the color of the primitive hit if there's one).
 * @param data The 3D texture storing the volume data (rgba).
 * @return The accumulated color after ray marching through the volume.
 */
[Differentiable]
public float4 accumulateColor<let N : uint>(
    no_diff Ray ray,
    no_diff BoundingBox volumeBound,
    no_diff uint maxSteps,
    no_diff float stepSize,
    no_diff float densityScale,
    no_diff float4 backgroundColor,
    DiffTexture3D<float, 4> data)
{
    // TODO: Student implementation starts here.
    float2 entry_exit = volumeBound.hit(ray);
    if (all(entry_exit == float2(0.0, 0.0))) {
        return backgroundColor;
    }

    float3 minBound = volumeBound.pMin;
    float3 maxBound = volumeBound.pMax;

    float t = entry_exit.y;
    float3 result = backgroundColor.xyz;
    float alpha = backgroundColor.w;

    [MaxIters(800)]
    while (t >= entry_exit.x) {
        float3 sample_pos = ray.origin + ray.direction * t;
        float3 uvw = (sample_pos - minBound) / (maxBound - minBound);
        
        float4 sampled_tex = sampleTrilinear(data, uvw);

        float a = sampled_tex.w * densityScale * stepSize; 
        float3 c = sampled_tex.xyz;

        result = (1.0 - a) * result + a * c;
        alpha = (1.0 - a) * alpha + a;

        t -= stepSize;
    }

    return float4(result, alpha);
    // TODO: Student implementation ends here.
}