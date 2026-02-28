implementing "../model.slang";
import math;

public struct Gaussian 
{
    public float3 position;
    public float4 rotation;
    public float3 scale;
    public float3 color;
    public float opacity;

    public float3 computeSHColor(Camera cam) {
        return color;
    }

    // project to pixel XY + NDC depth
    public float3 projectPoint(Camera cam) {
        float3 ndc_point = float3(0.0, 0.0, 0.0); // NDC
        float2 pixel_point = cam.ndcToPixel(ndc_point);
        return float3(pixel_point, ndc_point.z);
    }
    
    public float2x2 cov3Dto2D(float3x3 cov3D, Camera cam) {
        float2x2 cov2D;
        return cov2D;
    }

    public float3x3 getCov3D() {
        float3x3 cov3D;
        return cov3D;
    }

    public float2x2 invertCov2D(float2x2 cov2D) {
        float2x2 invCov2D;
        return invCov2D;
    }

    // gets pixel radius of cov2D
    public int getPixelRadius(float2x2 cov2D) {
        return 5;
    }

    public RectBounds getRectBounds(Camera cam, float3 pixel_center, int pixel_radius) {
        return new RectBounds(uint(pixel_center.x) - pixel_radius, uint(pixel_center.x) + pixel_radius, uint(pixel_center.y) - pixel_radius, uint(pixel_center.y) + pixel_radius);
    }

}

public struct RectBounds {
    public int minX;
    public int maxX;
    public int minY;
    public int maxY;
}
    
public struct StructOfGaussianArrays
{
    public StructuredBuffer<float3> positions;
    public StructuredBuffer<float4> rotations;
    public StructuredBuffer<float3> scales;
    public StructuredBuffer<float3> colors;
    public StructuredBuffer<float> opacities;
}

public void unpackGaussianSplat(
    int tid, 
    StructOfGaussianArrays soa, 
    RWStructuredBuffer<Gaussian> gaussians) 
{
    Gaussian gaussian;
    gaussian.position = soa.positions[tid];
    gaussian.rotation = soa.rotations[tid];
    gaussian.scale = soa.scales[tid];
    gaussian.color = soa.colors[tid];
    gaussian.opacity = soa.opacities[tid];

    // compute cov3d here and store

    gaussians[tid] = gaussian;
}

