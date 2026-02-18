implementing "../primitive.slang";
import math;

public struct SDFBuffer
{
    public StructuredBuffer<SphereSDF> spheres;
    public StructuredBuffer<CubeSDF> cubes;
    public uint sphereCount;
    public uint cubeCount;
    public CustomSDF customSDF;
    public bool renderCustomSDF;

    /**
     * Compute the minimum signed distance from a point to all SDF primitives in the buffer.
     * @param point The point from which to compute the signed distance.
     * @return A Tuple containing the minimum signed distance and the normal of the closest SDF primitive.
     */
    public Tuple<float, float3> sample(float3 point)
    {
        float minDist = float.maxValue;
        float3 closestNormal = float3(0.0, 0.0, 0.0);
        for (uint i = 0; i < this.sphereCount; i++)
        {
            float dist = this.spheres[i].sdf(point);
            if (dist < minDist)
            {
                minDist = dist;
                closestNormal = this.spheres[i].normal(point);
            }
        }
        for (uint i = 0; i < this.cubeCount; i++)
        {
            float dist = this.cubes[i].sdf(point);
            if (dist < minDist)
            {
                minDist = dist;
                closestNormal = this.cubes[i].normal(point);
            }
        }
        if (this.renderCustomSDF)
        {
            float dist = this.customSDF.sdf(point);
            if (dist < minDist)
            {
                minDist = dist;
                closestNormal = this.customSDF.normal(point);
            }
        }
        return Tuple<float, float3>(minDist, closestNormal);
    }
}

// Compose SDFs together using union, intersection, and difference operations
[Differentiable]
public float sdfUnion(float d1, float d2)
{
    return min(d1, d2);
}

[Differentiable]
public float sdfIntersection(float d1, float d2)
{
    return max(d1, d2);
}

[Differentiable]
public float sdfDifference(float d1, float d2)
{
    return max(d1, -d2);
}

public interface ISDF
{
    /**
     * Compute the signed distance from a po˝int to the surface of the primitive.
     * @param point The point from which to compute the signed distance.
     * @return The signed distance from the point to the surface of the primitive.
     */
    public float sdf(float3 point);

    /**
     * The normal of the SDF at a given point.
     * @param point The point at which to compute the normal.
     * @return The normal vector at the given point.
     */
    public float3 normal(float3 point);
}

public struct SphereSDF : ISDF
{
    public float3 center;
    public float radius;
    public float4x4 invModelMatrix;

    [NoDiffThis, Differentiable]
    public float sdf(float3 point)
    {
        // TODO: Student implementation starts here.
        float3 point_transformed = mul(invModelMatrix, float4(point, 0.0)).xyz;
        return length(point_transformed - center) - radius; // I should be doing a transformation from local->world but can't given no inverse() function
        // TODO: Student implementation ends here.
    }

    public float3 normal(float3 point)
    {
        // TODO: Student implementation starts here.
        float3 point_transformed = mul(invModelMatrix, float4(point, 0.0)).xyz;
        return normalize(point_transformed - center); // I should be doing a transformation from local->world but can't given no inverse() function
        // TODO: Student implementation ends here.
    }
}

public struct CubeSDF : ISDF
{
    public float3 size;
    public float4x4 invModelMatrix;

    [NoDiffThis, Differentiable]
    public float sdf(float3 point)
    {
        // TODO: Student implementation starts here.
        float3 point_transformed = mul(invModelMatrix, float4(point, 0.0)).xyz;
        float3 radii = size;
        float3 dist_to_edges = abs(point_transformed) - radii;
        float dist = length(max(dist_to_edges, 0.0)) + min(max(max(dist_to_edges.x, dist_to_edges.y), dist_to_edges.z), 0.0);
        return dist; // I should be doing a transformation from local->world but can't given no inverse() function
        // TODO: Student implementation ends here.
    }

    public float3 normal(float3 point)
    {
        // TODO: Student implementation starts here.
        float3 point_transformed = mul(invModelMatrix, float4(point, 0.0)).xyz;
        float3 radii = size;
        float3 dist_to_edges = abs(point_transformed) - radii;

        if (any(dist_to_edges > 0.0)) {
            return normalize(point_transformed * max(dist_to_edges, 0.0));
        }

        if (dist_to_edges.x > max(dist_to_edges.y, dist_to_edges.z)) {
            return normalize(point_transformed * float3(dist_to_edges.x, 0.0, 0.0));
        } if (dist_to_edges.y > dist_to_edges.z) {
            return normalize(point_transformed * float3(0.0, dist_to_edges.y, 0.0));
        }
        return normalize(point_transformed * float3(0.0, 0.0, dist_to_edges.z)); // I should be doing a transformation from local->world but can't given no inverse() function
        // TODO: Student implementation ends here.
    }
}

public struct CustomSDF : ISDF
{
    public float3 cubeSize;
    public float sphereRadius;
    public float4x4 invModelMatrix;

    [NoDiffThis, Differentiable]
    public float sdf(float3 point)
    {
        float3 localPoint = mul(this.invModelMatrix, float4(point, 1.0)).xyz;
        CubeSDF outerCube;
        outerCube.size = this.cubeSize;
        outerCube.invModelMatrix = float4x4(
            float4(1.0, 0.0, 0.0, 0.0),
            float4(0.5, 1.0, 0.0, 0.0),
            float4(0.0, 0.0, 1.0, 0.0),
            float4(0.0, 0.0, 0.0, 1.0));
        CubeSDF otherCube;
        otherCube.size = this.cubeSize / 4;
        otherCube.invModelMatrix = float4x4(
            float4(0.1, 1.0, 0.5, 0.0),
            float4(0.0, 0.1, 0.0, 0.0),
            float4(0.0, 0.0, 0.5, 0.0),
            float4(0.0, 0.0, 0.0, 1.0));
        SphereSDF innerSphere;
        innerSphere.center = float3(0.0, 1.0, 0.0);
        innerSphere.radius = this.sphereRadius;
        innerSphere.invModelMatrix = float4x4(
            float4(1.0, 0.0, 0.0, 0.0),
            float4(0.0, 1.0, 0.0, 0.0),
            float4(0.0, 0.0, 1.0, 0.0),
            float4(0.0, 0.0, 0.0, 1.0));
        SphereSDF otherSphere;
        otherSphere.center = float3(0.0, -0.7, 0.0);
        otherSphere.radius = this.sphereRadius;
        otherSphere.invModelMatrix = float4x4(
            float4(1.0, 0.0, 0.0, 0.0),
            float4(0.0, 1.0, 0.0, 0.0),
            float4(0.0, 0.0, 1.0, 0.0),
            float4(0.0, 0.0, 0.0, 1.0)); 
        float outerDist = outerCube.sdf(localPoint);
        float otherCubeDist = otherCube.sdf(localPoint);
        float innerDist = innerSphere.sdf(localPoint);
        float otherSphereDist = otherSphere.sdf(localPoint);
        // return sdfDifference(outerDist, innerDist);
        return sdfIntersection(sdfDifference(sdfDifference(outerDist, innerDist), otherCubeDist), otherSphereDist);
    }

    [Differentiable]
    static float sdfHelper(no_diff CustomSDF sdf, float3 point)
    {
        return sdf.sdf(point);
    }

    public float3 normal(float3 point)
    {
        DifferentialPair<float3> dp = diffPair(point);
        bwd_diff(CustomSDF.sdfHelper)(this, dp, 1.0);
        return normalize(dp.d);
    }
}
