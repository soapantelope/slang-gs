implementing "../material.slang";
import math;
import texture;

public struct SimpleMaterial
{
    public float3 shade(Ray ray, RayHitResult hitResult)
    {
        float NdotL = max(dot(normalize(hitResult.normal), normalize(-ray.direction)), 0.0);
        return float3(1.0, 1.0, 1.0) * NdotL;
    }
}

public struct SimpleMaterialAlbedo
{
    public SharedTexture2D[8] albedoMipmap;
    public uint totalLevels;

    public float3 shade(float3 rayDir, float3 normal, float3 albedo)
    {
        float NdotL = max(dot(normalize(normal), normalize(-rayDir)), 0.0);
        return albedo.rgb * NdotL;
    }
}
