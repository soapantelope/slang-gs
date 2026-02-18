implementing "../material.slang";

import texture;

public struct MaterialField<T>
{
    public T uniformData;
    public bool useTexture;
    public uint filteringMethod;
    public SharedTexture2D[8] mipmap;
    public uint totalLevels;
}

public struct PhysicsBasedParameters
{
    public float3 albedo;
}

public struct PhysicsBasedMaterial
{
    public MaterialField<float3> albedo;
}

public struct PhysicsBasedMaterialTextureBuf
{
    public SharedTexture2DBuffer<float3> albedoTexBuf;
}

public PhysicsBasedParameters sampleMaterial(PhysicsBasedMaterial material)
{
    PhysicsBasedParameters params;
    // Get the albedo value.
    params.albedo = material.albedo.uniformData;
    return params;
}

public PhysicsBasedParameters sampleMaterial(
    PhysicsBasedMaterial material,
    PhysicsBasedMaterialTextureBuf textureBuf,
    float2 uv, float mipLevel, bool visualizeLevelOfDetail)
{
    PhysicsBasedParameters params;
    float clampedMipLevel = clamp(mipLevel, 0.0, float(material.albedo.totalLevels - 1));
    if (visualizeLevelOfDetail)
    {
        // gradation of blue color to light cyan
        float3 level0_color = float3(0.0, 0.0, 255.0) / 255.0;
        float3 level1_color = float3(29.0, 32.0, 255.0) / 255.0;
        float3 level2_color = float3(58.0, 65.0, 255.0) / 255.0;
        float3 level3_color = float3(87.0, 97.0, 255.0) / 255.0;
        float3 level4_color = float3(116.0, 130.0, 255.0) / 255.0;
        float3 level5_color = float3(145.0, 162.0, 255.0) / 255.0;
        float3 level6_color = float3(174.0, 194.0, 255.0) / 255.0;
        float3 level7_color = float3(173.0, 216.0, 255.0) / 255.0;
        float3[8] level_colors = {
            level0_color,
            level1_color,
            level2_color,
            level3_color,
            level4_color,
            level5_color,
            level6_color,
            level7_color,
        };

        float3 color = float3(0.0, 0.0, 0.0);

        float level_lo = floor(clampedMipLevel);
        float level_hi = ceil(clampedMipLevel);

        if (material.albedo.filteringMethod < 2)
        {
            color = level_colors[0];
        }
        else if (material.albedo.filteringMethod == 2)
        {
            uint discreteMipLevel = uint(round(clampedMipLevel));
            color = level_colors[discreteMipLevel];
        }
        else
        {
            float t = clampedMipLevel - level_lo;
            if (level_lo == level_hi)
            {
                color = level_colors[uint(level_lo)];
            }
            else
            {
                color = level_colors[uint(level_lo)] * (1.0 - t) + level_colors[uint(level_hi)] * t;
            }
        }
        params.albedo = color;
        return params;
    }

    if (material.albedo.useTexture)
    {
        if (material.albedo.filteringMethod == 0)
        {
            params.albedo = textureBuf.albedoTexBuf.pointSample(material.albedo.mipmap[0], uv);
        }
        else if (material.albedo.filteringMethod == 1)
        {
            params.albedo = textureBuf.albedoTexBuf.bilinearSample(material.albedo.mipmap[0], uv);
        }
        else if (material.albedo.filteringMethod == 2)
        {
            uint discreteMipLevel = uint(round(clampedMipLevel));
            params.albedo = textureBuf.albedoTexBuf.bilinearSample(material.albedo.mipmap[discreteMipLevel], uv);
        }
        else if (material.albedo.filteringMethod == 3)
        {
            params.albedo = textureBuf.albedoTexBuf.trilinearSample(material.albedo.mipmap, uv, mipLevel, material.albedo.totalLevels);
        }
    }
    else
    {
        params.albedo = material.albedo.uniformData;
    }
    return params;
}
