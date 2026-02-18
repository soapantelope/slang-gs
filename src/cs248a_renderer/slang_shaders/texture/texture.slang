implementing "../texture.slang";

public struct SharedTexture2DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture2D tex, uint2 texelCoord)
    {
        uint2 texSize = tex.size;
        uint offset = tex.offset;
        uint index = texelCoord.y * texSize.x + texelCoord.x;
        return buffer[offset + index];
    }

    public T pointSample(SharedTexture2D tex, float2 uv)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.
        uint2 uv_scaled = uint2(uv * tex.size);
        outV = getBufferValue(tex, uv_scaled);
        // TODO: Student implementation ends here.
        return outV;
    }

    public T lerpT(T a, T b, float t) {
        return a + (b - a) * (T) t;
    }

    public T bilinearSample(SharedTexture2D tex, float2 uv)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.

        float2 uv_scaled = clamp(uv * tex.size - 0.5, 0.0, tex.size - 1.0);
        uint2 floored_uv = uint2(floor(uv_scaled));

        // float2 uv_scaled = uv * tex.size;
        // uint2 floored_uv = uint2(uv_scaled);

        float2 dist_to_scaled = uv_scaled - float2(floored_uv);

        T point_00 = getBufferValue(tex, floored_uv);
        T point_01 = getBufferValue(tex, floored_uv + uint2(0, 1));
        T point_10 = getBufferValue(tex, floored_uv + uint2(1, 0));
        T point_11 = getBufferValue(tex, floored_uv + uint2(1, 1));

        T lerp_y1 = lerpT(point_00, point_01, dist_to_scaled.y);
        T lerp_y2 = lerpT(point_10, point_11, dist_to_scaled.y);

        outV = lerpT(lerp_y1, lerp_y2, dist_to_scaled.x);

        // TODO: Student implementation ends here.
        return outV;
    }

    public T trilinearSample(SharedTexture2D[] mipmapLevels, float2 uv, float d, uint totalLevels)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.
        float eps = 0.001; // just in case it's the top mip level already
        uint floored_mipLevel = uint(d - eps);

        T sample_up = bilinearSample(mipmapLevels[floored_mipLevel + 1], uv);
        T sample_down = bilinearSample(mipmapLevels[floored_mipLevel], uv);

        outV = lerpT(sample_down, sample_up, (d - floored_mipLevel) / totalLevels);
        
        // TODO: Student implementation ends here.
        return outV;
    }
}

public struct SharedTexture2D
{
    public uint2 size;
    public uint offset;
}

public struct SharedTexture3DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture3D tex, uint3 voxelCoord)
    {
        uint3 texSize = tex.size;
        uint offset = tex.offset;
        uint index = voxelCoord.z * texSize.y * texSize.x +
                     voxelCoord.y * texSize.x +
                     voxelCoord.x;
        return buffer[offset + index];
    }

    /**
     * Sample the texture at the given uvw coordinates using nearest neighbor interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     */
    public T pointSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        if (any(uvw < 0.0) || any(uvw > 1.0)) {
            return {};
        }
        uint3 scaled_uvw = (uint3) (uvw * tex.size);
        return getBufferValue(tex, scaled_uvw);
        // TODO: Student implementation ends here.
    }

    public T lerpT(T a, T b, float t) {
        return a + (b - a) * (T) t;
    }

    /**
     * Sample the texture at the given uvw coordinates using trilinear interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     **/
    public T trilinearSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        if (any(uvw < 0.0) || any(uvw > 1.0)) {
            return {};
        }
        
        float3 scaled_uvw = uvw * tex.size;
        uint3 floored_uvw = uint3(scaled_uvw);
        float3 dist_to_floored = scaled_uvw - floored_uvw;

        T point_000 = getBufferValue(tex, floored_uvw);
        T point_010 = getBufferValue(tex, floored_uvw + uint3(0, 1, 0));
        T point_011 = getBufferValue(tex, floored_uvw + uint3(0, 1, 1));
        T point_001 = getBufferValue(tex, floored_uvw + uint3(0, 0, 1));
        T point_100 = getBufferValue(tex, floored_uvw + uint3(1, 0, 0));
        T point_110 = getBufferValue(tex, floored_uvw + uint3(1, 1, 0));
        T point_111 = getBufferValue(tex, floored_uvw + uint3(1, 1, 1));
        T point_101 = getBufferValue(tex, floored_uvw + uint3(1, 0, 1));
        
        T lerp_z1 = lerpT(point_000, point_001, dist_to_floored.z);
        T lerp_z2 = lerpT(point_010, point_011, dist_to_floored.z);
        T lerp_z3 = lerpT(point_100, point_101, dist_to_floored.z);
        T lerp_z4 = lerpT(point_110, point_111, dist_to_floored.z);

        T lerp_y1 = lerpT(lerp_z1, lerp_z2, dist_to_floored.y);
        T lerp_y2 = lerpT(lerp_z3, lerp_z4, dist_to_floored.y);

        T lerp_x = lerpT(lerp_y1, lerp_y2, dist_to_floored.x);

        return lerp_x;

        // TODO: Student implementation ends here.
    }
}

public struct SharedTexture3D
{
    public uint3 size;
    public uint offset;
}

