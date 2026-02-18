implementing "../primitive.slang";
import math;
import texture;

// The size of each marching step.
static float STEP_SIZE = 0.005; // i changed this cause it wasn't working with original amount
// The density threshold to consider a hit.
static float DENSITY_THRESHOLD = 0.25;
// The delta used for finite difference normal estimation.
static float FINITE_DIFF_DELTA = 0.001;

public struct Volume
{
    public BoundingBox bound;
    public SharedTexture3D tex;
    public float4x4 modelMatrix;
    public float4x4 invModelMatrix;

    /**
     * Test ray-volume bounding box intersection.
     * @param ray The ray to test against the volume.
     * @param volumeTexBuf The texture buffer contains all the volumetrically represented models in the scene .
     * @return An Optional containing RayHitResult if the ray hits the volume, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray, SharedTexture3DBuffer<float4> volumeTexBuf)
    {
        // TODO: Student implementation starts here.
        float2 entry_exit = bound.hit(ray);
        if (all(entry_exit == float2(0.0, 0.0))) {
            return {};
        }
        float t = entry_exit.x;
        while (t < entry_exit.y) {
            float3 sample_pos_world = ray.origin + ray.direction * t;
            float3 uvw = clamp(getUVW(sample_pos_world), 0.0, 1.0);
            float4 sampled_tex = volumeTexBuf.trilinearSample(tex, uvw);
            if (sampled_tex.w > DENSITY_THRESHOLD) {
                float3 dx_pos = uvw + float3(FINITE_DIFF_DELTA, 0.0, 0.0);
                float3 dx_neg = uvw - float3(FINITE_DIFF_DELTA, 0.0, 0.0);
                float3 dy_pos = uvw + float3(0.0, FINITE_DIFF_DELTA, 0.0);
                float3 dy_neg = uvw - float3(0.0, FINITE_DIFF_DELTA, 0.0);
                float3 dz_pos = uvw + float3(0.0, 0.0, FINITE_DIFF_DELTA);
                float3 dz_neg = uvw - float3(0.0, 0.0, FINITE_DIFF_DELTA);
                
                float dx = volumeTexBuf.trilinearSample(tex, dx_pos).w - volumeTexBuf.trilinearSample(tex, dx_neg).w;
                float dy = volumeTexBuf.trilinearSample(tex, dy_pos).w - volumeTexBuf.trilinearSample(tex, dy_neg).w;
                float dz = volumeTexBuf.trilinearSample(tex, dz_pos).w - volumeTexBuf.trilinearSample(tex, dz_neg).w;
                
                float3 normal = -normalize(float3(dx, dy, dz));
                return RayHitResult(t, normal);
            }
            t += STEP_SIZE;
        }
        return {};
    }

    /**
     * Get the uvw coordinates of a point in local space.
     * @param p The point in world space.
     * @return The uvw coordinates in [0, 1].
     */
    public float3 getUVW(float3 p)
    {
        // Compute uvw coordinates in [0, 1].
        float3 minBound = this.bound.pMin;
        float3 maxBound = this.bound.pMax;
        return (p - minBound) / (maxBound - minBound);
    }
};
