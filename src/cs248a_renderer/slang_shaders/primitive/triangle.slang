implementing "../primitive.slang";
import math;

public struct Triangle : IBoundingBox, IRayCastable
{
    public float3[3] vertices;
    public float3[3] normals;
    public float2[3] uvs;
    public float3[3] colors;
    public uint materialId;
    public uint offset;

    public BoundingBox boundingBox()
    {
        return BoundingBox(float3(0.0), float3(0.0));
    }

    /* Returns the average color of the triangle. */
    public float3 averageColor()
    {
        return (this.colors[0] + this.colors[1] + this.colors[2]) / 3.0;
    }

    /**
     * Calculates the barycentric coordinates of a point on the triangle.
     * @param p The point on the triangle to calculate the barycentric coordinates for.
     * @return The barycentric coordinates of the point.
     */
    public float3 calculateBarycentricCoord(float3 p)
    {
        const float EPSILON = 1e-6;
        float u = 0.0;
        float v = 0.0;
        float w = 0.0;

        // TODO: Student implementation starts here.
        
        // note im not multiplying by 0.5 for areas since it's ratios

        float3 ac = vertices[2] - vertices[0];
        float3 ab = vertices[1] - vertices[0];
        float3 bc = vertices[2] - vertices[1];

        float3 ap = p - vertices[0];

        float dist_ab_p = length(cross(ap, ab)) / length(ab);
        float area_ab = dist_ab_p * length(ab);

        float dist_ac_p = length(cross(ap, ac)) / length(ac);
        float area_ac = dist_ac_p * length(ac);
        
        float tri_height = length(cross(ac, ab)) / length(ab);
        float tri_area = tri_height * length(ab);

        w = area_ab / tri_area;
        v = area_ac / tri_area;
        u = 1 - w - v;

        // TODO: Student implementation ends here.

        return float3(u, v, w);
    }

    public float2 getUV(float3 barycentricCoord)
    {
        float2 uv = float2(0.0);

        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        uv = u * this.uvs[0] + v * this.uvs[1] + w * this.uvs[2];
        
        return uv;
    }

    public float3 getColor(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 color = u * this.colors[0] + v * this.colors[1] + w * this.colors[2];
        return color;
    }

    public float3 getNormal(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 normal = u * this.normals[0] + v * this.normals[1] + w * this.normals[2];
        return normalize(normal);
    }

    /**
     * Test ray-triangle intersection.
     * @param ray The ray to test against the triangle.
     * @return An Optional containing RayHitResult if the ray hits the triangle, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray)
    {
        // TODO: Student implementation starts here.

        float3 ab = vertices[1] - vertices[0];
        float3 bc = vertices[2] - vertices[1];
        float3 ca = vertices[0] - vertices[2];

        float3 normal = normalize(cross(ab, bc));
        float offset = dot(normal, vertices[0]);
        
        float t = (offset - dot(normal, ray.origin)) / dot(normal, ray.direction);
        float3 x = ray.origin + ray.direction * t;

        float3 ab_normal = normalize(cross(ab, normal));
        float3 bc_normal = normalize(cross(bc, normal));
        float3 ca_normal = normalize(cross(ca, normal));

        float3 ax = normalize(x - vertices[0]);
        float3 bx = normalize(x - vertices[1]);
        float3 cx = normalize(x - vertices[2]);

        float eps = 0.001;

        if (dot(ax, ab_normal) < eps && dot(bx, bc_normal) < eps && dot(cx, ca_normal) < eps) {
            RayHitResult hit = {t, normal};
            return hit;
        }
        return {};

        // TODO: Student implementation ends here.
    }

    public uint getMaterialId()
    {
        return this.materialId + this.offset;
    }
}
