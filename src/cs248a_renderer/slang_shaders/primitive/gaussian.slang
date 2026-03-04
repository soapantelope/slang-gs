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

    public float3 mul_homogeneous(float4x4 mat, float3 vec) {
        float4 res_homogeneous = mul(mat, float4(vec, 1.0));
        res_homogeneous /= res_homogeneous.w;
        return res_homogeneous.xyz;
    }

    // project to NDC UV XY + view depth
    public float3 projectPoint(Camera cam) {
        float3 view_point = mul_homogeneous(cam.viewMatrix, position);
        float3 cam_point = mul_homogeneous(mul(cam.projMatrix, cam.viewMatrix), position);
        return float3(cam_point.xy, view_point.z);
    }
    
    // how pixel UV changes wrt view coords
    public float3x3 getJacobian(Camera cam) {
        float2 fov = cam.getFOV();
        float hx = cam.canvasSize.x / (2.0 * tan(fov.x / 2.0));    
        float hy = cam.canvasSize.y / (2.0 * tan(fov.y / 2.0));

        float3 view_pos = mul_homogeneous(cam.viewMatrix, position);
        
        float3x3 jacobian = float3x3(
            hx / view_pos.z, 0.0, -(hx * view_pos.x) / (view_pos.z * view_pos.z),
            0.0, hy / view_pos.z, -(hy * view_pos.y) / (view_pos.z * view_pos.z),
            0.0, 0.0, 0.0
        );

        return jacobian;
    }

    public float2x2 cov3Dto2D(float3x3 cov3D, Camera cam) {
        float3x3 view_to_screen_jacobian = getJacobian(cam);
        float3x3 left = mul(view_to_screen_jacobian, float3x3(cam.viewMatrix));
        float2x2 cov2D = float2x2(mul(left, mul(cov3D, linalg.transpose(left))));
        return cov2D;
    }

    public float3x3 getCov3D() {
        float4 rot_quat_norm = normalize(rotation);
        float w = rot_quat_norm[0];
        float x = rot_quat_norm[1];
        float y = rot_quat_norm[2];
        float z = rot_quat_norm[3];

        float3x3 rot_mat = float3x3(
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
        );
        float3x3 scale_mat = float3x3(
            scale[0], 0.0, 0.0,
            0.0, scale[1], 0.0,
            0.0, 0.0, scale[2]
        );

        float3x3 transform_mat = mul(rot_mat, scale_mat);

        float3x3 cov3D = mul(transform_mat, linalg.transpose(transform_mat));
        return cov3D;
    }

    public float getDet2D(float2x2 mat) {
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    }

    public float2x2 invertCov2D(float2x2 cov2D) {
        float2x2 invCov2D = float2x2(
            cov2D[1][1], -cov2D[0][1],
            -cov2D[1][0], cov2D[0][0]
        ) / getDet2D(cov2D);
        return invCov2D;
    }

    // gets pixel radius of cov2D
    public float getPixelRadius(float2x2 cov2D) {
        float half_trace = (cov2D[0][0] + cov2D[1][1]) / 2.0;
        float eigenvalue_1 = half_trace + sqrt(half_trace * half_trace - getDet2D(cov2D));
        float eigenvalue_2 = half_trace - sqrt(half_trace * half_trace - getDet2D(cov2D));
        return ceil(3.0 * sqrt(max(eigenvalue_1, eigenvalue_2)));
    }

    public RectBounds getRectBounds(Camera cam, float2 pixel_center, float pixel_radius) {
        return new RectBounds(
            clamp(uint(pixel_center.x - pixel_radius), 0.0, cam.canvasSize.x), 
            clamp(uint(pixel_center.x + pixel_radius), 0.0, cam.canvasSize.x),
            clamp(uint(pixel_center.y - pixel_radius), 0.0, cam.canvasSize.y), 
            clamp(uint(pixel_center.y + pixel_radius), 0.0, cam.canvasSize.y)
        );
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

