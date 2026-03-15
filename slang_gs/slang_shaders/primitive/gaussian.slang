implementing "../model.slang";
import math;

public struct ProjectedGaussian : IDifferentiable
{
    public float3 rgb;
    public float2 screen_center;
    public float opacity;
    public float2x2 inv_cov2D;
}

[Differentiable]
public float4 computePixelGaussianColor(ProjectedGaussian gaussian, float2 pixel_idx) {
    float2 x = float2(pixel_idx.x - gaussian.screen_center.x, pixel_idx.y - gaussian.screen_center.y);
    float alpha = exp(-0.5 * dot(x, mul(gaussian.inv_cov2D, x))) * gaussian.opacity;
    float3 res_rgb = alpha * gaussian.rgb;
    return float4(res_rgb, alpha);
}

[Differentiable]
public ProjectedGaussian initProjectedGaussian(
    int gaussian_idx,  
    DiffTensorView<float> inv_cov2Ds,
    DiffTensorView<float> centers,
    DiffTensorView<float> rgbs,
    DiffTensorView<float> opacities,
) {
    ProjectedGaussian projected_gaussian;
    projected_gaussian.rgb = float3(rgbs[uint2(gaussian_idx, 0)], rgbs[uint2(gaussian_idx, 1)], rgbs[uint2(gaussian_idx, 2)]);
    projected_gaussian.screen_center = float2(centers[uint2(gaussian_idx, 0)], centers[uint2(gaussian_idx, 1)]);
    projected_gaussian.opacity = opacities[gaussian_idx];
    projected_gaussian.inv_cov2D = float2x2(
        inv_cov2Ds[uint3(gaussian_idx, 0, 0)], inv_cov2Ds[uint3(gaussian_idx, 0, 1)],
        inv_cov2Ds[uint3(gaussian_idx, 1, 0)], inv_cov2Ds[uint3(gaussian_idx, 1, 1)]
    );
    return projected_gaussian;
}

[Differentiable]
public float4 computeNewPixelColor(float4 pix_gaussian_color, float4 orig_pix_color) {
    float3 rgb = orig_pix_color.rgb + orig_pix_color.w * pix_gaussian_color.rgb;
    float transmittance = orig_pix_color.w * (1.0 - pix_gaussian_color.w);
    return float4(rgb, transmittance);
}

public float4 computePrevPixelColor(float4 pix_gaussian_color, float4 cur_pix_color) {
    float orig_transmittance = cur_pix_color.w / (1.0 - pix_gaussian_color.w);
    float3 orig_rgb = cur_pix_color.rgb - orig_transmittance * pix_gaussian_color.rgb;
    return float4(orig_rgb, orig_transmittance);
}

public struct Gaussian 
{
    public float3 position;
    public float4 rotation;
    public float3 scale;
    public float3 color;
}

public float3 computeSHColor(Gaussian g, Camera cam) {
    return clamp(g.color + float3(0.2, 0.2, 0.2), 0.0, 1.0);
}

public float3 mul_homogeneous(float4x4 mat, float3 vec) {
    float4 res_homogeneous = mul(mat, float4(vec, 1.0));
    res_homogeneous /= res_homogeneous.w;
    return res_homogeneous.xyz;
}

// project to NDC UV XY + view depth
public float3 projectPoint(Gaussian g, Camera cam) {
    float3 view_point = mul_homogeneous(cam.viewMatrix, g.position);
    float3 cam_point = mul_homogeneous(mul(cam.projMatrix, cam.viewMatrix), g.position);
    return float3(cam_point.xy, view_point.z);
}

// how pixel UV changes wrt view coords
public float3x3 getJacobian(Gaussian g, Camera cam) {
    float2 fov = cam.getFOV();
    float hx = cam.canvasSize.x / (2.0 * tan(fov.x / 2.0));    
    float hy = cam.canvasSize.y / (2.0 * tan(fov.y / 2.0));

    float3 view_pos = mul_homogeneous(cam.viewMatrix, g.position);
    
    float3x3 jacobian = float3x3(
        hx / view_pos.z, 0.0, -(hx * view_pos.x) / (view_pos.z * view_pos.z),
        0.0, hy / view_pos.z, -(hy * view_pos.y) / (view_pos.z * view_pos.z),
        0.0, 0.0, 0.0
    );

    return jacobian;
}

public float2x2 cov3Dto2D(Gaussian g, float3x3 cov3D, Camera cam) {
    float3x3 view_to_screen_jacobian = getJacobian(g, cam);
    float3x3 left = mul(view_to_screen_jacobian, float3x3(cam.viewMatrix));
    float2x2 cov2D = float2x2(mul(left, mul(cov3D, linalg.transpose(left))));
    return cov2D;
}

public float3x3 getCov3D(Gaussian g) {
    float4 rot_quat_norm = normalize(g.rotation);
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
        g.scale[0], 0.0, 0.0,
        0.0, g.scale[1], 0.0,
        0.0, 0.0, g.scale[2]
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

public RectBounds getRectTileBounds(Camera cam, float2 pixel_center, float pixel_radius) {
    return new RectBounds(
        (int) floor(clamp((pixel_center.x - pixel_radius) / cam.getTileDims().x, 0.0, cam.num_tiles.x)), 
        (int) ceil(clamp((pixel_center.x + pixel_radius) / cam.getTileDims().x, 0.0, cam.num_tiles.x)),
        (int) floor(clamp((pixel_center.y - pixel_radius) / cam.getTileDims().y, 0.0, cam.num_tiles.y)), 
        (int) ceil(clamp((pixel_center.y + pixel_radius) / cam.getTileDims().y, 0.0, cam.num_tiles.y))
    );
}


public struct RectBounds {
    public int minX;
    public int maxX;
    public int minY;
    public int maxY;
}
    
public Gaussian loadGaussianFromTensors(
    uint idx,
    DiffTensorView<float> positions,
    DiffTensorView<float> rotations,
    DiffTensorView<float> scales,
    DiffTensorView<float> colors,
) {
    Gaussian g;
    g.position = float3(positions[uint2(idx, 0)], positions[uint2(idx, 1)], positions[uint2(idx, 2)]);
    g.rotation = float4(rotations[uint2(idx, 0)], rotations[uint2(idx, 1)], rotations[uint2(idx, 2)], rotations[uint2(idx, 3)]);
    g.scale = float3(scales[uint2(idx, 0)], scales[uint2(idx, 1)], scales[uint2(idx, 2)]);
    g.color = float3(colors[uint2(idx, 0)], colors[uint2(idx, 1)], colors[uint2(idx, 2)]);
    return g;
}

