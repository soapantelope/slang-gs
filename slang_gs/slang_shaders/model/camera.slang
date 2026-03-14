implementing "../model.slang";
import math;

public struct Camera
{
    // The inverse view matrix of the camera (matrix transform from view space to world space).
    public float4x4 invViewMatrix;
    // world to camera view space
    public float4x4 viewMatrix;
    public float4x4 projMatrix;
    // The canvas size of the target image in pixels. W, H
    public uint2 canvasSize;
    // The focal length of the camera in pixels (distance between the pinhole and the image plane).
    public float focalLength;

    // num tiles per dim 
    public int2 num_tiles;

    /**
     * Generate a camera ray given a normalized uv coordinate on canvas.
     * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
     * @return The generated Ray. Before you start working the transform section, generate ray as if the camera is at origin looking towards negative z direction.
     * After you finish the transform section, you should transform the ray from view space to world space using the inverse view matrix.
     */
    public Ray generateRay(float2 uv)
    {
        // TODO: Student implementation starts here.
        float2 new_uv = (uv - 0.5) * canvasSize;
        float3 image_point = float3(new_uv, -focalLength);
        float3 origin = float3(0.0, 0.0, 0.0);
        Ray ray = Ray(origin, normalize(image_point - origin));
        return ray.transform(invViewMatrix);
        // TODO: Student implementation ends here.
    }

    public float2 getFOV() {
        float x = 2.0 * atan(0.5 * canvasSize.x  / focalLength);
        float y = 2.0 * atan(0.5 * canvasSize.y / focalLength);
        return float2(x, y);
    }

    public float2 ndcToPixel(float3 ndc) {
        return ((ndc.xy + 1.0) * canvasSize) * 0.5;
    }

    public int2 getTileDims() {
        return (int2) (canvasSize / num_tiles);
    }
}
