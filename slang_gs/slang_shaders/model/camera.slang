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
    public float2 focalLength;

    // num tiles per dim 
    public int2 num_tiles;

    public float2 getFOV() {
        float x = 2.0 * atan(0.5 * canvasSize.x  / focalLength.x);
        float y = 2.0 * atan(0.5 * canvasSize.y / focalLength.y);
        return float2(x, y);
    }

    public float2 ndcToPixel(float3 ndc) {
        return ((ndc.xy + 1.0) * canvasSize) * 0.5;
    }

    public int2 getTileDims() {
        return (int2) (canvasSize / num_tiles);
    }
}
