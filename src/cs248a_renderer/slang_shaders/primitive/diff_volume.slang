implementing "../primitive.slang";
import math;
import texture;

public struct DiffVolume
{
    // The bounding box of the volume in local space.
    public BoundingBox bound;
    // The 3D texture storing the volume data (rgba).
    public DiffTexture3D<float, 4> tex;
    // The differential of the 3D texture.
    public DiffTexture3D<float, 4>.Differential dTex;
    // The model matrix (object space -> world space) and its inverse.
    public float4x4 modelMatrix;
    public float4x4 invModelMatrix;
}
