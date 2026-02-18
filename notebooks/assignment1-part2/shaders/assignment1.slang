import texture;

public float4 sliceNearestSample(uint2 tid,
                                 uint2 canvasSize,
                                 SharedTexture3DBuffer<float4> volumeTexBuf,
                                 SharedTexture3D volumeTex,
                                 float slice)
{
    float3 uvw;
    uvw.y = (float(tid.x) + 0.5) / float(canvasSize.x);
    uvw.z = (float(tid.y) + 0.5) / float(canvasSize.y);
    uvw.x = slice;
    return volumeTexBuf.pointSample(volumeTex, uvw);
}

public float4 sliceTrilinearSample(uint2 tid,
                                   uint2 canvasSize,
                                   SharedTexture3DBuffer<float4> volumeTexBuf,
                                   SharedTexture3D volumeTex,
                                   float slice)
{
    float3 uvw;
    uvw.y = (float(tid.x) + 0.5) / float(canvasSize.x);
    uvw.z = (float(tid.y) + 0.5) / float(canvasSize.y);
    uvw.x = slice;
    return volumeTexBuf.trilinearSample(volumeTex, uvw);
}
