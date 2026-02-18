import texture;

public float4 sampling_fwd(
    uint2 tid,
    uint2 canvasSize,
    StructuredBuffer<float4> sampleTex,
    uint2 texSize)
{
    // TODO: Student implementation starts here.
    float2 uv = (float2) tid / canvasSize;
    DiffTexture2D<float, 4> diffTex = DiffTexture2D<float, 4>(sampleTex, texSize);
    return sampleBilinear<float, 4>(diffTex, uv);
    // TODO: Student implementation ends here.
}

public void sampling_bwd(
    uint2 tid,
    uint2 canvasSize,
    StructuredBuffer<float4> sampleTex,
    RWStructuredBuffer<Atomic<float>[4]> sampleTexGrad,
    uint2 texSize,
    float4 dOut)
{
    // TODO: Student implementation starts here.
    float2 uv = (float2) tid / canvasSize;
    DiffTexture2D<float, 4> diffTex = DiffTexture2D<float, 4>(sampleTex, texSize);
    DiffTexture2D_Differential<float, 4> diffTexDiff = DiffTexture2D_Differential<float, 4>(sampleTexGrad);

    DifferentialPtrPair<DiffTexture2D<float, 4>> diffPtrPair = DifferentialPtrPair<DiffTexture2D<float, 4>>(diffTex, diffTexDiff);
    bwd_diff(sampleBilinear)(diffPtrPair, uv, dOut);
    // TODO: Student implementation ends here.
}
