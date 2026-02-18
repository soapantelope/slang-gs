implementing "../texture.slang";

public struct DiffTexture2D_Differential<T, let N : int> : IDifferentiablePtrType
    where T : IAtomicable
{
    typealias Differential = DiffTexture2D_Differential<T, N>;
    public RWStructuredBuffer<Atomic<T>[N]> dTex;

    public __init(RWStructuredBuffer<Atomic<T>[N]> dTex)
    {
        this.dTex = dTex;
    }
}

public struct DiffTexture2D<T, let N : int> : IDifferentiablePtrType
    where T : IAtomicable
{
    typealias Differential = DiffTexture2D_Differential<T, N>;
    public StructuredBuffer<vector<T, N>> tex;
    public uint2 size;

    public __init(StructuredBuffer<vector<T, N>> tex, uint2 size)
    {
        this.tex = tex;
        this.size = size;
    }
}

/// Sample a 2D texture using point filtering.
[BackwardDerivative(samplePoint_bwd)]
public vector<T, N> samplePoint<T, let N : int>(DiffTexture2D<T, N> diffTex, no_diff float2 uv)
    where T : IArithmeticAtomicable, IDifferentiable
{
    uint2 texelCoord = uint2(uv * float2(diffTex.size));
    uint index = texelCoord.y * diffTex.size.x + texelCoord.x;
    return diffTex.tex[index];
}

/// Backward pass for point texture sampling.
public void samplePoint_bwd<T, let N : int>(DifferentialPtrPair<DiffTexture2D<T, N>> diffTexPair, float2 uv, vector<T, N> dOut)
    where T : IArithmeticAtomicable
{
    uint2 texelCoord = uint2(uv * float2(diffTexPair.p.size));
    uint index = texelCoord.y * diffTexPair.p.size.x + texelCoord.x;
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
    {
        diffTexPair.d.dTex[index][i].add(dOut[i]);
    }
}

// Sample a 2D texture using bilinear filtering.
[BackwardDerivative(sampleBilinear_bwd)]
public vector<T, N> sampleBilinear<T, let N : int>(DiffTexture2D<T, N> diffTex, no_diff float2 uv)
    where T : IArithmeticAtomicable, IDifferentiable, IFloat
{
    // Convert UV to continuous texel space
    float2 st = uv * float2(diffTex.size);

    // Base integer coordinates and fractional parts
    int x0 = int(floor(st.x));
    int y0 = int(floor(st.y));
    float fx = st.x - float(x0);
    float fy = st.y - float(y0);

    // Neighbor coordinates
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp to texture bounds
    int maxX = int(diffTex.size.x) - 1;
    int maxY = int(diffTex.size.y) - 1;
    x0 = clamp(x0, 0, maxX);
    y0 = clamp(y0, 0, maxY);
    x1 = clamp(x1, 0, maxX);
    y1 = clamp(y1, 0, maxY);

    // Compute linear indices
    uint idx00 = uint(y0) * diffTex.size.x + uint(x0);
    uint idx10 = uint(y0) * diffTex.size.x + uint(x1);
    uint idx01 = uint(y1) * diffTex.size.x + uint(x0);
    uint idx11 = uint(y1) * diffTex.size.x + uint(x1);

    // Fetch texels
    vector<T, N> c00 = diffTex.tex[idx00];
    vector<T, N> c10 = diffTex.tex[idx10];
    vector<T, N> c01 = diffTex.tex[idx01];
    vector<T, N> c11 = diffTex.tex[idx11];

    // Bilinear blend per component using float math, then cast back to T
    float w00 = (1.0 - fx) * (1.0 - fy);
    float w10 = fx * (1.0 - fy);
    float w01 = (1.0 - fx) * fy;
    float w11 = fx * fy;

    vector<T, N> outV;
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
    {
        T v00 = c00[i];
        T v10 = c10[i];
        T v01 = c01[i];
        T v11 = c11[i];
        T vv = (T)w00 * v00 + (T)w10 * v10 + (T)w01 * v01 + (T)w11 * v11;
        outV[i] = vv;
    }
    return outV;
}

// Backward pass for bilinear texture sampling.
public void sampleBilinear_bwd<T, let N : int>(DifferentialPtrPair<DiffTexture2D<T, N>> diffTexPair, float2 uv, vector<T, N> dOut)
    where T : IArithmeticAtomicable, IFloat
{
    // TODO: Student implementation starts here.

    // Convert UV to continuous texel space
    float2 st = uv * float2(diffTexPair.p.size);

    // Base integer coordinates and fractional parts
    int x0 = int(floor(st.x));
    int y0 = int(floor(st.y));
    float fx = st.x - float(x0);
    float fy = st.y - float(y0);

    // Neighbor coordinates
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp to texture bounds
    int maxX = int(diffTexPair.p.size.x) - 1;
    int maxY = int(diffTexPair.p.size.y) - 1;
    x0 = clamp(x0, 0, maxX);
    y0 = clamp(y0, 0, maxY);
    x1 = clamp(x1, 0, maxX);
    y1 = clamp(y1, 0, maxY);

    // Compute linear indices
    uint idx00 = uint(y0) * diffTexPair.p.size.x + uint(x0);
    uint idx10 = uint(y0) * diffTexPair.p.size.x + uint(x1);
    uint idx01 = uint(y1) * diffTexPair.p.size.x + uint(x0);
    uint idx11 = uint(y1) * diffTexPair.p.size.x + uint(x1);

    // weights
    float w00 = (1.0 - fx) * (1.0 - fy);
    float w10 = fx * (1.0 - fy);
    float w01 = (1.0 - fx) * fy;
    float w11 = fx * fy;

    [ForceUnroll]
    for (int i = 0; i < N; ++i)
    {
        T d = dOut[i];
        diffTexPair.d.dTex[idx00][i].add((T)w00 * d);
        diffTexPair.d.dTex[idx10][i].add((T)w10 * d);
        diffTexPair.d.dTex[idx01][i].add((T)w01 * d);
        diffTexPair.d.dTex[idx11][i].add((T)w11 * d);
    }
    // TODO: Student implementation ends here.
}

public struct DiffTexture3D_Differential<T, let N : int> : IDifferentiablePtrType
    where T : IAtomicable
{
    typealias Differential = DiffTexture3D_Differential<T, N>;
    public RWStructuredBuffer<Atomic<T>[N]> dTex;

    public __init(RWStructuredBuffer<Atomic<T>[N]> dTex)
    {
        this.dTex = dTex;
    }
}

public struct DiffTexture3D<T, let N : int> : IDifferentiablePtrType
    where T : IAtomicable
{
    typealias Differential = DiffTexture3D_Differential<T, N>;
    public StructuredBuffer<vector<T, N>> tex;
    public uint3 size;

    public __init(StructuredBuffer<vector<T, N>> tex, uint3 size)
    {
        this.tex = tex;
        this.size = size;
    }
}

/// Sample a 3D texture using point filtering.
[BackwardDerivative(samplePoint3D_bwd)]
public vector<T, N> samplePoint3D<T, let N : int>(DiffTexture3D<T, N> diffTex, no_diff float3 uv)
    where T : IArithmeticAtomicable, IDifferentiable
{
    uint3 texelCoord = uint3(uv * float3(diffTex.size));
    uint index = (texelCoord.z * diffTex.size.y + texelCoord.y) * diffTex.size.x + texelCoord.x;
    return diffTex.tex[index];
}

/// Backward pass for point texture sampling (3D).
public void samplePoint3D_bwd<T, let N : int>(DifferentialPtrPair<DiffTexture3D<T, N>> diffTexPair, float3 uv, vector<T, N> dOut)
    where T : IArithmeticAtomicable
{
    uint3 texelCoord = uint3(uv * float3(diffTexPair.p.size));
    uint index = (texelCoord.z * diffTexPair.p.size.y + texelCoord.y) * diffTexPair.p.size.x + texelCoord.x;
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
    {
        diffTexPair.d.dTex[index][i].add(dOut[i]);
    }
}

// Sample a 3D texture using trilinear filtering.
[BackwardDerivative(sampleTrilinear_bwd)]
public vector<T, N> sampleTrilinear<T, let N : int>(DiffTexture3D<T, N> diffTex, no_diff float3 uv)
    where T : IArithmeticAtomicable, IDifferentiable, IFloat
{
    // Convert UVW to continuous texel space
    float3 stp = uv * float3(diffTex.size);

    // Base integer coordinates and fractional parts
    int x0 = int(floor(stp.x));
    int y0 = int(floor(stp.y));
    int z0 = int(floor(stp.z));
    float fx = stp.x - float(x0);
    float fy = stp.y - float(y0);
    float fz = stp.z - float(z0);

    // Neighbor coordinates
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    // Clamp to texture bounds
    int maxX = int(diffTex.size.x) - 1;
    int maxY = int(diffTex.size.y) - 1;
    int maxZ = int(diffTex.size.z) - 1;
    x0 = clamp(x0, 0, maxX);
    y0 = clamp(y0, 0, maxY);
    z0 = clamp(z0, 0, maxZ);
    x1 = clamp(x1, 0, maxX);
    y1 = clamp(y1, 0, maxY);
    z1 = clamp(z1, 0, maxZ);

    // Compute linear indices
    uint sx = diffTex.size.x;
    uint sy = diffTex.size.y;
    uint idx000 = (uint(z0) * sy + uint(y0)) * sx + uint(x0);
    uint idx100 = (uint(z0) * sy + uint(y0)) * sx + uint(x1);
    uint idx010 = (uint(z0) * sy + uint(y1)) * sx + uint(x0);
    uint idx110 = (uint(z0) * sy + uint(y1)) * sx + uint(x1);
    uint idx001 = (uint(z1) * sy + uint(y0)) * sx + uint(x0);
    uint idx101 = (uint(z1) * sy + uint(y0)) * sx + uint(x1);
    uint idx011 = (uint(z1) * sy + uint(y1)) * sx + uint(x0);
    uint idx111 = (uint(z1) * sy + uint(y1)) * sx + uint(x1);

    // Fetch texels
    vector<T, N> c000 = diffTex.tex[idx000];
    vector<T, N> c100 = diffTex.tex[idx100];
    vector<T, N> c010 = diffTex.tex[idx010];
    vector<T, N> c110 = diffTex.tex[idx110];
    vector<T, N> c001 = diffTex.tex[idx001];
    vector<T, N> c101 = diffTex.tex[idx101];
    vector<T, N> c011 = diffTex.tex[idx011];
    vector<T, N> c111 = diffTex.tex[idx111];

    // Trilinear weights
    float wx0 = (1.0 - fx);
    float wy0 = (1.0 - fy);
    float wz0 = (1.0 - fz);
    float wx1 = fx;
    float wy1 = fy;
    float wz1 = fz;

    float w000 = wx0 * wy0 * wz0;
    float w100 = wx1 * wy0 * wz0;
    float w010 = wx0 * wy1 * wz0;
    float w110 = wx1 * wy1 * wz0;
    float w001 = wx0 * wy0 * wz1;
    float w101 = wx1 * wy0 * wz1;
    float w011 = wx0 * wy1 * wz1;
    float w111 = wx1 * wy1 * wz1;

    vector<T, N> outV;
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
    {
        T v000 = c000[i];
        T v100 = c100[i];
        T v010 = c010[i];
        T v110 = c110[i];
        T v001 = c001[i];
        T v101 = c101[i];
        T v011 = c011[i];
        T v111 = c111[i];
        T vv = (T)w000 * v000 + (T)w100 * v100 + (T)w010 * v010 + (T)w110 * v110 + (T)w001 * v001 + (T)w101 * v101 + (T)w011 * v011 + (T)w111 * v111;
        outV[i] = vv;
    }
    return outV;
}

// Backward pass for trilinear texture sampling (3D).
public void sampleTrilinear_bwd<T, let N : int>(DifferentialPtrPair<DiffTexture3D<T, N>> diffTexPair, float3 uv, vector<T, N> dOut)
    where T : IArithmeticAtomicable, IFloat
{
    // TODO: Student implementation starts here.
    // Convert UVW to continuous texel space
    float3 stp = uv * float3(diffTexPair.p.size);

    // Base integer coordinates and fractional parts
    int x0 = int(floor(stp.x));
    int y0 = int(floor(stp.y));
    int z0 = int(floor(stp.z));
    float fx = stp.x - float(x0);
    float fy = stp.y - float(y0);
    float fz = stp.z - float(z0);

    // Neighbor coordinates
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    // Clamp to texture bounds
    int maxX = int(diffTexPair.p.size.x) - 1;
    int maxY = int(diffTexPair.p.size.y) - 1;
    int maxZ = int(diffTexPair.p.size.z) - 1;
    x0 = clamp(x0, 0, maxX);
    y0 = clamp(y0, 0, maxY);
    z0 = clamp(z0, 0, maxZ);
    x1 = clamp(x1, 0, maxX);
    y1 = clamp(y1, 0, maxY);
    z1 = clamp(z1, 0, maxZ);

    // Compute linear indices
    uint sx = diffTexPair.p.size.x;
    uint sy = diffTexPair.p.size.y;
    uint idx000 = (uint(z0) * sy + uint(y0)) * sx + uint(x0);
    uint idx100 = (uint(z0) * sy + uint(y0)) * sx + uint(x1);
    uint idx010 = (uint(z0) * sy + uint(y1)) * sx + uint(x0);
    uint idx110 = (uint(z0) * sy + uint(y1)) * sx + uint(x1);
    uint idx001 = (uint(z1) * sy + uint(y0)) * sx + uint(x0);
    uint idx101 = (uint(z1) * sy + uint(y0)) * sx + uint(x1);
    uint idx011 = (uint(z1) * sy + uint(y1)) * sx + uint(x0);
    uint idx111 = (uint(z1) * sy + uint(y1)) * sx + uint(x1);

    // Trilinear weights
    float wx0 = (1.0 - fx);
    float wy0 = (1.0 - fy);
    float wz0 = (1.0 - fz);
    float wx1 = fx;
    float wy1 = fy;
    float wz1 = fz;

    float w000 = wx0 * wy0 * wz0;
    float w100 = wx1 * wy0 * wz0;
    float w010 = wx0 * wy1 * wz0;
    float w110 = wx1 * wy1 * wz0;
    float w001 = wx0 * wy0 * wz1;
    float w101 = wx1 * wy0 * wz1;
    float w011 = wx0 * wy1 * wz1;
    float w111 = wx1 * wy1 * wz1;

    vector<T, N> outV;
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
    {
        T d = dOut[i];
        diffTexPair.d.dTex[idx000][i].add((T)w000 * d);
        diffTexPair.d.dTex[idx100][i].add((T)w100 * d);
        diffTexPair.d.dTex[idx010][i].add((T)w010 * d);
        diffTexPair.d.dTex[idx110][i].add((T)w110 * d);
        diffTexPair.d.dTex[idx001][i].add((T)w001 * d);
        diffTexPair.d.dTex[idx101][i].add((T)w101 * d);
        diffTexPair.d.dTex[idx011][i].add((T)w011 * d);
        diffTexPair.d.dTex[idx111][i].add((T)w111 * d);
    }
    // TODO: Student implementation ends here.
}
