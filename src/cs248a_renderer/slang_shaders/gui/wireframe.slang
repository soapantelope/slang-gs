struct Uniform
{
    float4x4 modelMatrix;
    float4x4 viewMatrix;
    float4x4 projMatrix;
    float4 color;
};

ConstantBuffer<Uniform> uniforms;

struct VertexInput
{
    float3 position : POSITION;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput vIn)
{
    VertexOutput vOut;
    float4 worldPos = mul(uniforms.modelMatrix, float4(vIn.position, 1.0));
    float4 viewPos = mul(uniforms.viewMatrix, worldPos);
    vOut.position = mul(uniforms.projMatrix, viewPos);
    vOut.color = uniforms.color;
    return vOut;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput pIn: VertexOutput)
    : SV_Target
{
    return pIn.color;
}
