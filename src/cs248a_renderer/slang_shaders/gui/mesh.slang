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
    float3 normal : NORMAL;
};

struct VertexOutput
{
    float3 viewPos : TEXCOORD0;
    float3 viewNormal : TEXCOORD1;
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput vIn)
{
    VertexOutput vOut;
    // Transform position to clip space
    float4 worldPos = mul(uniforms.modelMatrix, float4(vIn.position, 1.0));
    float4 viewPos = mul(uniforms.viewMatrix, worldPos);
    vOut.viewPos = viewPos.xyz;
    vOut.position = mul(uniforms.projMatrix, viewPos);
    // Transform normal to view space.
    float4 normalWorld = mul(uniforms.modelMatrix, float4(vIn.normal, 0.0));
    float4 normalView = mul(uniforms.viewMatrix, normalWorld);
    normalView = normalize(normalView);
    vOut.viewNormal = normalView.xyz;

    vOut.color = uniforms.color;
    return vOut;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput pIn: VertexOutput)
    : SV_Target
{
    // Simple shading.
    float NdotL = max(dot(normalize(pIn.viewNormal), normalize(float3(0.0, 0.0, 1.0))), 0.0);
    return float4(float3(NdotL), 1.0);
}
