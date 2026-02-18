__exported import NeuralNetworks;
import slangpy;
import texture;


static const float PI_F = 3.14159265358979323846;

/**
 * The neural texture model. It takes in a 22-dimensional input and outputs a 4-dimensional color.
 */
public typealias INeuralTexture = IModel<float[22], float4>;


/**
 * The texture statistics. It contains the mean and standard deviation of the texture.
 */
public struct TextureStats<T>
{
    public T mean;
    public T std;
}

public struct Uniforms
{
    public Texture2D<float4> tex;
    public SamplerState sampler;
    public bool usePositionalEncoding;
    public TextureStats<float4> stats;
    public uint2 canvasSize;
}

/**
 * Computes the positional encoding for a given uv coordinate.
 * @param uv The uv coordinate to compute the positional encoding for.
 * @return The sinusoidal encoding with 5 levels of frequency.
 */
float[20] getPositionalEncoding(no_diff float2 uv)
{
    const uint NUM_LEVELS = 5;
    float[NUM_LEVELS * 4] pos_enc;

    // TODO: Student implementation starts here.
    for(int i = 0; i < NUM_LEVELS * 2; ++i)
    {
        if (i % 2 == 0) {
            pos_enc[i] = sin(pow(2.0, i / 2.0) * PI_F * uv[0]);
            pos_enc[i + 10] = sin(pow(2.0, i / 2.0) * PI_F * uv[1]);
        } else {
            pos_enc[i] = cos(pow(2.0, (i - 1) / 2.0) * PI_F * uv[0]);
            pos_enc[i + 10] = cos(pow(2.0, (i - 1) / 2.0) * PI_F * uv[1]);
        }
    }
    // TODO: Student implementation starts here.
    return pos_enc;
}

[Differentiable]
float4 sample<NeuralTexture : INeuralTexture>(no_diff float2 uv, NeuralTexture mlp, Uniforms uniforms)
{
    // Normalize UV: from 0 to 1 range to -1 to 1 range
    float2 normalized_uv = 2 * uv - 1;
    float[20] pos_enc;
    if (uniforms.usePositionalEncoding)
    {
        pos_enc = getPositionalEncoding(normalized_uv);
    }
    else
    {
        for(int i = 0; i < 20; ++i)
        {
            pos_enc[i] = 0.0;
        }
    }
    float[22] input;
    for (int i = 0; i < 22; ++i)
    {
        if (i == 20)
        {
            input[i] = normalized_uv.x;
        }
        else if (i == 21)
        {
            input[i] = normalized_uv.y;
        }
        else
        {
            input[i] = pos_enc[i];
        }
    }
    return mlp.forward(input) * uniforms.stats.std + uniforms.stats.mean;
}

public float4 render<NeuralTexture : INeuralTexture>(uint2 tid, NeuralTexture mlp, Uniforms uniforms)
{
    float2 uv = (float2(tid) + 0.5) / float2(uniforms.canvasSize);
    return sample<NeuralTexture>(uv, mlp, uniforms);
}

public float4 renderFloat<NeuralTexture : INeuralTexture>(float2 tid, NeuralTexture mlp, Uniforms uniforms)
{
    float2 uv = (tid + 0.5) / float2(uniforms.canvasSize);
    return sample<NeuralTexture>(uv, mlp, uniforms);
}

public void renderBackward<NeuralTexture : INeuralTexture>(uint2 tid, NeuralTexture mlp, Uniforms uniforms, float4 dOutput)
{
    float2 uv = (float2(tid) + 0.5) / float2(uniforms.canvasSize);
    bwd_diff(sample<NeuralTexture>)(uv, mlp, uniforms, dOutput);
}

public void renderBackwardFloat<NeuralTexture : INeuralTexture>(float2 tid, NeuralTexture mlp, Uniforms uniforms, float4 dOutput)
{
    float2 uv = (tid + 0.5) / float2(uniforms.canvasSize);
    bwd_diff(sample<NeuralTexture>)(uv, mlp, uniforms, dOutput);
}

public float4 sample_texture(float2 uv, Uniforms uniforms) {
    return uniforms.tex.SampleLevel(uniforms.sampler, uv, 0.0);
}
