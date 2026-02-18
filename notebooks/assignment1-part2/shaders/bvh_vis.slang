import math;
import model;
import primitive;

public float4 visualizeBVH(uint2 tid, Camera camera, StructuredBuffer<BVHNode> nodes, uint numNodes, uint depth, float alpha)
{
    // Calculate screen space uv coordinate.
    float2 uv = (float2(tid) + 0.5) / float2(camera.canvasSize);
    Ray camRay = camera.generateRay(uv);
    const static float3 BOX_COLOR = float3(1.0, 0.0, 0.0);
    const static float MIN_ATTENUATION = 0.01;
    const static float STEP_EPS = 1e-4;
    float3 accumColor = float3(0.0, 0.0, 0.0);
    float accumAlpha = 0.0;
    float attenuation = 1.0;
    float currentT = 0.0;
    [MaxIters(4096)]
    for (uint iter = 0; iter < 4096 && attenuation > MIN_ATTENUATION; ++iter)
    {
        float closestEntry = float.maxValue;
        uint closestIndex = uint.maxValue;
        for (uint i = 0; i < numNodes; ++i)
        {
            uint maskWord = i >> 5;
            uint maskBit = 1u << (i & 31);
            BVHNode node = nodes[i];
            if (node.depth != depth)
            {
                continue;
            }
            float2 tHit = node.bound.hit(camRay);
            if (tHit.x > tHit.y)
            {
                continue;
            }
            if (tHit.x < currentT)
            {
                continue;
            }
            if (tHit.x < closestEntry)
            {
                closestEntry = tHit.x;
                closestIndex = i;
            }
        }
        if (closestIndex == uint.maxValue)
        {
            break;
        }
        BVHNode hitNode = nodes[closestIndex];
        float sampleAlpha = alpha;
        accumColor += attenuation * sampleAlpha * BOX_COLOR;
        accumAlpha += attenuation * sampleAlpha;
        attenuation *= (1.0 - sampleAlpha);
        currentT = closestEntry + STEP_EPS;
    }
    return float4(accumColor, accumAlpha);
}
