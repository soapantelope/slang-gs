import math;
import model;
import primitive;

/**
 * Fill a solid red color.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @return The color red as a float4 vector.
 */
public float4 fillRed(uint2 tid)
{
    // Return the color as a float4 vector (red, green, blue, alpha).
    // In this example, we set alpha to 1 to indicate a fully opaque pixel.
    // We'll cover alpha, transparency later in the class.
    return float4(1.0, 0.0, 0.0, 1.0);
}

/**
 * Fill a uv gradient pattern.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @return The color corresponding to the uv(xy) coordinate of the pixel.
 */
public float4 fillGradient(uint2 tid, uint2 canvasSize)
{
    // TODO: Student implementation starts here.

    return float4((float) tid[1] / canvasSize[1], (float) tid[0] / canvasSize[0], 0.0, 1.0);

    // TODO: Student implementation ends here.
}

/**
 * Fill a checkerboard pattern.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @param numSquares The number of squares along each axis.
 * @return The color white or black depending on the checkerboard pattern.
 */
public float4 fillCheckerboard(uint2 tid, uint2 canvasSize, uint numSquares)
{
    // TODO: Student implementation starts here.
    uint x_num = tid[0] * numSquares / canvasSize[0];
    uint y_num = tid[1] * numSquares / canvasSize[1];
    if (x_num % 2 == y_num % 2) {
         return float4(1.0, 1.0, 1.0, 1.0);
    }
    return float4(0.0, 0.0, 0.0, 1.0);
   

    // TODO: Student implementation ends here.
}

/**
 * Rasterize a single triangle.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @param v0 The first vertex of the triangle in normalized coordinates.
 * @param v1 The second vertex of the triangle in normalized coordinates.
 * @param v2 The third vertex of the triangle in normalized coordinates.
 * @return The color white if the pixel is inside the triangle, otherwise black.
 */
public float4 triangleRasterizer(uint2 tid, uint2 canvasSize, float2 v0, float2 v1, float2 v2)
{
    // TODO: Student implementation starts here.

    float normal_dots[3];
    float2 vertices[] = {v0, v1, v2};
    float2 x = float2((float) tid[0] / canvasSize[0], (float) tid[1] / canvasSize[1]);
    
    for (int i = 0; i < 3; i++) {
        float2 va = vertices[i];
        float2 vb = vertices[(i + 1) % vertices.getCount()];
        float2 normal = float2(vb[1] - va[1], va[0] - vb[0]);
        float2 x_vec = x - va;
        normal_dots[i] = dot(normal, x_vec);
    }

    if (normal_dots[0] < 0 && normal_dots[1] < 0 && normal_dots[2] < 0) {
        return float4(1.0, 1.0, 1.0, 1.0);
    } else {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    // TODO: Student implementation ends here.
}

/**
 * Test ray-triangle intersection for a single triangle.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @param triangles A buffer containing triangles to test against. Only the first triangle is used in this function.
 * @return The color of the triangle if hit, otherwise black.
 */
public float4 rayTriangleTest(uint2 tid, uint2 canvasSize, StructuredBuffer<Triangle> triangles)
{
    // Get the first triangle in the buffer.
    Triangle triangle = triangles[0];
    // We generate rays with origin from (0, 0, 0) to (1, 1, 0) and direction towards the negative z axis (0, 0, -1).
    float2 uv = float2(tid) / float2(canvasSize);
    Ray ray = Ray(float3(uv, 0.0), float3(0.0, 0.0, -1.0), float2(0.0, float.maxValue));
    // How can we make sure the rays are set up correctly? Just return the value as color.
    // return float4(ray.origin.xy, 0.0, 1.0);
    Optional<RayHitResult> hitResult = triangle.hit(ray);
    if (hitResult.hasValue)
    {
        return float4(triangle.averageColor(), 1.0);
    }
    else
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
}

public int tryGetCount(StructuredBuffer<int> buff) {
    return buff.getCount();
}
/**
 * Test ray-triangle intersection for multiple triangles.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @param triangles A buffer containing triangles to test against.
 * @param triangleCount The number of triangles in the buffer.
 * @return The color of the closest triangle if hit, otherwise black.
 */
public float4 rayTrianglesTest(uint2 tid, uint2 canvasSize, StructuredBuffer<Triangle> triangles, uint triangleCount)
{
    // We generate rays with origin from (0, 0, 0) to (1, 1, 0) and direction towards the negative z axis (0, 0, -1).
    float2 uv = float2(tid) / float2(canvasSize);
    Ray ray = Ray(float3(uv, 0.0), float3(0.0, 0.0, -1.0), float2(0.0, float.maxValue));

    // TODO: Student implementation starts here.

    Optional<Triangle> triangle_to_render;
    float min_t = float.maxValue;
    for (uint i = 0; i < triangleCount; i++) {
        Triangle triangle = triangles[i];
        Optional<RayHitResult> hitResult = triangle.hit(ray);
        if (hitResult.hasValue) {
            if (!triangle_to_render.hasValue || hitResult.value.t < min_t) {
                min_t = hitResult.value.t;
                triangle_to_render = triangle;
            }
        }
    }

    if (triangle_to_render.hasValue) {
        return float4(triangle_to_render.value.averageColor(), 1.0);
    }

    return float4(0.0, 0.0, 0.0, 1.0);

    // TODO: Student implementation starts here.
}
