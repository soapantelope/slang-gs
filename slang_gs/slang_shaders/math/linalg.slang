implementing "../math.slang";

namespace linalg
{

[Differentiable]
public matrix<T, M, N> transpose<T, int M, int N>(matrix<T, N, M> x)
{
    matrix<T, M, N> res;
    [ForceUnroll]
    for (int i = 0; i < M; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < N; j++)
        {
            res[i][j] = x[j][i];
        }
    }
    return res;
}

}
