implementing "../math.slang";

[Differentiable]
public vector<float, N> sigmoid<let N : int>(vector<float, N> x)
{
    vector<float, N> one = vector<float, N>(1.0);
    return one / (one + exp(-x));
}
