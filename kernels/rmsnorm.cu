#include <cuda_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ float warp_sum(float v)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template <int BLOCK_SIZE = 256>
__global__ void rmsnorm_fwd_fp32_d2048(
    const float *__restrict__ x,     // [R, 2048]
    const float *__restrict__ gamma, // [2048]
    const float *__restrict__ bias,  // [2048] or nullptr
    float *__restrict__ y,           // [R, 2048]
    int R,
    float eps)
{
    constexpr int D = 2048;
    constexpr int ELEMS_PER_THREAD = D / BLOCK_SIZE; // 8
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;       // 8

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= R)
        return;

    __shared__ float warp_sums[NUM_WARPS];
    __shared__ float s_rstd;

    const float *x_row = x + (size_t)row * D;
    float *y_row = y + (size_t)row * D;

    float local_sum = 0.0f;
    float x_reg[ELEMS_PER_THREAD];
    int col_reg[ELEMS_PER_THREAD];

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i)
    {
        int col = tid + i * BLOCK_SIZE;
        float xv = x_row[col];
        x_reg[i] = xv;
        col_reg[i] = col;
        local_sum = fmaf(xv, xv, local_sum);
    }

    local_sum = warp_sum(local_sum);

    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0)
    {
        warp_sums[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0)
    {
        float block_sum = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0)
        {
            s_rstd = rsqrtf(block_sum / float(D) + eps);
        }
    }
    __syncthreads();

    float rstd = s_rstd;

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i)
    {
        int col = col_reg[i];
        float out = x_reg[i] * rstd * gamma[col];
        if (bias != nullptr)
        {
            out += bias[col];
        }
        y_row[col] = out;
    }
}

torch::Tensor rmsnorm_forward(torch::Tensor x,
                              torch::Tensor gamma,
                              double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be FP32");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32, "gamma must be FP32");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 2048, "x must be [R, 2048]");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == 2048, "gamma must be [2048]");
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous(), "inputs must be contiguous");

    auto y = torch::empty_like(x);

    int R = x.size(0);
    dim3 grid(R);
    dim3 block(256);

    rmsnorm_fwd_fp32_d2048<256><<<grid, block>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        nullptr,
        y.data_ptr<float>(),
        R,
        static_cast<float>(eps));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &rmsnorm_forward, "RMSNorm forward (FP32, D=2048)");
}
