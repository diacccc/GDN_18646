#include <cuda_bf16.h>
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

template <int THREADS_PER_ROW, int ROWS_PER_BLOCK>
__global__ void rmsnorm_fwd_bf16_d2048(
    const __nv_bfloat16 *__restrict__ x,     // [R, 2048]
    const __nv_bfloat16 *__restrict__ gamma, // [2048]
    const __nv_bfloat16 *__restrict__ bias,  // [2048] or nullptr
    __nv_bfloat16 *__restrict__ y,           // [R, 2048]
    int R,
    float eps)
{
    constexpr int D = 2048;
    constexpr int WARPS_PER_ROW = THREADS_PER_ROW / 32;
    constexpr int ELEMS_PER_THREAD = D / THREADS_PER_ROW;

    static_assert(THREADS_PER_ROW % 32 == 0, "THREADS_PER_ROW must be a multiple of 32");

    const int row_in_block = threadIdx.y;
    const int tid = threadIdx.x;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
    const bool row_valid = row < R;

    __shared__ float warp_sums[ROWS_PER_BLOCK][WARPS_PER_ROW];
    __shared__ float s_rstd[ROWS_PER_BLOCK];

    float local_sum = 0.0f;
    float x_reg[ELEMS_PER_THREAD];

    if (row_valid)
    {
        const __nv_bfloat16 *x_row = x + (size_t)row * D;

#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i)
        {
            int col = tid + i * THREADS_PER_ROW;
            float xv = __bfloat162float(x_row[col]);
            x_reg[i] = xv;
            local_sum = fmaf(xv, xv, local_sum);
        }
    }

    local_sum = warp_sum(local_sum);

    const int lane = tid & 31;
    const int warp_in_row = tid >> 5;

    if (lane == 0)
    {
        warp_sums[row_in_block][warp_in_row] = local_sum;
    }
    __syncthreads();

    if (warp_in_row == 0)
    {
        float row_sum = (lane < WARPS_PER_ROW) ? warp_sums[row_in_block][lane] : 0.0f;
        row_sum = warp_sum(row_sum);
        if (lane == 0)
        {
            s_rstd[row_in_block] = rsqrtf(row_sum / float(D) + eps);
        }
    }
    __syncthreads();

    if (!row_valid)
    {
        return;
    }

    const float rstd = s_rstd[row_in_block];
    __nv_bfloat16 *y_row = y + (size_t)row * D;

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i)
    {
        int col = tid + i * THREADS_PER_ROW;
        float out = x_reg[i] * rstd * __bfloat162float(gamma[col]);
        if (bias != nullptr)
        {
            out += __bfloat162float(bias[col]);
        }
        y_row[col] = __float2bfloat16_rn(out);
    }
}

static void check_inputs(torch::Tensor x, torch::Tensor gamma)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(x.get_device() == gamma.get_device(), "x and gamma must be on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16, "gamma must be BF16");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 2048, "x must be [R, 2048]");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == 2048, "gamma must be [2048]");
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous(), "inputs must be contiguous");
}

template <int THREADS_PER_ROW, int ROWS_PER_BLOCK>
static void launch_rmsnorm(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *gamma,
    __nv_bfloat16 *y,
    int R,
    float eps)
{
    dim3 block(THREADS_PER_ROW, ROWS_PER_BLOCK);
    dim3 grid((R + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    rmsnorm_fwd_bf16_d2048<THREADS_PER_ROW, ROWS_PER_BLOCK><<<grid, block>>>(
        x,
        gamma,
        nullptr,
        y,
        R,
        eps);
}

static void launch_rmsnorm_dispatch(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *gamma,
    __nv_bfloat16 *y,
    int R,
    float eps,
    int block_threads,
    int rows_per_block)
{
    TORCH_CHECK(rows_per_block >= 1, "rows_per_block must be >= 1");
    TORCH_CHECK(block_threads >= 32, "block_threads must be >= 32");
    TORCH_CHECK(block_threads <= 1024, "block_threads must be <= 1024");
    TORCH_CHECK(block_threads % rows_per_block == 0,
                "block_threads must be divisible by rows_per_block");

    const int threads_per_row = block_threads / rows_per_block;
    TORCH_CHECK(threads_per_row >= 32, "threads_per_row must be >= 32");
    TORCH_CHECK(threads_per_row % 32 == 0,
                "threads_per_row must be a multiple of 32");

    switch (threads_per_row)
    {
    case 32:
        switch (rows_per_block)
        {
        case 1:
            launch_rmsnorm<32, 1>(x, gamma, y, R, eps);
            return;
        case 2:
            launch_rmsnorm<32, 2>(x, gamma, y, R, eps);
            return;
        case 4:
            launch_rmsnorm<32, 4>(x, gamma, y, R, eps);
            return;
        case 8:
            launch_rmsnorm<32, 8>(x, gamma, y, R, eps);
            return;
        }
        break;
    case 64:
        switch (rows_per_block)
        {
        case 1:
            launch_rmsnorm<64, 1>(x, gamma, y, R, eps);
            return;
        case 2:
            launch_rmsnorm<64, 2>(x, gamma, y, R, eps);
            return;
        case 4:
            launch_rmsnorm<64, 4>(x, gamma, y, R, eps);
            return;
        case 8:
            launch_rmsnorm<64, 8>(x, gamma, y, R, eps);
            return;
        }
        break;
    case 128:
        switch (rows_per_block)
        {
        case 1:
            launch_rmsnorm<128, 1>(x, gamma, y, R, eps);
            return;
        case 2:
            launch_rmsnorm<128, 2>(x, gamma, y, R, eps);
            return;
        case 4:
            launch_rmsnorm<128, 4>(x, gamma, y, R, eps);
            return;
        case 8:
            launch_rmsnorm<128, 8>(x, gamma, y, R, eps);
            return;
        }
        break;
    case 256:
        switch (rows_per_block)
        {
        case 1:
            launch_rmsnorm<256, 1>(x, gamma, y, R, eps);
            return;
        case 2:
            launch_rmsnorm<256, 2>(x, gamma, y, R, eps);
            return;
        case 4:
            launch_rmsnorm<256, 4>(x, gamma, y, R, eps);
            return;
        }
        break;
    case 512:
        switch (rows_per_block)
        {
        case 1:
            launch_rmsnorm<512, 1>(x, gamma, y, R, eps);
            return;
        case 2:
            launch_rmsnorm<512, 2>(x, gamma, y, R, eps);
            return;
        }
        break;
    case 1024:
        switch (rows_per_block)
        {
        case 1:
            launch_rmsnorm<1024, 1>(x, gamma, y, R, eps);
            return;
        }
        break;
    }



    TORCH_CHECK(false,
                "Unsupported config: block_threads=", block_threads,
                ", rows_per_block=", rows_per_block,
                ", threads_per_row=", threads_per_row);
}

static torch::Tensor rmsnorm_forward_impl(
    torch::Tensor x,
    torch::Tensor gamma,
    double eps,
    int block_threads,
    int rows_per_block)
{
    check_inputs(x, gamma);

    auto y = torch::empty_like(x);
    const int R = x.size(0);

    launch_rmsnorm_dispatch(
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(gamma.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr<at::BFloat16>()),
        R,
        static_cast<float>(eps),
        block_threads,
        rows_per_block);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "RMSNorm kernel launch failed: ",
                cudaGetErrorString(err));

    return y;
}

torch::Tensor rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    double eps,
    int64_t block_threads,
    int64_t rows_per_block)
{
    return rmsnorm_forward_impl(
        x,
        gamma,
        eps,
        static_cast<int>(block_threads),
        static_cast<int>(rows_per_block));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &rmsnorm_forward, "RMSNorm forward (BF16, D=2048, configurable block_threads / rows_per_block)");
}
