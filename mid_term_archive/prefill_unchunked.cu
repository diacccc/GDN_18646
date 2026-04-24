#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <stddef.h>
#include <math.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Compile-time constants                                               */
/* ------------------------------------------------------------------ */
#define H   16
#define K   128
#define V   256
#define C   64

/* ------------------------------------------------------------------ */
/* Index helpers                                                        */
/* ------------------------------------------------------------------ */

#define S_IDX(b, h, ki, vi) ((b) * H * K * V + (h) * K * V + (ki) * V + (vi))
#define MASK_IDX(b, t)    ((b) * T + (t))
#define A_IDX(b, t, h) ((b) * H * T + (h) * T + (t))
#define V_IDX(b, t, h, vi) ((b) * H * T * V + (h) * T * V + (t) * V + (vi))
#define K_IDX(b, t, h, ki) ((b) * H * T * K + (h) * T * K + (t) * K + (ki))


/* ------------------------------------------------------------------ */
/* Gate helpers                                                         */
/* ------------------------------------------------------------------ */
__device__ static inline float softplus(float x)
{
    if (x >=  20.0f) return x;
    return log1pf(expf(x));
}

__device__ static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/* ------------------------------------------------------------------ */
/* Kernel (bf16 input/output, fp32 state)                              */
/* ------------------------------------------------------------------ */
__global__ void __launch_bounds__(V, 1)
prefill(
    int B, int T,
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    const float    *A_log,
    const __nv_bfloat16 *a,
    float           dt_bias,
    const __nv_bfloat16 *b_logits,
    const float    *mask,
    const float    *state_in,
    float          *state_out,
    float           scale,
    __nv_bfloat16  *output
)
{
    int b  = blockIdx.x;
    int h  = blockIdx.y;
    int vi = threadIdx.x;

    /* State accumulator always in float32 for numerical stability */
    float S[K];

    if (state_in) {
        for (int ki = 0; ki < K; ki++)
            S[ki] = state_in[S_IDX(b, h, ki, vi)];
    } else {
        for (int ki = 0; ki < K; ki++)
            S[ki] = 0.0f;
    }

    for (int t = 0; t < T; t++) {
        float m    = mask[MASK_IDX(b, t)];
        float x    = __bfloat162float(a[A_IDX(b, t, h)]) + dt_bias;
        float g    = expf(-expf(A_log[h]) * softplus(x));
        float beta = sigmoid(__bfloat162float(b_logits[A_IDX(b, t, h)]));


        float old_v = 0.0f;

        for (int ki = 0; ki < K; ki++){
            /* 1. Gate state in-place */
            S[ki] *= g;

            /* 2. old_v = dot(k, S) */
            old_v = fmaf(__bfloat162float(k[K_IDX(b, t, h, ki)]), S[ki], old_v);
        }

        /* 3. new_v = beta * v + (1 - beta) * old_v */
        float new_v = beta * __bfloat162float(v[V_IDX(b, t, h, vi)])
                    + (1.0f - beta) * old_v;

        float v_diff = m * (new_v - old_v);
        float o = 0.0f;
        for (int ki = 0; ki < K; ki++){
            /* 4. S += k * (new_v - old_v), masked */
            S[ki] = fmaf(__bfloat162float(k[K_IDX(b, t, h, ki)]), v_diff, S[ki]);

            /* 5. output = mask * scale * dot(q, S) */
            o = fmaf(__bfloat162float(q[K_IDX(b, t, h, ki)]), S[ki], o);
        }

        output[V_IDX(b, t, h, vi)] = __float2bfloat16(m * scale * o);
    }

    for (int ki = 0; ki < K; ki++)
        state_out[S_IDX(b, h, ki, vi)] = S[ki];
}

std::tuple<torch::Tensor, torch::Tensor> gdn_prefill(
    torch::Tensor q,         /* [B, H, T, K] */
    torch::Tensor k,         /* [B, H, T, K] */
    torch::Tensor v,         /* [B, H, T, V] */
    torch::Tensor A_log,     /* [H]          */
    torch::Tensor a,         /* [B, H, T]    */
    float         dt_bias,
    torch::Tensor b_logits,  /* [B, H, T]    */
    torch::Tensor mask,      /* [B, T]        */
    torch::Tensor state_in,  /* [B, H, K, V], empty tensor allowed */
    float       scale

) {
    const int B = q.size(0);
    const int T = q.size(2);

    TORCH_CHECK(q.size(1) == H && q.size(3) == K,   "q shape must be [B,16,T,128]");
    TORCH_CHECK(k.size(1) == H && k.size(3) == K,   "k shape must be [B,16,T,128]");
    TORCH_CHECK(v.size(1) == H && v.size(3) == V,   "v shape must be [B,16,T,256]");
    TORCH_CHECK(A_log.size(0) == H,                  "A_log shape must be [16]");
    TORCH_CHECK(a.size(0) == B && a.size(1) == H && a.size(2) == T,
                "a shape must be [B,16,T]");
    TORCH_CHECK(b_logits.sizes() == a.sizes(),        "b_logits shape must match a");
    TORCH_CHECK(mask.size(0) == B && mask.size(1) == T, "mask shape must be [B,T]");

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && a.is_cuda() && b_logits.is_cuda(),
                "q, k, v, a, b_logits must be CUDA tensors");
    TORCH_CHECK(mask.is_cuda() && A_log.is_cuda() && state_in.is_cuda(),
                "mask, A_log, state_in must be CUDA tensors");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                a.is_contiguous() && b_logits.is_contiguous() && mask.is_contiguous() &&
                A_log.is_contiguous() && state_in.is_contiguous(),
                "all tensors must be contiguous");

    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "a must be bfloat16");
    TORCH_CHECK(b_logits.scalar_type() == torch::kBFloat16, "b_logits must be bfloat16");

    TORCH_CHECK(A_log.scalar_type()   == torch::kFloat32, "A_log must be float32");
    TORCH_CHECK(mask.scalar_type()    == torch::kFloat32, "mask must be float32");

    if (scale == 0.0)
        scale = 1.0f / sqrtf((float)K);

    /* ---- Optional state_in ---- */
    TORCH_CHECK(state_in.scalar_type() == torch::kFloat32, "state_in must be float32");
    const float* state_in_ptr = nullptr;
    if (state_in.numel() > 0) {
        TORCH_CHECK(state_in.size(0) == B &&
                    state_in.size(1) == H &&
                    state_in.size(2) == K  &&
                    state_in.size(3) == V,  "state_in shape must be [B,16,128,256]");
        state_in_ptr = state_in.data_ptr<float>();
    }

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto opts_f32  = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());

    torch::Tensor output = torch::zeros({B, H, T, V}, opts_bf16);
    torch::Tensor state_out = torch::zeros({B, H, K, V}, opts_f32);

    dim3 grid(B, H);

    prefill<<<grid, V>>>(
        B, T,
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        A_log.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        dt_bias,
        reinterpret_cast<const __nv_bfloat16*>(b_logits.data_ptr<at::BFloat16>()),
        mask.data_ptr<float>(),
        state_in_ptr,
        state_out.data_ptr<float>(),
        scale,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>())
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "prefill kernel launch failed");
    return {output, state_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefill", &gdn_prefill, "Prefill forward (bf16 I/O, fp32 state)");
    m.attr("H_DIM") = H;
    m.attr("K_DIM") = K;
    m.attr("V_DIM") = V;
    m.attr("C_DIM") = C;
}