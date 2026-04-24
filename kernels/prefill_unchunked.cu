#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stddef.h>
#include <math.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Compile-time constants                                               */
/* ------------------------------------------------------------------ */
#define H   16
#define K   128
#define V   256

/* ------------------------------------------------------------------ */
/* Index helpers                                                        */
/* ------------------------------------------------------------------ */
#define SI_IDX(b, h, vi, ki) ((b) * H * V * K + (h) * V * K + (vi) * K + (ki))
#define MASK_IDX(b, t)    ((b) * T + (t))
#define A_IDX(b, t, h) ((b) * T * H + (t) * H + (h))
#define V_IDX(b, t, h, vi) ((b) * T * H * V + (t) * H * V + (h) * V + (vi))
#define K_IDX(b, t, h, ki) ((b) * T * H * K + (t) * H * K + (h) * K + (ki))

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
/* Kernel                                                               */
/* ------------------------------------------------------------------ */
__global__ void prefill(
    int B, int T,
    const float *q,
    const float *k,
    const float *v,
    const float *A_log,
    const float *a,
    const float *dt_bias,
    const float *b_logits,
    const float *mask,
    const float *state_in,
    float        scale,
    float       *output
)
{
    int b  = blockIdx.x;
    int h  = blockIdx.y;
    int vi = threadIdx.x;

    float S[K];

    if (state_in) {
        for (int ki = 0; ki < K; ki++)
            S[ki] = state_in[SI_IDX(b, h, vi, ki)];
    } else {
        for (int ki = 0; ki < K; ki++)
            S[ki] = 0.0f;
    }

    for (int t = 0; t < T; t++) {
        float m    = mask[MASK_IDX(b, t)];
        float x    = a[A_IDX(b, t, h)] + dt_bias[h];
        float g    = expf(-expf(A_log[h]) * softplus(x));
        float beta = sigmoid(b_logits[A_IDX(b, t, h)]);

        /* 1. Gate state in-place */
        for (int ki = 0; ki < K; ki++)
            S[ki] *= g;

        /* 2. old_v = dot(k, S) */
        float old_v = 0.0f;
        for (int ki = 0; ki < K; ki++)
            old_v = fmaf(k[K_IDX(b, t, h, ki)], S[ki], old_v);

        /* 3. new_v = beta * v + (1 - beta) * old_v */
        float new_v = beta * v[V_IDX(b, t, h, vi)] + (1.0f - beta) * old_v;

        /* 4. S += k * (new_v - old_v), masked */
        float v_diff = m * (new_v - old_v);
        for (int ki = 0; ki < K; ki++)
            S[ki] = fmaf(k[K_IDX(b, t, h, ki)], v_diff, S[ki]);

        /* 5. output = mask * scale * dot(q, S) */
        float o = 0.0f;
        for (int ki = 0; ki < K; ki++)
            o = fmaf(q[K_IDX(b, t, h, ki)], S[ki], o);

        output[V_IDX(b, t, h, vi)] = m * scale * o;
    }
}

torch::Tensor prefill_forward(
    torch::Tensor q,         /* [B, T, H, K] */
    torch::Tensor k,         /* [B, T, H, K] */
    torch::Tensor v,         /* [B, T, H, V] */
    torch::Tensor A_log,     /* [H]          */
    torch::Tensor a,         /* [B, T, H]    */
    torch::Tensor dt_bias,   /* [H]          */
    torch::Tensor b_logits,  /* [B, T, H]    */
    torch::Tensor mask,      /* [B, T]        */
    torch::Tensor state_in,  /* [B, H, V, K] */
    float       scale
) {
    const int B = q.size(0);
    const int T = q.size(1);

    TORCH_CHECK(q.size(2) == H && q.size(3) == K,   "q shape must be [B,T,16,128]");
    TORCH_CHECK(k.size(2) == H && k.size(3) == K,   "k shape must be [B,T,16,128]");
    TORCH_CHECK(v.size(2) == H && v.size(3) == V,   "v shape must be [B,T,16,128]");
    TORCH_CHECK(A_log.size(0) == H,                  "A_log shape must be [16]");
    TORCH_CHECK(a.size(0) == B && a.size(1) == T && a.size(2) == H,
                "a shape must be [B,T,16]");
    TORCH_CHECK(dt_bias.size(0) == H,                "dt_bias shape must be [16]");
    TORCH_CHECK(b_logits.sizes() == a.sizes(),        "b_logits shape must match a");
    TORCH_CHECK(mask.size(0) == B && mask.size(1) == T, "mask shape must be [B,T]");

    if (scale == 0.0)
        scale = 1.0 / sqrt((double)K);

    TORCH_CHECK(state_in.is_contiguous(),           "state_in must be contiguous");
    TORCH_CHECK(state_in.scalar_type() == torch::kFloat32, "state_in must be float32");
    TORCH_CHECK(state_in.size(0) == B &&
                state_in.size(1) == H &&
                state_in.size(2) == V  &&
                state_in.size(3) == K,  "state_in shape must be [B,16,256,128]");
    const float* state_in_ptr = state_in.data_ptr<float>();

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    torch::Tensor output = torch::zeros({B, T, H, V}, opts);

    dim3 grid(B, H);
    prefill<<<grid, V>>>(B, T, q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                     A_log.data_ptr<float>(), a.data_ptr<float>(), dt_bias.data_ptr<float>(),
                     b_logits.data_ptr<float>(), mask.data_ptr<float>(), state_in_ptr, scale,
                     output.data_ptr<float>());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gdn_prefill_kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &prefill_forward, "Prefill forward");
}
