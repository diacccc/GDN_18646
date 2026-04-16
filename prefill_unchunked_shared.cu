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
template <typename scalar_t>
__global__ void prefill(
    int B, int T,
    const scalar_t *q,
    const scalar_t *k,
    const scalar_t *v,
    const float    *A_log,
    const scalar_t *a,
    const float    *dt_bias,
    const scalar_t *b_logits,
    const float    *mask,
    const float    *state_in,
    float           scale,
    scalar_t       *output
)
{
    int b  = blockIdx.x;
    int h  = blockIdx.y;
    int vi = threadIdx.x;

    /* State accumulator always in float32 for numerical stability */
    float S[K];

    __shared__ float smem_k[K], smem_q[K], smem_v[V];

    if (state_in) {
        for (int ki = 0; ki < K; ki++)
            S[ki] = state_in[SI_IDX(b, h, vi, ki)];
    } else {
        for (int ki = 0; ki < K; ki++)
            S[ki] = 0.0f;
    }

    for (int t = 0; t < T; t++) {
        float m    = mask[MASK_IDX(b, t)];
        /* Convert scalar_t -> float on every load */
        float x    = static_cast<float>(a[A_IDX(b, t, h)]) + dt_bias[h];
        float g    = expf(-expf(A_log[h]) * softplus(x));
        float beta = sigmoid(static_cast<float>(b_logits[A_IDX(b, t, h)]));

        __syncthreads();
        // cooperative loading (q,k,v)
        smem_v[vi] = static_cast<float>(v[V_IDX(b, t, h, vi)]);
        for (int i = vi; i < K; i += V) {
            smem_k[i] = static_cast<float>(k[K_IDX(b, t, h, i)]);
            smem_q[i] = static_cast<float>(q[K_IDX(b, t, h, i)]);
        }
        __syncthreads();

        float old_v = 0.0f;

        for (int ki = 0; ki < K; ki++){
            /* 1. Gate state in-place */
            S[ki] *= g;

            /* 2. old_v = dot(k, S) */
            old_v = fmaf(smem_k[ki], S[ki], old_v);
        }

        /* 3. new_v = beta * v + (1 - beta) * old_v */
        float new_v = beta * smem_v[vi] + (1.0f - beta) * old_v;

        float v_diff = m * (new_v - old_v);
        float o = 0.0f;
        for (int ki = 0; ki < K; ki++){
            /* 4. S += k * (new_v - old_v), masked */
            S[ki] = fmaf(smem_k[ki], v_diff, S[ki]);

            /* 5. output = mask * scale * dot(q, S) */
            o = fmaf(smem_q[ki], S[ki], o);
        }

        /* Convert float -> scalar_t on store */
        output[V_IDX(b, t, h, vi)] = static_cast<scalar_t>(m * scale * o);
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
    TORCH_CHECK(v.size(2) == H && v.size(3) == V,   "v shape must be [B,T,16,256]");
    TORCH_CHECK(A_log.size(0) == H,                  "A_log shape must be [16]");
    TORCH_CHECK(a.size(0) == B && a.size(1) == T && a.size(2) == H,
                "a shape must be [B,T,16]");
    TORCH_CHECK(dt_bias.size(0) == H,                "dt_bias shape must be [16]");
    TORCH_CHECK(b_logits.sizes() == a.sizes(),        "b_logits shape must match a");
    TORCH_CHECK(mask.size(0) == B && mask.size(1) == T, "mask shape must be [B,T]");

    /* q/k/v/a/b_logits accept float32 or bfloat16 */
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 ||
                q.scalar_type() == torch::kBFloat16,
                "q must be float32 or bfloat16");

    /* A_log, dt_bias, mask, state_in always remain float32 */
    TORCH_CHECK(A_log.scalar_type()   == torch::kFloat32, "A_log must be float32");
    TORCH_CHECK(dt_bias.scalar_type() == torch::kFloat32, "dt_bias must be float32");
    TORCH_CHECK(mask.scalar_type()    == torch::kFloat32, "mask must be float32");

    if (scale == 0.0)
        scale = 1.0f / sqrtf((float)K);

    /* ---- Optional state_in ---- */
    TORCH_CHECK(state_in.is_contiguous(),           "state_in must be contiguous");
    TORCH_CHECK(state_in.scalar_type() == torch::kFloat32, "state_in must be float32");
    TORCH_CHECK(state_in.size(0) == B &&
                state_in.size(1) == H &&
                state_in.size(2) == V  &&
                state_in.size(3) == K,  "state_in shape must be [B,16,256,128]");
    const float* state_in_ptr = state_in.data_ptr<float>();

    /* ---- Allocate outputs (dtype follows q) ---- */
    auto opts = torch::TensorOptions()
                    .dtype(q.scalar_type())
                    .device(q.device());

    torch::Tensor output = torch::zeros({B, T, H, V}, opts);

    dim3 grid(B, H);

    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16, q.scalar_type(), "prefill_forward", [&] {
            prefill<scalar_t><<<grid, V>>>(
                B, T,
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                A_log.data_ptr<float>(),
                a.data_ptr<scalar_t>(),
                dt_bias.data_ptr<float>(),
                b_logits.data_ptr<scalar_t>(),
                mask.data_ptr<float>(),
                state_in_ptr,
                scale,
                output.data_ptr<scalar_t>()
            );
        }
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "prefill kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &prefill_forward, "Prefill forward");
}