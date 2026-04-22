/* prefill_chunked.cu - two-pass GDN prefill, bf16 I/O, fp32 state */

#include <cuda_runtime.h>
#include <mma.h>
#include <math_functions.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

using namespace nvcuda::wmma;
namespace py = pybind11;

/* ------------------------------------------------------------------ */
/* Compile-time constants                                               */
/* ------------------------------------------------------------------ */
static constexpr int H_DIM =  16;
static constexpr int K_DIM = 128;
static constexpr int V_DIM = 256;
static constexpr int C_DIM =  64;

/* ------------------------------------------------------------------ */
/* Index helpers                                                        */
/* ------------------------------------------------------------------ */
#define QK_IDX(b,t,h,k)    ((b)*H_DIM*T*K_DIM + (h)*T*K_DIM + (t)*K_DIM + (k))
#define V_IDX(b,t,h,vv)    ((b)*H_DIM*T*V_DIM + (h)*T*V_DIM + (t)*V_DIM + (vv))
#define O_IDX(b,t,h,vv)    ((b)*H_DIM*T*V_DIM + (h)*T*V_DIM + (t)*V_DIM + (vv))
#define S_IDX(b,h,k,vv)    ((b)*H_DIM*K_DIM*V_DIM + (h)*K_DIM*V_DIM + (k)*V_DIM + (vv))
#define A_IDX(b,t,h)       ((b)*H_DIM*T + (h)*T + (t))
#define DV_IDX(b,h,n,c,vv) ((b)*H_DIM*N*C_DIM*V_DIM + (h)*N*C_DIM*V_DIM + (n)*C_DIM*V_DIM + (c)*V_DIM + (vv))
#define KB_IDX(b,h,n,c,k)  ((b)*H_DIM*N*C_DIM*K_DIM + (h)*N*C_DIM*K_DIM + (n)*C_DIM*K_DIM + (c)*K_DIM + (k))
#define GC_IDX(b,h,n,c)     ((b)*H_DIM*N*C_DIM + (h)*N*C_DIM + (n)*C_DIM + (c))

__device__ __forceinline__ float softplus(float x) {
    if (x >=  20.0f) return x;
    return log1pf(expf(x));
}
__device__ __forceinline__ float sigmoid(float x)  { return 1.0f / (1.0f + expf(-x)); }

/* ------------------------------------------------------------------ */
/* Pass 1: grid=(B*H*N), block=V_DIM                                   */
/* ------------------------------------------------------------------ */
__global__ void gdn_prefill_pass1(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ A_log,       // [H]
    const __nv_bfloat16* __restrict__ a,
    const float               dt_bias,
    const __nv_bfloat16* __restrict__ b_logits,
    const float* __restrict__ mask,
          float* __restrict__ dV_intra,
          float* __restrict__ K_bgcum,
            float* __restrict__ Gcum,
    int B, int T, int N
) {
    const int bhn   = blockIdx.x;
    const int b     = bhn / (H_DIM * N);
    const int h     = (bhn / N) % H_DIM;
    const int n     = bhn % N;
    const int vi    = threadIdx.x;
    const int chunk = n * C_DIM;

    __shared__ __nv_bfloat16 smem_k_bf16 [C_DIM][K_DIM];
    __shared__ float          smem_WY[C_DIM][C_DIM];
    __shared__ float          smem_Gcum[C_DIM];
    __shared__ float          smem_inv_Gcum[C_DIM];
    __shared__ float          smem_beta   [C_DIM];

    const float A_val = A_log[h];

    /* Step 1: log_g → smem_Gcum, beta → smem_beta */
    if (vi < C_DIM) {
        int t_abs = chunk + vi;
        float m_t = mask[b * T + t_abs];
        float x   = __bfloat162float(a[A_IDX(b, t_abs, h)]) + dt_bias;
        smem_Gcum[vi] = (m_t > 0.0f) ? -expf(A_val) * softplus(x) : 0.0f;
    } else if (vi < C_DIM * 2) {
        int i     = vi - C_DIM;
        int t_abs = chunk + i;
        smem_beta[i] = sigmoid(__bfloat162float(b_logits[A_IDX(b, t_abs, h)]));
    }
    __syncthreads();

    /* Step 2: Hillis-Steele prefix sum on log_g */
    float val = (vi < C_DIM) ? smem_Gcum[vi] : 0.0f;
    for (int stride = 1; stride < C_DIM; stride <<= 1) {
        float tmp = (vi >= stride && vi < C_DIM) ? smem_Gcum[vi - stride] : 0.0f;
        __syncthreads();
        if (vi < C_DIM) {
            val += tmp;
            smem_Gcum[vi] = val;
        }
        __syncthreads();
    }
    __syncthreads();

    /* Step 3: log-gate → cumulative gate */
    if (vi < C_DIM) {
        smem_Gcum[vi] = expf(smem_Gcum[vi]);
        smem_inv_Gcum[vi] = 1.0f / smem_Gcum[vi];
    }
    __syncthreads();

    /* Step 4: load k → smem_k_bf16 */
    for (int idx = vi; idx < C_DIM * K_DIM; idx += V_DIM) {
        int ci = idx / K_DIM;
        int ki = idx % K_DIM;
        smem_k_bf16[ci][ki] = k[QK_IDX(b, chunk + ci, h, ki)];
    }
    __syncthreads();

    /* Step 5a: fill WY lower triangle, identity on diagonal and above */
    {
        int n_lower = C_DIM * (C_DIM - 1) / 2;
        for (int idx = vi; idx < n_lower; idx += V_DIM) {
            int ti = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)idx)) * 0.5f);
            int si = idx - ti * (ti - 1) / 2;
            float G_ts = smem_Gcum[ti] * smem_inv_Gcum[si];
            float dot  = 0.0f;
            float m_ti = mask[b * T + chunk + ti];
            float b_ti = smem_beta[ti] * m_ti;
            for (int ki = 0; ki < K_DIM; ki++) {
                float k_ti = __bfloat162float(smem_k_bf16[ti][ki]);
                float k_si = __bfloat162float(smem_k_bf16[si][ki]);
                dot += b_ti * k_ti * k_si;
            }
            smem_WY[ti][si] = (-G_ts * dot);
        }
        /* diagonal and upper triangle */
        for (int idx = vi; idx < C_DIM * C_DIM; idx += V_DIM) {
            int ti = idx / C_DIM;
            int si = idx % C_DIM;
            if (si >= ti)
                smem_WY[ti][si] = ((si == ti) ? 1.0f : 0.0f);
        }
    }
    __syncthreads();

    /* Step 5b: forward substitution in-place → (I-L)^{-1} */
    for (int i = 1; i < C_DIM; i++) {
        float acc = 0.0f;
        if (vi < i) {
            for (int j = 0; j < i; j++)
                acc += (smem_WY[i][j])
                     * (smem_WY[j][vi]);
        }
        __syncthreads();
        if (vi < C_DIM)
            smem_WY[i][vi] = (
                (smem_WY[i][vi]) + acc
            );
        __syncthreads();
    }
    /* Step 6: K_bgcum = WY @ (Kb * gcum) → gmem */
    for (int idx = vi; idx < C_DIM * K_DIM; idx += V_DIM) {
        int ti = idx / K_DIM;
        int ki = idx % K_DIM;
        float acc = 0.0f;
        for (int si = 0; si < C_DIM; si++) {
            float m_s  = mask[b * T + chunk + si];
            float gcum_s = smem_Gcum[si];
            float kb_s = smem_beta[si] * m_s
                    * __bfloat162float(smem_k_bf16[si][ki])
                    * gcum_s;
            acc += (smem_WY[ti][si]) * kb_s;
        }
        K_bgcum[KB_IDX(b, h, n, ti, ki)] = acc;
    }

    /* Step 7: dV_intra[i, vi] = sum_s WY[i,s] * beta[s] * v[s, vi] */
    float dV_reg[C_DIM];
    for (int i = 0; i < C_DIM; i++) {
        float acc = 0.0f;
        for (int s = 0; s < C_DIM; s++) {
            float m_s = mask[b * T + chunk + s];
            acc += (smem_WY[i][s])
                * smem_beta[s] * m_s
                * __bfloat162float(v[V_IDX(b, chunk + s, h, vi)]);
        }
        dV_reg[i] = acc;
    }

    /* Step 8: write dV_intra to gmem */
    for (int i = 0; i < C_DIM; i++)
        dV_intra[DV_IDX(b, h, n, i, vi)] = dV_reg[i];

    /* Step 9: persist cumulative gates for pass 2 */
    if (vi < C_DIM)
        Gcum[GC_IDX(b, h, n, vi)] = smem_Gcum[vi];
}


/* ------------------------------------------------------------------ */
/* Pass 2: grid=(B*H), block=V_DIM                                     */
/* ------------------------------------------------------------------ */
__global__ void gdn_prefill_pass2(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const float* __restrict__ mask,
    const float* __restrict__ Gcum,
    const float* __restrict__ dV_intra,
    const float* __restrict__ K_bgcum,
    const float* __restrict__ state_in,   // nullable
          float* __restrict__ state_out,
          __nv_bfloat16* __restrict__ out,
    int B, int T, int N,
    float scale
) {
    const int bh = blockIdx.x;
    const int b  = bh / H_DIM;
    const int h  = bh % H_DIM;
    const int vi = threadIdx.x;

    __shared__ __nv_bfloat16 smem_k_bf16[C_DIM][K_DIM];   /* 16 KB */

    /* union: as_q_bf16 for WMMA load, as_attn after store_matrix_sync - same size */
    __shared__ union {
        __nv_bfloat16 as_q_bf16[C_DIM][K_DIM];
        float         as_attn  [C_DIM][C_DIM];
    } smem_q_or_attn;                                      /* 16 KB */

    __shared__ float smem_Gcum[C_DIM];                      /* 0.25 KB */
    __shared__ float smem_inv_Gcum[C_DIM];                  /* 0.25 KB */

    /* Initialize S_reg */
    float S_reg[K_DIM];
    if (state_in != nullptr) {
        for (int ki = 0; ki < K_DIM; ki++)
            S_reg[ki] = state_in[S_IDX(b, h, ki, vi)];
    } else {
        for (int ki = 0; ki < K_DIM; ki++)
            S_reg[ki] = 0.0f;
    }

    /* Sequential chunk loop - S_reg in registers throughout */
    for (int n = 0; n < N; n++) {
        const int chunk = n * C_DIM;

        /* Step 1: load cumulative gates from pass 1 */
        if (vi < C_DIM) {
            smem_Gcum[vi] = Gcum[GC_IDX(b, h, n, vi)];
            smem_inv_Gcum[vi] = 1.0f / smem_Gcum[vi];
        }
        __syncthreads();

        /* Step 2: dV_intra -> dV_reg, subtract cross-chunk S contribution */
        float dV_reg[C_DIM];
        for (int i = 0; i < C_DIM; i++)
            dV_reg[i] = dV_intra[DV_IDX(b, h, n, i, vi)];

        for (int i = 0; i < C_DIM; i++) {
            float old_v = 0.0f;
            for (int ki = 0; ki < K_DIM; ki++)
                old_v += K_bgcum[KB_IDX(b, h, n, i, ki)] * S_reg[ki];
            dV_reg[i] -= old_v;
        }

        /* Step 3: load k → smem_k_bf16, q (scaled) → as_q_bf16 */
        for (int idx = vi; idx < C_DIM * K_DIM; idx += V_DIM) {
            int ci = idx / K_DIM;
            int ki = idx % K_DIM;
            smem_k_bf16[ci][ki] = k[QK_IDX(b, chunk + ci, h, ki)];
            smem_q_or_attn.as_q_bf16[ci][ki] = __float2bfloat16(
                __bfloat162float(q[QK_IDX(b, chunk + ci, h, ki)]) * scale);
        }
        __syncthreads();

        /* Step 4: attn = q @ k^T via WMMA → as_attn (fp32)
         * Phase A: compute fragments while as_q_bf16 is live.
         * Phase B: __syncthreads(), then store into as_attn. */
        {
            const int warp_id  = vi / 32;
            const int warp_row = warp_id / 2;   // 0..3
            const int warp_col = warp_id % 2;   // 0..1
            const int col0 = warp_col;
            const int col1 = warp_col + 2;

            fragment<matrix_a,    16, 16, 16, __nv_bfloat16, row_major> q_frag;
            fragment<matrix_b,    16, 16, 16, __nv_bfloat16, col_major> k_frag;
            fragment<accumulator, 16, 16, 16, float>                    c_frag0;
            fragment<accumulator, 16, 16, 16, float>                    c_frag1;

            fill_fragment(c_frag0, 0.0f);
            fill_fragment(c_frag1, 0.0f);

            /* Phase A: accumulate into c_frags */
            for (int k_tile = 0; k_tile < K_DIM / 16; k_tile++) {
                load_matrix_sync(q_frag,
                    smem_q_or_attn.as_q_bf16[warp_row * 16] + k_tile * 16,
                    K_DIM);

                load_matrix_sync(k_frag,
                    smem_k_bf16[col0 * 16] + k_tile * 16,
                    K_DIM);
                mma_sync(c_frag0, q_frag, k_frag, c_frag0);

                load_matrix_sync(k_frag,
                    smem_k_bf16[col1 * 16] + k_tile * 16,
                    K_DIM);
                mma_sync(c_frag1, q_frag, k_frag, c_frag1);
            }

            __syncthreads();

            /* Phase B: store into as_attn */
            store_matrix_sync(
                &smem_q_or_attn.as_attn[warp_row * 16][col0 * 16],
                c_frag0, C_DIM, mem_row_major);
            store_matrix_sync(
                &smem_q_or_attn.as_attn[warp_row * 16][col1 * 16],
                c_frag1, C_DIM, mem_row_major);
        }
        __syncthreads();

        /* Step 5: causal mask + G[c,s] = gcum[c]/gcum[s] applied in-place */
        for (int idx = vi; idx < C_DIM * C_DIM; idx += V_DIM) {
            int c    = idx / C_DIM;
            int s    = idx % C_DIM;
            float G_cs = smem_Gcum[c] * smem_inv_Gcum[s];
            smem_q_or_attn.as_attn[c][s] =
                (s <= c) ? smem_q_or_attn.as_attn[c][s] * G_cs : 0.0f;
        }
        __syncthreads();

        /* Step 6: output = o_cross (q·S scaled by gcum) + o_intra (attn·dV) */
        for (int c = 0; c < C_DIM; c++) {
            int   t_abs  = chunk + c;
            float gcum_c = smem_Gcum[c];
            float m_c    = mask[b * T + t_abs];

            float o_cross = 0.0f;
            for (int ki = 0; ki < K_DIM; ki++)
                o_cross += __bfloat162float(q[QK_IDX(b, t_abs, h, ki)])
                         * scale * gcum_c * S_reg[ki];

            float o_intra = 0.0f;
            for (int s = 0; s <= c; s++)
                o_intra += smem_q_or_attn.as_attn[c][s] * dV_reg[s];

            out[O_IDX(b, t_abs, h, vi)] = __float2bfloat16(m_c * (o_cross + o_intra));
        }

        /* Step 7: S = gcum[C-1]*S + sum_c k[c]*(gcum[C-1]/gcum[c])*dV[c] */
        {
            float last_gcum = smem_Gcum[C_DIM - 1];
            for (int ki = 0; ki < K_DIM; ki++)
                S_reg[ki] *= last_gcum;

            for (int c = 0; c < C_DIM; c++) {
                float pos_decay = smem_Gcum[C_DIM - 1] * smem_inv_Gcum[c];
                float dv_c      = dV_reg[c];
                for (int ki = 0; ki < K_DIM; ki++)
                    S_reg[ki] += __bfloat162float(smem_k_bf16[c][ki]) * pos_decay * dv_c;
            }
        }

        __syncthreads();
    } /* end N loop */

    /* Write S_reg to state_out */
    for (int ki = 0; ki < K_DIM; ki++)
        state_out[S_IDX(b, h, ki, vi)] = S_reg[ki];
}


/* ------------------------------------------------------------------ */
/* Host function                                                        */
/* ------------------------------------------------------------------ */
std::tuple<torch::Tensor, torch::Tensor> gdn_prefill(
    torch::Tensor q,            // [B, H, T, K]
    torch::Tensor k,            // [B, H, T, K]
    torch::Tensor v,            // [B, H, T, V]
    torch::Tensor A_log,        // [H]
    torch::Tensor a,            // [B, H, T]
    float         dt_bias,
    torch::Tensor b_logits,     // [B, H, T]
    torch::Tensor mask,         // [B, T]
    torch::Tensor state_in,     // [B, H, K, V]  pass empty tensor for zeros
    float         scale
) {
    TORCH_CHECK(q.is_cuda(),    "q must be a CUDA tensor");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.is_cuda() && k.is_contiguous() && k.dtype() == torch::kBFloat16, "k must be contiguous bfloat16 CUDA");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.dtype() == torch::kBFloat16, "v must be contiguous bfloat16 CUDA");
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dtype() == torch::kBFloat16, "a must be contiguous bfloat16 CUDA");
    TORCH_CHECK(b_logits.is_cuda() && b_logits.is_contiguous() && b_logits.dtype() == torch::kBFloat16, "b_logits must be contiguous bfloat16 CUDA");

    const int B = q.size(0);
    const int T = q.size(2);
    const int N = T / C_DIM;

    TORCH_CHECK(T % C_DIM == 0, "T must be divisible by C=", C_DIM);
    TORCH_CHECK(q.size(1) == H_DIM, "H must be ", H_DIM);
    TORCH_CHECK(q.size(3) == K_DIM, "K must be ", K_DIM);
    TORCH_CHECK(v.size(3) == V_DIM, "V must be ", V_DIM);

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());

    /* Intermediate buffers */
    auto dV_intra = torch::empty({B, H_DIM, N, C_DIM, V_DIM}, opts_f32);
    auto K_bgcum  = torch::empty({B, H_DIM, N, C_DIM, K_DIM}, opts_f32);
    auto Gcum     = torch::empty({B, H_DIM, N, C_DIM}, opts_f32);

    /* Outputs */
    auto out       = torch::empty({B, H_DIM, T, V_DIM}, opts_bf16);
    auto state_out = torch::empty({B, H_DIM, K_DIM, V_DIM}, opts_f32);

    const float* state_in_ptr = state_in.numel() > 0
        ? state_in.data_ptr<float>() : nullptr;

    /* Pass 1 */
    dim3 grid1(B * H_DIM * N);
    dim3 block(V_DIM);
    gdn_prefill_pass1<<<grid1, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        A_log.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        dt_bias,
        reinterpret_cast<const __nv_bfloat16*>(b_logits.data_ptr<at::BFloat16>()),
        mask.data_ptr<float>(),
        dV_intra.data_ptr<float>(),
        K_bgcum.data_ptr<float>(),
        Gcum.data_ptr<float>(),
        B, T, N
    );

    /* Pass 2 */
    dim3 grid2(B * H_DIM);
    gdn_prefill_pass2<<<grid2, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        mask.data_ptr<float>(),
        Gcum.data_ptr<float>(),
        dV_intra.data_ptr<float>(),
        K_bgcum.data_ptr<float>(),
        state_in_ptr,
        state_out.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        B, T, N, scale
    );

    return {out, state_out};
}

PYBIND11_MODULE(gdn_prefill, m) {
    m.doc() = "GDN two-pass prefill kernel";

    m.def("prefill", &gdn_prefill,
        "Two-pass GDN prefill.\n"
        "Args:\n"
        "  q, k       : [B, H, T, K] bfloat16 CUDA\n"
        "  v          : [B, H, T, V] bfloat16 CUDA\n"
        "  A_log      : [H]          float32 CUDA\n"
        "  a          : [B, H, T]    bfloat16 CUDA\n"
        "  dt_bias    : float scalar\n"
        "  b_logits   : [B, H, T]    bfloat16 CUDA\n"
        "  mask       : [B, T]       float32 CUDA\n"
        "  state_in   : [B, H, K, V] float32 CUDA (empty tensor = zero init)\n"
        "  scale      : float scalar (default 1/sqrt(K))\n"
        "Returns: (out [B,H,T,V] bf16, state_out [B,H,K,V] fp32)",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("A_log"), py::arg("a"), py::arg("dt_bias"),
        py::arg("b_logits"), py::arg("mask"), py::arg("state_in"),
        py::arg("scale")
    );

    /* Expose constants for Python-side shape validation */
    m.attr("H_DIM") = H_DIM;
    m.attr("K_DIM") = K_DIM;
    m.attr("V_DIM") = V_DIM;
    m.attr("C_DIM") = C_DIM;
}
