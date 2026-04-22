// prefill_chunked_fused.cu
//
// Fused single-kernel GDN prefill + pybind11 bindings

#include <cuda_runtime.h>
#include <mma.h>
#include <math_functions.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

using namespace nvcuda::wmma;
namespace py = pybind11;

static constexpr int H_DIM = 16;
static constexpr int K_DIM = 128;
static constexpr int V_DIM = 256;
static constexpr int C_DIM = 64;

#define QK_IDX(b,t,h,k) ((b) * H_DIM * T * K_DIM + (h) * T * K_DIM + (t) * K_DIM + (k))
#define V_IDX(b,t,h,vv) ((b) * H_DIM * T * V_DIM + (h) * T * V_DIM + (t) * V_DIM + (vv))
#define O_IDX(b,t,h,vv) ((b) * H_DIM * T * V_DIM + (h) * T * V_DIM + (t) * V_DIM + (vv))
#define S_IDX(b,h,k,vv) ((b) * H_DIM * K_DIM * V_DIM + (h) * K_DIM * V_DIM + (k) * V_DIM + (vv))
#define A_IDX(b,t,h)    ((b) * H_DIM * T + (h) * T + (t))

__device__ __forceinline__ float softplus(float x) {
    if (x >= 20.0f) return x;
    return log1pf(expf(x));
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void gdn_prefill_fused(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a,
    const float dt_bias,
    const __nv_bfloat16* __restrict__ b_logits,
    const float* __restrict__ mask,
    const float* __restrict__ state_in,
    float* __restrict__ state_out,
    __nv_bfloat16* __restrict__ out,
    int B, int T, int N,
    float scale
) {
    const int bh = blockIdx.x;
    const int b = bh / H_DIM;
    const int h = bh % H_DIM;
    const int vi = threadIdx.x;

    __shared__ __nv_bfloat16 smem_k_bf16[C_DIM][K_DIM];
    __shared__ float smem_gcum[C_DIM];
    __shared__ float smem_beta[C_DIM];

    // Reused across non-overlapping phases: WY solve, Q tiles, and attn tiles.
    __shared__ union {
        float as_WY[C_DIM][C_DIM];
        __nv_bfloat16 as_q_bf16[C_DIM][K_DIM];
        float as_attn[C_DIM][C_DIM];
    } smem_work;

    // Dynamic shared memory: [C_DIM, V_DIM] scratch for kbS values.
    extern __shared__ float smem_dyn[];
    float* smem_kbS = smem_dyn;

    float S_reg[K_DIM];
    if (state_in != nullptr) {
        for (int ki = 0; ki < K_DIM; ki++) {
            S_reg[ki] = state_in[S_IDX(b, h, ki, vi)];
        }
    } else {
        for (int ki = 0; ki < K_DIM; ki++) {
            S_reg[ki] = 0.0f;
        }
    }

    const float A_val = A_log[h];

    for (int n = 0; n < N; n++) {
        const int chunk = n * C_DIM;

        // Step 1: build cumulative gate and beta for this chunk.
        if (vi < C_DIM) {
            const int t_abs = chunk + vi;
            const float m_t = mask[b * T + t_abs];
            const float x = __bfloat162float(a[A_IDX(b, t_abs, h)]) + dt_bias;
            smem_gcum[vi] = (m_t > 0.0f) ? -expf(A_val) * softplus(x) : 0.0f;
        } else if (vi < 2 * C_DIM) {
            const int i = vi - C_DIM;
            const int t_abs = chunk + i;
            smem_beta[i] = sigmoid(__bfloat162float(b_logits[A_IDX(b, t_abs, h)]));
        }
        __syncthreads();

        float val = (vi < C_DIM) ? smem_gcum[vi] : 0.0f;
        for (int stride = 1; stride < C_DIM; stride <<= 1) {
            const float tmp = (vi >= stride && vi < C_DIM) ? smem_gcum[vi - stride] : 0.0f;
            __syncthreads();
            if (vi < C_DIM) {
                val += tmp;
                smem_gcum[vi] = val;
            }
            __syncthreads();
        }

        if (vi < C_DIM) {
            smem_gcum[vi] = expf(smem_gcum[vi]);
        }
        __syncthreads();

        // Step 2: load keys for this chunk.
        for (int idx = vi; idx < C_DIM * K_DIM; idx += V_DIM) {
            const int ci = idx / K_DIM;
            const int ki = idx % K_DIM;
            smem_k_bf16[ci][ki] = k[QK_IDX(b, chunk + ci, h, ki)];
        }
        __syncthreads();

        // Step 3: construct WY (lower-triangular + diagonal).
        {
            const int n_lower = C_DIM * (C_DIM - 1) / 2;
            for (int idx = vi; idx < n_lower; idx += V_DIM) {
                const int ti = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)idx)) * 0.5f);
                const int si = idx - ti * (ti - 1) / 2;
                const float G_ts = smem_gcum[ti] / smem_gcum[si];
                const float m_ti = mask[b * T + chunk + ti];
                const float b_ti = smem_beta[ti] * m_ti;

                float dot = 0.0f;
                for (int ki = 0; ki < K_DIM; ki++) {
                    const float k_ti = __bfloat162float(smem_k_bf16[ti][ki]);
                    const float k_si = __bfloat162float(smem_k_bf16[si][ki]);
                    dot += b_ti * k_ti * k_si;
                }
                smem_work.as_WY[ti][si] = -G_ts * dot;
            }

            for (int idx = vi; idx < C_DIM * C_DIM; idx += V_DIM) {
                const int ti = idx / C_DIM;
                const int si = idx % C_DIM;
                if (si >= ti) {
                    smem_work.as_WY[ti][si] = (si == ti) ? 1.0f : 0.0f;
                }
            }
        }
        __syncthreads();

        // Step 4: forward substitution in-place.
        for (int i = 1; i < C_DIM; i++) {
            float acc = 0.0f;
            if (vi < i) {
                for (int j = 0; j < i; j++) {
                    acc += smem_work.as_WY[i][j] * smem_work.as_WY[j][vi];
                }
            }
            __syncthreads();
            if (vi < C_DIM) {
                smem_work.as_WY[i][vi] += acc;
            }
            __syncthreads();
        }

        // Step 5: build dV_intra in registers.
        float dV_reg[C_DIM];
        for (int i = 0; i < C_DIM; i++) {
            float acc = 0.0f;
            for (int s = 0; s < C_DIM; s++) {
                const float m_s = mask[b * T + chunk + s];
                acc += smem_work.as_WY[i][s]
                     * smem_beta[s] * m_s
                     * __bfloat162float(v[V_IDX(b, chunk + s, h, vi)]);
            }
            dV_reg[i] = acc;
        }

        // Step 6: subtract cross-chunk S contribution from dV.
        for (int s = 0; s < C_DIM; s++) {
            const float m_s = mask[b * T + chunk + s];
            const float coeff = smem_beta[s] * m_s * smem_gcum[s];
            float dot = 0.0f;
            for (int ki = 0; ki < K_DIM; ki++) {
                dot += coeff * __bfloat162float(smem_k_bf16[s][ki]) * S_reg[ki];
            }
            smem_kbS[s * V_DIM + vi] = dot;
        }
        __syncthreads();

        for (int i = 0; i < C_DIM; i++) {
            float old_v = 0.0f;
            for (int s = 0; s < C_DIM; s++) {
                old_v += smem_work.as_WY[i][s] * smem_kbS[s * V_DIM + vi];
            }
            dV_reg[i] -= old_v;
        }
        __syncthreads();

        // Step 7: load Q tiles into aliased shared memory for WMMA.
        for (int idx = vi; idx < C_DIM * K_DIM; idx += V_DIM) {
            const int ci = idx / K_DIM;
            const int ki = idx % K_DIM;
            smem_work.as_q_bf16[ci][ki] = __float2bfloat16(
                __bfloat162float(q[QK_IDX(b, chunk + ci, h, ki)]) * scale
            );
        }
        __syncthreads();

        // Step 8: compute attention q @ k^T via WMMA into as_attn.
        {
            const int warp_id = vi / 32;
            const int warp_row = warp_id / 2;
            const int warp_col = warp_id % 2;
            const int col0 = warp_col;
            const int col1 = warp_col + 2;

            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> q_frag;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> k_frag;
            fragment<accumulator, 16, 16, 16, float> c_frag0;
            fragment<accumulator, 16, 16, 16, float> c_frag1;

            fill_fragment(c_frag0, 0.0f);
            fill_fragment(c_frag1, 0.0f);

            for (int k_tile = 0; k_tile < K_DIM / 16; k_tile++) {
                load_matrix_sync(q_frag,
                    smem_work.as_q_bf16[warp_row * 16] + k_tile * 16,
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

            store_matrix_sync(&smem_work.as_attn[warp_row * 16][col0 * 16],
                              c_frag0, C_DIM, mem_row_major);
            store_matrix_sync(&smem_work.as_attn[warp_row * 16][col1 * 16],
                              c_frag1, C_DIM, mem_row_major);
        }
        __syncthreads();

        // Step 9: apply causal gate scaling to attention.
        for (int idx = vi; idx < C_DIM * C_DIM; idx += V_DIM) {
            const int c = idx / C_DIM;
            const int s = idx % C_DIM;
            const float G_cs = smem_gcum[c] / smem_gcum[s];
            smem_work.as_attn[c][s] = (s <= c) ? smem_work.as_attn[c][s] * G_cs : 0.0f;
        }
        __syncthreads();

        // Step 10: write outputs.
        for (int c = 0; c < C_DIM; c++) {
            const int t_abs = chunk + c;
            const float gcum_c = smem_gcum[c];
            const float m_c = mask[b * T + t_abs];

            float o_cross = 0.0f;
            for (int ki = 0; ki < K_DIM; ki++) {
                o_cross += __bfloat162float(q[QK_IDX(b, t_abs, h, ki)]) * scale * gcum_c * S_reg[ki];
            }

            float o_intra = 0.0f;
            for (int s = 0; s <= c; s++) {
                o_intra += smem_work.as_attn[c][s] * dV_reg[s];
            }

            out[O_IDX(b, t_abs, h, vi)] = __float2bfloat16(m_c * (o_cross + o_intra));
        }

        // Step 11: update recurrent state.
        {
            const float last_gcum = smem_gcum[C_DIM - 1];
            for (int ki = 0; ki < K_DIM; ki++) {
                S_reg[ki] *= last_gcum;
            }

            for (int c = 0; c < C_DIM; c++) {
                const float pos_decay = smem_gcum[C_DIM - 1] / smem_gcum[c];
                const float dv_c = dV_reg[c];
                for (int ki = 0; ki < K_DIM; ki++) {
                    S_reg[ki] += __bfloat162float(smem_k_bf16[c][ki]) * pos_decay * dv_c;
                }
            }
        }

        __syncthreads();
    }

    for (int ki = 0; ki < K_DIM; ki++) {
        state_out[S_IDX(b, h, ki, vi)] = S_reg[ki];
    }
}

std::tuple<torch::Tensor, torch::Tensor> gdn_prefill(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor A_log,
    torch::Tensor a,
    float dt_bias,
    torch::Tensor b_logits,
    torch::Tensor mask,
    torch::Tensor state_in,
    float scale
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
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

    auto out = torch::empty({B, H_DIM, T, V_DIM}, opts_bf16);
    auto state_out = torch::empty({B, H_DIM, K_DIM, V_DIM}, opts_f32);

    const float* state_in_ptr = state_in.numel() > 0 ? state_in.data_ptr<float>() : nullptr;

    dim3 grid(B * H_DIM);
    dim3 block(V_DIM);
    const size_t kbS_bytes = static_cast<size_t>(C_DIM) * static_cast<size_t>(V_DIM) * sizeof(float);
    cudaError_t attr_err = cudaFuncSetAttribute(
        gdn_prefill_fused,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kbS_bytes)
    );
    TORCH_CHECK(attr_err == cudaSuccess, "Failed to set dynamic shared memory attribute for gdn_prefill_fused");

    gdn_prefill_fused<<<grid, block, kbS_bytes>>>(
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
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        B, T, N, scale
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gdn_prefill_fused kernel launch failed");

    return {out, state_out};
}

PYBIND11_MODULE(gdn_prefill, m) {
    m.doc() = "GDN fused prefill kernel";

    m.def(
        "prefill", &gdn_prefill,
        "Fused GDN prefill.\n"
        "Returns: (out [B,H,T,V] bf16, state_out [B,H,K,V] fp32)",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("A_log"), py::arg("a"), py::arg("dt_bias"),
        py::arg("b_logits"), py::arg("mask"), py::arg("state_in"),
        py::arg("scale")
    );

    m.attr("H_DIM") = H_DIM;
    m.attr("K_DIM") = K_DIM;
    m.attr("V_DIM") = V_DIM;
    m.attr("C_DIM") = C_DIM;
}
