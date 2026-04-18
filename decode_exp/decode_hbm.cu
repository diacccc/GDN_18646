/*
 * gdn_decode_kernel.cu
 * Fused CUDA kernel for the Gated DeltaNet (GDN) decode recurrence (T=1).
 *
 * Design:
 *   - One CUDA thread block per (b, h) pair  → B*H blocks total
 *   - dv_val threads per block, one thread owns column S[:,j]
 *   - q, k ∈ R^{dk=128} and v ∈ R^{dv} loaded collaboratively → shared memory
 *   - Scalar params (a, b_gate, A_log, dt_bias, scale) loaded by thread 0 → shared
 *   - State S[b,h,:,:] (dk×dv fp32) streamed through registers from global memory
 *   - Coalesced 128-byte transactions: consecutive threads hit consecutive dv columns
 *
 * Per-thread computation (column j = threadIdx.x):
 *   g    = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h]))
 *   beta = sigmoid(b_gate[b,h])
 *   r_j  = g * dot(k, S[:,j])
 *   δ_j  = beta * (v[j] - r_j)
 *   S[:,j] <- g*S[:,j] + δ_j * k
 *   o_j  = scale * dot(q, S[:,j])
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdio.h>

// ── Architecture constants (must match Python side) ──────────────────────────
#define DK  128
#define DV  256

// ── Device helpers ───────────────────────────────────────────────────────────
__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float softplus_f32(float x) {
    return (x > 20.f) ? x : log1pf(expf(x));
}

__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.f / (1.f + expf(-x));
}

// ── Main kernel ──────────────────────────────────────────────────────────────
// One block per (b, h) pair. blockIdx.x = b * H + h.
// Threads-per-block = dv_val. Each thread handles column j = threadIdx.x.

template <int dv_val>
__global__ void gdn_decode_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ v_ptr,
    const __nv_bfloat16* __restrict__ a_ptr,
    const __nv_bfloat16* __restrict__ b_ptr,
    const float*          __restrict__ Alog_ptr,
    const float*          __restrict__ dtb_ptr,
    const float*          __restrict__ S_ptr,
    float                             scale,
    float*               __restrict__ o_ptr,
    float*               __restrict__ S_out_ptr,
    int                               H_val
) {
    const int j  = threadIdx.x;
    const int bh = blockIdx.x;       // one block per (b,h) pair
    const int b  = bh / H_val;
    const int h  = bh % H_val;

    // ── Shared memory layout ─────────────────────────────────────────────────
    // [0      .. DK-1  ] q       (DK floats)
    // [DK     .. 2*DK-1] k       (DK floats)
    // [2*DK   .. 2*DK+1] g, beta (2 floats)
    extern __shared__ float smem[];
    float* q_sh = smem;
    float* k_sh = smem + DK;
    float* scal = smem + 2*DK;   // [0]=g, [1]=beta

    // ── Global memory offsets ────────────────────────────────────────────────
    const int bh_qk = b * H_val * DK     + h * DK;
    const int bh_v  = b * H_val * dv_val + h * dv_val;
    const int bh_sc = bh;                            // b * H_val + h
    const int bh_S  = bh * DK * dv_val;

    // ── Load q, k into shared memory ─────────────────────────────────────────
    if (j < DK) {
        q_sh[j] = bf16_to_f32(q_ptr[bh_qk + j]);
        k_sh[j] = bf16_to_f32(k_ptr[bh_qk + j]);
    }

    // ── Scalars via thread 0 ─────────────────────────────────────────────────
    if (j == 0) {
        float a_val = bf16_to_f32(a_ptr[bh_sc]);
        float b_val = bf16_to_f32(b_ptr[bh_sc]);
        scal[0] = expf(-expf(Alog_ptr[h]) * softplus_f32(a_val + dtb_ptr[h]));
        scal[1] = sigmoid_f32(b_val);
    }

    __syncthreads();

    const float g    = scal[0];
    const float beta = scal[1];
    const float v_j  = bf16_to_f32(v_ptr[bh_v + j]);

    // ── Pass 1: accumulate r_j from global S ─────────────────────────────────
    float ks0 = 0.f;
    for (int i = 0; i < DK; i++) {
        ks0 += k_sh[i] * S_ptr[bh_S + i * dv_val + j];
    }
    const float r_j     = g * ks0;
    const float delta_j = beta * (v_j - r_j);

    // ── Pass 2: write S_out and accumulate o_j ────────────────────────────────
    float o_j = 0.f;
    for (int i = 0; i < DK; i++) {
        float sn = g * S_ptr[bh_S + i * dv_val + j] + delta_j * k_sh[i];
        S_out_ptr[bh_S + i * dv_val + j] = sn;
        o_j += q_sh[i] * sn;
    }

    o_ptr[bh_v + j] = scale * o_j;
    // No trailing __syncthreads() — block is done.
}

// ── Launcher ─────────────────────────────────────────────────────────────────
extern "C" {

void gdn_decode_cuda(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* a_sc,
    const __nv_bfloat16* b_sc,
    const float*          A_log,
    const float*          dt_bias,
    const float*          S_in,
    float                 scale,
    float*                o_out,
    float*                S_out,
    int B, int H, int dv
) {
    const int BH = B * H;
    dim3 grid(BH);                                    // one block per (b,h)
    size_t smem = (2*DK + 2) * sizeof(float);        // q + k + g/beta

    if (dv == 128) {
        gdn_decode_kernel<128><<<grid, 128, smem>>>(
            q, k, v, a_sc, b_sc, A_log, dt_bias, S_in,
            scale, o_out, S_out, H);
    } else if (dv == 256) {
        gdn_decode_kernel<256><<<grid, 256, smem>>>(
            q, k, v, a_sc, b_sc, A_log, dt_bias, S_in,
            scale, o_out, S_out, H);
    }
}

} // extern "C"


#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x)       TORCH_CHECK(x.device().is_cuda(),  #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(),     #x " must be contiguous")
#define CHECK_INPUT(x)      CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void gdn_decode_forward(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a_sc,
    at::Tensor b_sc,
    at::Tensor A_log,
    at::Tensor dt_bias,
    at::Tensor S_in,
    float scale,
    at::Tensor o_out,
    at::Tensor S_out
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(a_sc);
    CHECK_INPUT(b_sc);
    CHECK_INPUT(A_log);
    CHECK_INPUT(dt_bias);
    CHECK_INPUT(S_in);
    CHECK_INPUT(o_out);
    CHECK_INPUT(S_out);

    int B  = q.size(0);
    int H  = q.size(1);
    int dv = v.size(2);

    gdn_decode_cuda(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(a_sc.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(b_sc.data_ptr<at::BFloat16>()),
        A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        S_in.data_ptr<float>(),
        scale,
        o_out.data_ptr<float>(),
        S_out.data_ptr<float>(),
        B, H, dv
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gdn_decode_forward, "GDN decode forward");
}