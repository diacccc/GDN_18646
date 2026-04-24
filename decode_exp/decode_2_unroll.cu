/*
 * gdn_decode_kernel.cu
 * Fused CUDA kernel for the Gated DeltaNet (GDN) decode recurrence (T=1).
 *
 * Design (matches the report spec exactly):
 *   - One CUDA thread block per (b, h) pair  -> up to B*H blocks total
 *   - 128 threads per block (= dv), one thread owns column S[:,j]
 *   - q, k ∈ R^{dk=128} and v ∈ R^{dv=128} loaded collaboratively -> shared memory
 *   - Scalar params (a, b_gate, A_log, dt_bias, scale) loaded by thread 0 -> shared
 *   - State S[b,h,:,:] (dk×dv = 128×128 fp32 = 64 KB) streamed through registers
 *   - Coalesced 128-byte transactions: consecutive threads hit consecutive dv columns
 *
 * Per-thread computation (column j = threadIdx.x):
 *   g    = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h]))
 *   beta = sigmoid(b_gate[b,h])
 *   r_j  = g * dot(k, S[:,j])          -- dot over dk
 *   δ_j  = beta * (v[j] - r_j)
 *   S[:,j] <- g*S[:,j] + δ_j * k       -- rank-1 fused update (write-once)
 *   o_j  = scale * dot(q, S[:,j])      -- output projection
 *
 * Inputs  (all CUDA pointers, contiguous):
 *   q, k  : (B, H, dk) bf16
 *   v     : (B, H, dv) bf16
 *   a_sc  : (B, H)     bf16
 *   b_sc  : (B, H)     bf16
 *   A_log : (H,)       fp32
 *   dt_b  : (H,)       fp32
 *   S_in  : (B, H, dk, dv) fp32   -- read-only; we write S_out
 *   scale : fp32 scalar (passed via __constant__ or inline)
 *
 * Outputs:
 *   o_out : (B, H, dv) fp32
 *   S_out : (B, H, dk, dv) fp32
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>   // CUDART_INF_F
#include <stdio.h>

// ── Architecture constants (must match Python side) ──────────────────────────
#define DK  128
#define DV  256  

// ── Device helpers ───────────────────────────────────────────────────────────
__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float softplus_f32(float x) {
    // log(1 + exp(x)), numerically stable
    return (x > 20.f) ? x : log1pf(expf(x));
}

__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.f / (1.f + expf(-x));
}

// ── Main kernel ──────────────────────────────────────────────────────────────
// Template on dv so it can be instantiated for dv=128 or dv=256.
// Threads-per-block = dv_val.  Each thread handles column j = threadIdx.x.

#define PAIRS 8   // number of (b,h) pairs per block 

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
    int                               H_val,     // number of heads
    int                               BH_total   // B * H
) {
    const int j = threadIdx.x;

    extern __shared__ float smem[];
    float* q_sh = smem;
    float* k_sh = smem + DK;
    float* scal = smem + 2*DK;  // [0]=g, [1]=beta

    const int bh_base = blockIdx.x * PAIRS;

    for (int p = 0; p < PAIRS; ++p) {
        const int bh = bh_base + p;
        if (bh >= BH_total) break;

        const int b = bh / H_val;
        const int h = bh % H_val;

        // ── Base offsets ──────────────────────────────────────────────────────
        const int bh_qk = b * H_val * DK + h * DK;
        const int bh_v  = b * H_val * dv_val + h * dv_val;
        const int bh_sc = b * H_val + h;
        const int bh_S  = (b * H_val + h) * DK * dv_val;

        // ── Load q, k into shared memory ─────────────────────────────────────
        if (j < DK) {
            q_sh[j] = bf16_to_f32(q_ptr[bh_qk + j]);
            k_sh[j] = bf16_to_f32(k_ptr[bh_qk + j]);
        }

        // ── Scalars (thread 0) ────────────────────────────────────────────────
        if (j == 0) {
            float a_val  = bf16_to_f32(a_ptr[bh_sc]);
            float b_val  = bf16_to_f32(b_ptr[bh_sc]);
            scal[0] = expf(-expf(Alog_ptr[h]) * softplus_f32(a_val + dtb_ptr[h]));
            scal[1] = sigmoid_f32(b_val);
        }

        __syncthreads();

        const float g     = scal[0];
        const float beta  = scal[1];
        const float v_j   = bf16_to_f32(v_ptr[bh_v + j]);

        // ── Pass 1: 8-way unrolled, accumulate r_j ───────────────────────────
        float ks0=0.f, ks1=0.f;
        for (int i = 0; i < DK; i += 2) {
            float s0 = S_ptr[bh_S + (i+0)*dv_val + j];
            float s1 = S_ptr[bh_S + (i+1)*dv_val + j];
            ks0 += k_sh[i+0]*s0;
            ks1 += k_sh[i+1]*s1;
        }
        const float r_j     = g * (ks0+ks1);
        const float delta_j = beta * (v_j - r_j);

        // ── Pass 2: 8-way unrolled, write S_out and accumulate o_j ───────────
        float o_j = 0.f;
        for (int i = 0; i < DK; i += 2) {
            float s0 = S_ptr[bh_S + (i+0)*dv_val + j];
            float s1 = S_ptr[bh_S + (i+1)*dv_val + j];
            float sn0 = g*s0 + delta_j*k_sh[i+0];
            float sn1 = g*s1 + delta_j*k_sh[i+1];
            S_out_ptr[bh_S+(i+0)*dv_val+j] = sn0;
            S_out_ptr[bh_S+(i+1)*dv_val+j] = sn1;
            o_j += q_sh[i+0]*sn0 + q_sh[i+1]*sn1;
        }

        o_ptr[bh_v + j] = scale * o_j;

        // ── Sync before next pair so smem is safe to overwrite ────────────────
        __syncthreads();
    }
}

// ── Launcher ─────────────────────────────────────────────────────────────────
// Exposed to Python via ctypes / torch.utils.cpp_extension.
extern "C" {

void gdn_decode_cuda(
    const __nv_bfloat16* q,        // (B, H, DK)
    const __nv_bfloat16* k,        // (B, H, DK)
    const __nv_bfloat16* v,        // (B, H, DV)
    const __nv_bfloat16* a_sc,     // (B, H)
    const __nv_bfloat16* b_sc,     // (B, H)
    const float*          A_log,   // (H,)
    const float*          dt_bias, // (H,)
    const float*          S_in,    // (B, H, DK, DV)
    float                 scale,
    float*                o_out,   // (B, H, DV)
    float*                S_out,   // (B, H, DK, DV)
    int B, int H, int dv
) {
    int BH_total  = B * H;
    int num_blocks = (BH_total + PAIRS - 1) / PAIRS;  // ceiling division

    dim3 grid(num_blocks);          // 1D grid now
    size_t smem = (2*DK + 2) * sizeof(float);

    if (dv == 128) {
        gdn_decode_kernel<128><<<grid, 128, smem>>>(
            q, k, v, a_sc, b_sc, A_log, dt_bias, S_in, scale,
            o_out, S_out, H, BH_total);
    } else if (dv == 256) {
        gdn_decode_kernel<256><<<grid, 256, smem>>>(
            q, k, v, a_sc, b_sc, A_log, dt_bias, S_in, scale,
            o_out, S_out, H, BH_total);
    }
}

} // extern "C"


#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Macros for safety checks
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void gdn_decode_forward(
    at::Tensor q,       // (B, H, DK) bf16
    at::Tensor k,       // (B, H, DK) bf16
    at::Tensor v,       // (B, H, DV) bf16
    at::Tensor a_sc,    // (B, H)     bf16
    at::Tensor b_sc,    // (B, H)     bf16
    at::Tensor A_log,   // (H,)       fp32
    at::Tensor dt_bias, // (H,)       fp32
    at::Tensor S_in,    // (B, H, DK, DV) fp32
    float scale,
    at::Tensor o_out,   // (B, H, DV) fp32
    at::Tensor S_out    // (B, H, DK, DV) fp32
) {
    // 1. Validate all inputs
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

    // 2. Extract dimensions
    int B  = q.size(0);
    int H  = q.size(1);
    int dv = v.size(2);

    // 3. Call the CUDA launcher
    // Note: We cast data_ptr to the appropriate pointer types.
    // PyTorch's at::BFloat16 is binary-compatible with __nv_bfloat16.
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
