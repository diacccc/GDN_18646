/*
 * gdn_decode_kernel.cu
 * Fused CUDA kernel for the Gated DeltaNet (GDN) decode recurrence (T=1).
 *
 * Design (matches the report spec exactly):
 *   - One CUDA thread block per (b, h) pair  → up to B*H blocks total
 *   - 128 threads per block (= dv), one thread owns column S[:,j]
 *   - q, k ∈ R^{dk=128} and v ∈ R^{dv=128} loaded collaboratively → shared memory
 *   - Scalar params (a, b_gate, A_log, dt_bias, scale) loaded by thread 0 → shared
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
#define DV  256   // NOTE: in the notebook dv=256 per head but the paper/report
                  // describes 128 threads = dv per block; we keep dv=DV
                  // configurable via the template parameter below.

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

template <int dv_val>
__global__ void gdn_decode_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,     // (B, H, DK)
    const __nv_bfloat16* __restrict__ k_ptr,     // (B, H, DK)
    const __nv_bfloat16* __restrict__ v_ptr,     // (B, H, dv)
    const __nv_bfloat16* __restrict__ a_ptr,     // (B, H)
    const __nv_bfloat16* __restrict__ b_ptr,     // (B, H)
    const float*          __restrict__ Alog_ptr,  // (H,)
    const float*          __restrict__ dtb_ptr,   // (H,)
    const float*          __restrict__ S_ptr,     // (B, H, DK, dv)
    float                             scale,
    float*               __restrict__ o_ptr,     // (B, H, dv)
    float*               __restrict__ S_out_ptr  // (B, H, DK, dv)
) {
    // ── Block → (b, h) mapping ───────────────────────────────────────────────
    const int H_val = gridDim.y;            // number of heads (dynamic)
    const int b     = blockIdx.x;
    const int h     = blockIdx.y;
    const int j     = threadIdx.x;         // column of S this thread owns [0, dv)

    // ── Shared memory layout ─────────────────────────────────────────────────
    // [0         .. DK-1      ] : q_sh[DK]
    // [DK        .. 2*DK-1    ] : k_sh[DK]
    // [2*DK      .. 2*DK+dv-1 ] : v_sh[dv]
    // [2*DK+dv   .. 2*DK+dv+1] : scalars {g, beta}
    //
    // Total: 2*DK + dv + 2  floats  = 2*128 + 128 + 2 = 386 floats = 1.5 KB
    extern __shared__ float smem[];

    float* q_sh   = smem;               // [DK]
    float* k_sh   = smem + DK;          // [DK]
    float* v_sh   = smem + 2*DK;        // [dv]
    float* scal   = smem + 2*DK + dv_val; // [0]=g  [1]=beta

    // ── Base pointers for this (b, h) ────────────────────────────────────────
    const int bh_qk = b * H_val * DK + h * DK;
    const int bh_v  = b * H_val * dv_val + h * dv_val;
    const int bh_sc = b * H_val + h;
    const int bh_S  = b * H_val * DK * dv_val + h * DK * dv_val;

    // ── Collaborative vector load: thread j loads element j (or wraps) ───────
    // q and k are DK=128 elements; with 128 threads for dv=128 each thread
    // loads exactly one q[j] and k[j].
    // When dv_val > DK we still need to load DK elements; threads [0..DK-1] load.
    if (j < DK) {
        q_sh[j] = bf16_to_f32(q_ptr[bh_qk + j]);
        k_sh[j] = bf16_to_f32(k_ptr[bh_qk + j]);
    }
    v_sh[j] = bf16_to_f32(v_ptr[bh_v + j]);

    // ── Scalar parameters loaded by thread 0 ─────────────────────────────────
    if (j == 0) {
        float a_val   = bf16_to_f32(a_ptr[bh_sc]);
        float b_val   = bf16_to_f32(b_ptr[bh_sc]);
        float Alog_h  = Alog_ptr[h];
        float dtb_h   = dtb_ptr[h];
        float raw     = a_val + dtb_h;
        scal[0] = expf(-expf(Alog_h) * softplus_f32(raw));  // g
        scal[1] = sigmoid_f32(b_val);                        // beta
    }

    __syncthreads();  // All shared memory ready

    // ── Register-streaming: load column S[:,j] from global memory ────────────
    // S layout: (B, H, DK, dv) — dv is the contiguous (innermost) dimension.
    // Thread j accesses S[b,h, i, j] = S_ptr[bh_S + i*dv_val + j]  for i in [0,DK)
    // → threads 0..dv-1 access consecutive j → fully coalesced 128-byte transactions.

    float g    = scal[0];
    float beta = scal[1];

    // Broadcast-read q, k from shared memory (warp serves same index → 1 cycle)
    // Accumulate dot products r_j and o_j while streaming through S[:,j]

    float r_j  = 0.f;       // k^T S[:,j]  (before gated decay conceptually:
                             //  we apply decay inline during the stream)
    // Strategy: stream S, compute r first, then do a second pass for update+output.
    // BUT to keep only ONE pass (read-once / write-once), we need r_j before
    // we start updating. This requires a single accumulate pass (read only), then
    // a second pass (read-modify-write). That gives 2 full reads + 1 write of S.
    //
    // Alternative (report's approach): Note r_j = g * k^T S[:,j], so we can fold
    // the gated decay into the read and accumulate r in one streaming pass, then
    // do a second pass to write S_out[:,j] and compute o_j.
    //
    // Two-pass approach (2 reads + 1 write — optimal given data dependencies):
    //   Pass 1: stream read S[:,j] → accumulate r_j = g * dot(k, S[:,j])
    //   Pass 2: stream read+write S_out[:,j] = g*S[:,j] + δ_j*k, accumulate o_j

    // ── Pass 1: compute r_j = g * k^T S[:,j] ────────────────────────────────
    for (int i = 0; i < DK; ++i) {
        float s_ij = S_ptr[bh_S + i * dv_val + j];
        r_j += k_sh[i] * s_ij;
    }
    r_j *= g;

    // ── Compute delta_j ──────────────────────────────────────────────────────
    float v_j   = v_sh[j];
    float delta_j = beta * (v_j - r_j);

    // ── Pass 2: fused update + output projection ──────────────────────────────
    // S_out[:,j] = g * S[:,j] + delta_j * k
    // o_j        = scale * q^T S_out[:,j]
    float o_j = 0.f;
    for (int i = 0; i < DK; ++i) {
        float s_ij     = S_ptr[bh_S + i * dv_val + j];
        float s_new_ij = g * s_ij + delta_j * k_sh[i];
        S_out_ptr[bh_S + i * dv_val + j] = s_new_ij;
        o_j += q_sh[i] * s_new_ij;
    }
    o_j *= scale;

    // ── Write output ─────────────────────────────────────────────────────────
    o_ptr[bh_v + j] = o_j;
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
    dim3 grid(B, H);           // one block per (b, h)

    // Shared memory: 2*DK floats (q,k) + dv floats (v) + 2 floats (g,beta)
    size_t smem = (2 * DK + dv + 2) * sizeof(float);

    if (dv == 128) {
        gdn_decode_kernel<128><<<grid, 128, smem>>>(
            q, k, v, a_sc, b_sc, A_log, dt_bias, S_in, scale, o_out, S_out);
    } else if (dv == 256) {
        // 256 threads per block; q/k load handled by first 128 threads
        gdn_decode_kernel<256><<<grid, 256, smem>>>(
            q, k, v, a_sc, b_sc, A_log, dt_bias, S_in, scale, o_out, S_out);
    } else {
        printf("gdn_decode_cuda: unsupported dv=%d (must be 128 or 256)\n", dv);
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
