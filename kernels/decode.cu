/*
 * gdn_decode_kernel.cu
 * Fused CUDA kernel for the Gated DeltaNet (GDN) decode recurrence (T=1).
 *
 * Design (matches the report spec exactly):
 *   - One CUDA thread block per (b, h) pair  -> up to B*H blocks total
 *   - 128 threads per block (= dv), one thread owns column S[:,j]
 *   - q, k in R^{dk=128} and v in R^{dv=128} loaded collaboratively -> shared memory
 *   - Scalar params (a, b_gate, A_log, dt_bias, scale) loaded by thread 0 -> shared
 *   - State S[b,h,:,:] (dk x dv = 128 x 128 fp32 = 64 KB) streamed through registers
 *   - Coalesced 128-byte transactions: consecutive threads hit consecutive dv columns
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdio.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ── Architecture constants ───────────────────────────────────────────────────
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
#define PAIRS 2

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
    int                               H_val,
    int                               BH_total
) {
    const int j = threadIdx.x;

    extern __shared__ float smem[];
    float* q_sh = smem;
    float* k_sh = smem + DK;
    float* scal = smem + 2*DK;

    const int bh_base = blockIdx.x * PAIRS;

    for (int p = 0; p < PAIRS; ++p) {
        const int bh = bh_base + p;
        if (bh >= BH_total) break;

        const int b = bh / H_val;
        const int h = bh % H_val;

        const int bh_qk = b * H_val * DK + h * DK;
        const int bh_v  = b * H_val * dv_val + h * dv_val;
        const int bh_sc = b * H_val + h;
        const int bh_S  = (b * H_val + h) * DK * dv_val;

        if (j < DK) {
            q_sh[j] = bf16_to_f32(q_ptr[bh_qk + j]);
            k_sh[j] = bf16_to_f32(k_ptr[bh_qk + j]);
        }

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

        // Pass 1: accumulate r_j
        float ks0=0.f, ks1=0.f, ks2=0.f, ks3=0.f;
        float ks4=0.f, ks5=0.f, ks6=0.f, ks7=0.f;
        for (int i = 0; i < DK; i += 8) {
            float s0 = S_ptr[bh_S + (i+0)*dv_val + j];
            float s1 = S_ptr[bh_S + (i+1)*dv_val + j];
            float s2 = S_ptr[bh_S + (i+2)*dv_val + j];
            float s3 = S_ptr[bh_S + (i+3)*dv_val + j];
            float s4 = S_ptr[bh_S + (i+4)*dv_val + j];
            float s5 = S_ptr[bh_S + (i+5)*dv_val + j];
            float s6 = S_ptr[bh_S + (i+6)*dv_val + j];
            float s7 = S_ptr[bh_S + (i+7)*dv_val + j];
            ks0 += k_sh[i+0]*s0; ks1 += k_sh[i+1]*s1;
            ks2 += k_sh[i+2]*s2; ks3 += k_sh[i+3]*s3;
            ks4 += k_sh[i+4]*s4; ks5 += k_sh[i+5]*s5;
            ks6 += k_sh[i+6]*s6; ks7 += k_sh[i+7]*s7;
        }
        const float r_j     = g * (ks0+ks1+ks2+ks3+ks4+ks5+ks6+ks7);
        const float delta_j = beta * (v_j - r_j);

        // Pass 2: write S_out and accumulate o_j
        float o_j = 0.f;
        for (int i = 0; i < DK; i += 8) {
            float s0 = S_ptr[bh_S + (i+0)*dv_val + j];
            float s1 = S_ptr[bh_S + (i+1)*dv_val + j];
            float s2 = S_ptr[bh_S + (i+2)*dv_val + j];
            float s3 = S_ptr[bh_S + (i+3)*dv_val + j];
            float s4 = S_ptr[bh_S + (i+4)*dv_val + j];
            float s5 = S_ptr[bh_S + (i+5)*dv_val + j];
            float s6 = S_ptr[bh_S + (i+6)*dv_val + j];
            float s7 = S_ptr[bh_S + (i+7)*dv_val + j];
            float sn0 = g*s0 + delta_j*k_sh[i+0];
            float sn1 = g*s1 + delta_j*k_sh[i+1];
            float sn2 = g*s2 + delta_j*k_sh[i+2];
            float sn3 = g*s3 + delta_j*k_sh[i+3];
            float sn4 = g*s4 + delta_j*k_sh[i+4];
            float sn5 = g*s5 + delta_j*k_sh[i+5];
            float sn6 = g*s6 + delta_j*k_sh[i+6];
            float sn7 = g*s7 + delta_j*k_sh[i+7];
            S_out_ptr[bh_S+(i+0)*dv_val+j] = sn0;
            S_out_ptr[bh_S+(i+1)*dv_val+j] = sn1;
            S_out_ptr[bh_S+(i+2)*dv_val+j] = sn2;
            S_out_ptr[bh_S+(i+3)*dv_val+j] = sn3;
            S_out_ptr[bh_S+(i+4)*dv_val+j] = sn4;
            S_out_ptr[bh_S+(i+5)*dv_val+j] = sn5;
            S_out_ptr[bh_S+(i+6)*dv_val+j] = sn6;
            S_out_ptr[bh_S+(i+7)*dv_val+j] = sn7;
            o_j += q_sh[i+0]*sn0 + q_sh[i+1]*sn1
                 + q_sh[i+2]*sn2 + q_sh[i+3]*sn3
                 + q_sh[i+4]*sn4 + q_sh[i+5]*sn5
                 + q_sh[i+6]*sn6 + q_sh[i+7]*sn7;
        }

        o_ptr[bh_v + j] = scale * o_j;
        __syncthreads();
    }
}

// ── PyTorch wrapper ───────────────────────────────────────────────────────────
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void gdn_decode_forward(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor a_sc, at::Tensor b_sc,
    at::Tensor A_log, at::Tensor dt_bias, at::Tensor S_in,
    float scale, at::Tensor o_out, at::Tensor S_out
) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);
    CHECK_INPUT(a_sc); CHECK_INPUT(b_sc);
    CHECK_INPUT(A_log); CHECK_INPUT(dt_bias);
    CHECK_INPUT(S_in); CHECK_INPUT(o_out); CHECK_INPUT(S_out);

    int B  = q.size(0);
    int H  = q.size(1);
    int dv = v.size(2);

    int BH_total  = B * H;
    int num_blocks = (BH_total + PAIRS - 1) / PAIRS;
    dim3 grid(num_blocks);
    size_t smem = (2*DK + 2) * sizeof(float);

    if (dv == 128) {
        gdn_decode_kernel<128><<<grid, 128, smem>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(a_sc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b_sc.data_ptr<at::BFloat16>()),
            A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
            S_in.data_ptr<float>(), scale,
            o_out.data_ptr<float>(), S_out.data_ptr<float>(),
            H, BH_total);
    } else if (dv == 256) {
        gdn_decode_kernel<256><<<grid, 256, smem>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(a_sc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b_sc.data_ptr<at::BFloat16>()),
            A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
            S_in.data_ptr<float>(), scale,
            o_out.data_ptr<float>(), S_out.data_ptr<float>(),
            H, BH_total);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gdn_decode_forward, "GDN decode forward");
}
