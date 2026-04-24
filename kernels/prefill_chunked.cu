/* prefill_chunked.cu - fused GDN prefill kernel, bf16 I/O, fp32 state */

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
static constexpr int H_DIM =  16;   // number of heads
static constexpr int K_DIM = 128;   // key / query dimension
static constexpr int V_DIM = 256;   // value dimension (= block size)
static constexpr int C_DIM =  64;   // chunk size

/* ------------------------------------------------------------------ */
/* Element-wise helpers                                                 */
/* ------------------------------------------------------------------ */
__device__ __forceinline__ float softplus(float x) {
    return (x >= 20.f) ? x : log1pf(expf(x));
}
__device__ __forceinline__ float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

/* ------------------------------------------------------------------ */
/* Index helpers (layout: [B,H,T,D] row-major)                          */
/* ------------------------------------------------------------------ */
#define QK_IDX(b,h,t,ki)  ((b)*H_DIM*T*K_DIM + (h)*T*K_DIM + (t)*K_DIM + (ki))
#define  V_IDX(b,h,t,vv)  ((b)*H_DIM*T*V_DIM + (h)*T*V_DIM + (t)*V_DIM + (vv))
#define  A_IDX(b,h,t)     ((b)*H_DIM*T + (h)*T + (t))
#define  S_IDX(b,h,ki,vv) ((b)*H_DIM*K_DIM*V_DIM + (h)*K_DIM*V_DIM + (ki)*V_DIM + (vv))

/* ------------------------------------------------------------------ */
/* Kernel: grid=(B*H_DIM,), block=(V_DIM=256,)                          */
/* ------------------------------------------------------------------ */
__global__ void __launch_bounds__(V_DIM, 1)
gdn_prefill_v2_kernel(
    const __nv_bfloat16* __restrict__ q,          // [B,H,T,K]  bf16
    const __nv_bfloat16* __restrict__ k,          // [B,H,T,K]  bf16
    const __nv_bfloat16* __restrict__ v,          // [B,H,T,V]  bf16
    const float*         __restrict__ A_log,      // [H]        fp32
    const __nv_bfloat16* __restrict__ a,          // [B,H,T]    bf16
    float                             dt_bias,    // scalar
    const __nv_bfloat16* __restrict__ b_logits,   // [B,H,T]    bf16
    const float*         __restrict__ mask,       // [B,T]      fp32
    const float*         __restrict__ state_in,   // [B,H,K,V]  fp32  (nullable)
          float*         __restrict__ state_out,  // [B,H,K,V]  fp32
          __nv_bfloat16* __restrict__ out,        // [B,H,T,V]  bf16
    int B, int T, int N,                          // N = T/C_DIM
    float scale
) {
    /* Block/thread identity */
    const int bh      = blockIdx.x;
    const int b       = bh / H_DIM;
    const int h       = bh % H_DIM;
    const int vi      = threadIdx.x;          // ∈ [0, V_DIM)
    const int warp_id = vi / 32;              // ∈ [0, 8)

    const float A_val = A_log[h];

    /* Dynamic shared memory layout (140032 B total, set via cudaFuncSetAttribute)
     * Allocated by the host launcher via the third kernel-launch argument.
     */
    extern __shared__ char smem_buf[];

    /* Byte offsets - same order and sizes as the original static layout. */
    constexpr int OFF_K       = 0;
    constexpr int OFF_GCUM    = OFF_K       + C_DIM * K_DIM * (int)sizeof(__nv_bfloat16); //  16384
    constexpr int OFF_BETA    = OFF_GCUM    + C_DIM          * (int)sizeof(float);         // +  256
    constexpr int OFF_MASK    = OFF_BETA    + C_DIM          * (int)sizeof(float);         // +  256
    constexpr int OFF_WY      = OFF_MASK    + C_DIM          * (int)sizeof(float);         // +  256
    constexpr int OFF_WY_BF16 = OFF_WY     + C_DIM * C_DIM  * (int)sizeof(float);         // +16384
    constexpr int OFF_VQ      = OFF_WY_BF16 + C_DIM * C_DIM * (int)sizeof(__nv_bfloat16); // + 8192
    constexpr int OFF_DV      = OFF_VQ     + C_DIM * V_DIM  * (int)sizeof(__nv_bfloat16); // +32768

    auto* smem_k       = reinterpret_cast<__nv_bfloat16 (*)[K_DIM]>(smem_buf + OFF_K);
    auto* smem_gcum    = reinterpret_cast<float *>(smem_buf + OFF_GCUM);
    auto* smem_beta    = reinterpret_cast<float *>(smem_buf + OFF_BETA);
    auto* smem_mask_s  = reinterpret_cast<float *>(smem_buf + OFF_MASK);
    auto* smem_WY      = reinterpret_cast<float (*)[C_DIM]>(smem_buf + OFF_WY);
    auto* smem_WY_bf16 = reinterpret_cast<__nv_bfloat16 (*)[C_DIM]>(smem_buf + OFF_WY_BF16);
    /* Former union: two aliases at the same base offset (phases are non-overlapping). */
    auto* smem_vbeta   = reinterpret_cast<__nv_bfloat16 (*)[V_DIM]>(smem_buf + OFF_VQ);
    auto* smem_q_bf16  = reinterpret_cast<__nv_bfloat16 (*)[K_DIM]>(smem_buf + OFF_VQ);
    auto* smem_dV      = reinterpret_cast<float (*)[V_DIM]>(smem_buf + OFF_DV);

    /* Initialise recurrent state S: S_reg[ki] = S[b,h,ki,vi] */
    float S_reg[K_DIM];
    if (state_in != nullptr) {
        for (int ki = 0; ki < K_DIM; ki++)
            S_reg[ki] = state_in[S_IDX(b, h, ki, vi)];
    } else {
        for (int ki = 0; ki < K_DIM; ki++)
            S_reg[ki] = 0.f;
    }

    /* Main chunk loop (ref_prefill lines 136-157) */
    for (int n = 0; n < N; n++) {
        const int chunk = n * C_DIM;   // first absolute token index in this chunk

        /* Step A: gates and betas (ref_prefill lines 55-58, 70-71, 101-102)
         *   log_g[c] = -exp(A_val)*softplus(a[c] + dt_bias)  if mask[c]>0, else 0
         *   gcum[c]  = exp(sum_{i=0..c} log_g[i])  = g_0*g_1*...*g_c
         *   beta[c]  = sigmoid(b_logits[c])
         */
        if (vi < C_DIM) {
            int t = chunk + vi;
            float m_t = mask[b * T + t];
            float x   = __bfloat162float(a[A_IDX(b, h, t)]) + dt_bias;
            smem_gcum  [vi] = (m_t > 0.f) ? -expf(A_val) * softplus(x) : 0.f;
            smem_mask_s[vi] = m_t;
        } else if (vi < 2 * C_DIM) {
            int t = chunk + (vi - C_DIM);
            smem_beta[vi - C_DIM] = sigmoid(__bfloat162float(b_logits[A_IDX(b, h, t)]));
        }
        __syncthreads(); /* (A) gcum/beta/mask fully written */

        /* Hillis-Steele prefix sum on log_g -> cumulative log-gate.
         * After this, smem_gcum[c] = sum_{i=0..c} log_g[i]. */
        float lgc = (vi < C_DIM) ? smem_gcum[vi] : 0.f;
        for (int stride = 1; stride < C_DIM; stride <<= 1) {
            float prev = (vi >= stride && vi < C_DIM) ? smem_gcum[vi - stride] : 0.f;
            __syncthreads();
            if (vi < C_DIM) { lgc += prev; smem_gcum[vi] = lgc; }
            __syncthreads();
        }
        if (vi < C_DIM) smem_gcum[vi] = expf(smem_gcum[vi]);
        __syncthreads(); /* gcum in final exp form */

        /* Step B: load keys - smem_k[c][ki] = k[b,h,chunk+c,ki] */
        for (int idx = vi; idx < C_DIM * K_DIM; idx += V_DIM) {
            int ci = idx / K_DIM, ki = idx % K_DIM;
            smem_k[ci][ki] = k[QK_IDX(b, h, chunk + ci, ki)];
        }
        __syncthreads(); /* (B) smem_k ready */

        /* Step C: build WY lower triangle (ref_prefill lines 113-118)
         *   WY[t][s] = -G(t,s)*beta[t]*m[t]*(k[t].k[s])  for t > s
         *   G(t,s)   = gcum[t] / gcum[s]  (ref line 108)
         * Diagonal starts at 0 (identity added after substitution in step D.5).
         * Work distribution: V_DIM threads cover n_lower = C*(C-1)/2 entries.
         */
        {
            const int n_lower = C_DIM * (C_DIM - 1) / 2;
            for (int idx = vi; idx < n_lower; idx += V_DIM) {
                /* map flat lower-triangle index -> (ti > si), inverse of idx = ti*(ti-1)/2 + si */
                int ti = (int)((1.f + sqrtf(1.f + 8.f * (float)idx)) * 0.5f);
                int si = idx - ti * (ti - 1) / 2;

                float G_ts = smem_gcum[ti] / smem_gcum[si];
                float b_ti = smem_beta[ti] * smem_mask_s[ti];
                float dot  = 0.f;
                for (int ki = 0; ki < K_DIM; ki++) {
                    dot += b_ti
                         * __bfloat162float(smem_k[ti][ki])
                         * __bfloat162float(smem_k[si][ki]);
                }
                smem_WY[ti][si] = -G_ts * dot;
            }
            /* diagonal = 0, strict upper = 0 (identity added after sub in step D.5) */
            for (int idx = vi; idx < C_DIM * C_DIM; idx += V_DIM) {
                int ti = idx / C_DIM, si = idx % C_DIM;
                if (si >= ti) smem_WY[ti][si] = 0.f;
            }
        }
        __syncthreads(); /* (C) smem_WY initial fill done */

        /* Step D: forward substitution -> (I - L)^{-1} - I (ref lines 114-117)
         *   For i = 1..C-1: WY[i, :i] += sum_{j<i} WY[i,j]*WY[j, :i]
         * WY[j][j]=0 throughout; thread vi updates column vi of row i.
         * Each iteration needs prior rows stable -> two syncs per row.
         */
        for (int i = 1; i < C_DIM; i++) {
            float acc = 0.f;
            if (vi < i) {
                for (int j = 0; j < i; j++)
                    acc += smem_WY[i][j] * smem_WY[j][vi];
            }
            __syncthreads(); /* all threads have read WY[j][vi] for this row */
            if (vi < i) smem_WY[i][vi] += acc;
            __syncthreads(); /* row i committed before row i+1 reads it */
        }

        /* Step D.5: add identity -> WY = (I - L)^{-1} (ref line 118: WY += eye) */
        for (int idx = vi; idx < C_DIM * C_DIM; idx += V_DIM) {
            int ti = idx / C_DIM, si = idx % C_DIM;
            if (si == ti) smem_WY[ti][si] += 1.f;
        }
        __syncthreads(); /* (D.5) smem_WY is now (I-L)^{-1} with diagonal=1 */

        /* Step E: prepare WMMA inputs
         *   (E1) WY fp32 -> bf16 -> smem_WY_bf16   (Matrix A for dV WMMA)
         *   (E2) v_beta bf16 -> smem_vbeta          (Matrix B for dV WMMA)
         *        v_beta[s][vv] = beta[s]*m[s]*v[b,h,chunk+s,vv]
         */
        /* (E1) WY fp32 -> bf16 */
        for (int idx = vi; idx < C_DIM * C_DIM; idx += V_DIM) {
            int ti = idx / C_DIM, si = idx % C_DIM;
            smem_WY_bf16[ti][si] = __float2bfloat16(smem_WY[ti][si]);
        }
        /* (E2) v_beta into smem_vbeta */
        for (int idx = vi; idx < C_DIM * V_DIM; idx += V_DIM) {
            int ci = idx / V_DIM, vv = idx % V_DIM;
            float vval = __bfloat162float(v[V_IDX(b, h, chunk + ci, vv)]);
            smem_vbeta[ci][vv] = __float2bfloat16(
                smem_beta[ci] * smem_mask_s[ci] * vval);
        }
        __syncthreads(); /* (E) smem_WY_bf16 and smem_vbeta fully written */

        /* Fragment register arrays shared between steps F and I.
         * UA = k-tile (A-frag) unroll; UC = col-tile (acc/B-frag) unroll.
         * Declared at this scope so the compiler reuses the same register
         * slots for both blocks; changing UA or UC adjusts both in lockstep.
         */
        constexpr int UA       = 2;           /* k_tile unroll */
        constexpr int UC       = 2;           /* col-tile unroll */
        const int     warp_row = warp_id / 2; /* 0..3; same in F and I */

        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag[UA];
        fragment<accumulator, 16, 16, 16, float>                  acc[UC];

        /* Step F: WMMA - dV_intra = WY_bf16 @ v_beta_bf16
         * ref_prefill line 121: new_v_minus_old_v_intra = WY @ v_beta
         * Shape: [C=64,C=64] x [C=64,V=256] -> [C=64,V=256] fp32 accum
         * Tile layout: warp_row (0..3), col_base = (warp_id%2)*8
         *   A = smem_WY_bf16 [C_DIM,C_DIM] row_major
         *   B = smem_vbeta   [C_DIM,V_DIM] row_major
         *   C = smem_dV      [C_DIM,V_DIM] fp32
         * col_tile unrolled UC (acc[UC], vb[UC]); k_tile unrolled UA (a_frag[UA]).
         * Peak 6 live frags: a_frag[UA]+vb[UC]+acc[UC] = 4+4+4+4+8+8 = 32 regs.
         */
        {
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> vb[UC];
            const int col_base = (warp_id % 2) * 8; /* 0 or 8 */
            #pragma unroll 1
            for (int col_tile = col_base; col_tile < col_base + 8; col_tile += UC) {
                #pragma unroll
                for (int j = 0; j < UC; j++) fill_fragment(acc[j], 0.f);

                #pragma unroll 1
                for (int k_tile = 0; k_tile < C_DIM / 16; k_tile += UA) {
                    #pragma unroll
                    for (int i = 0; i < UA; i++)
                        load_matrix_sync(a_frag[i],
                            &smem_WY_bf16[warp_row * 16][(k_tile + i) * 16],
                            C_DIM);
                    #pragma unroll
                    for (int i = 0; i < UA; i++) {
                        #pragma unroll
                        for (int j = 0; j < UC; j++)
                            load_matrix_sync(vb[j],
                                &smem_vbeta[(k_tile + i) * 16][(col_tile + j) * 16],
                                V_DIM);
                        #pragma unroll
                        for (int j = 0; j < UC; j++)
                            mma_sync(acc[j], a_frag[i], vb[j], acc[j]);
                    }
                }
                #pragma unroll
                for (int j = 0; j < UC; j++)
                    store_matrix_sync(
                        &smem_dV[warp_row * 16][(col_tile + j) * 16],
                        acc[j], V_DIM, mem_row_major);
            }
        }
        __syncthreads(); /* (F) smem_dV = WY @ v_beta written */

        /* Step G: cross-chunk S correction (ref_prefill lines 125, 142-145)
         *   k_bgcum = WY @ (k_beta * gcum)  [C,K]  (ref line 125)
         *   old_v   = k_bgcum @ S           [C,V]  (ref line 142)
         *   dV      = dV_intra - old_v      [C,V]  (ref line 145)
         * Inlined (no k_bgcum materialization):
         *   kbgS[s]        = beta[s]*m[s]*gcum[s] * dot(k[s,:], S_reg)
         *   smem_dV[i][vi] -= sum_s WY[i][s] * kbgS[s]
         */

        /* First pass: compute kbgS[s] for all s into registers */
        float kbgS[C_DIM];
        for (int s = 0; s < C_DIM; s++) {
            float coeff = smem_beta[s] * smem_mask_s[s] * smem_gcum[s];
            float dot   = 0.f;
            for (int ki = 0; ki < K_DIM; ki++)
                dot += coeff * __bfloat162float(smem_k[s][ki]) * S_reg[ki];
            kbgS[s] = dot;
        }
        /* Second pass: subtract WY @ kbgS from smem_dV column-wise */
        for (int i = 0; i < C_DIM; i++) {
            float old_v = 0.f;
            for (int s = 0; s < C_DIM; s++)
                old_v += smem_WY[i][s] * kbgS[s];
            smem_dV[i][vi] -= old_v;
        }
        __syncthreads(); /* (G) smem_dV now holds full dV = new_v - old_v */

        /* Step H: load q_scaled (ref_prefill line 84: q *= scale)
         * Store q*scale into smem_q_bf16 (vbeta phase is done).
         * Used in both step I (WMMA attn) and step K (o_cross).
         */
        for (int idx = vi; idx < C_DIM * K_DIM; idx += V_DIM) {
            int ci = idx / K_DIM, ki = idx % K_DIM;
            float qval = __bfloat162float(q[QK_IDX(b, h, chunk + ci, ki)]);
            smem_q_bf16[ci][ki] = __float2bfloat16(qval * scale);
        }
        __syncthreads(); /* (H) smem_q_bf16 ready; safe to overwrite smem_WY with attn */

        /* Step I: WMMA - attn = q_scaled @ k^T (ref_prefill line 151)
         * Shape: [C=64,K=128] x [K=128,C=64] -> [C=64,C=64] fp32 accum
         * Written into smem_WY (WY no longer needed after step G).
         * Tile layout: warp_row, col[UC] = {warp_id%2, warp_id%2+2}.
         * k stored row-major [C,K]; loaded col_major to compute k^T.
         * k_tile unrolled UA (a_frag[UA] = q frags); UC col frags (kb[UC]).
         * Reuses a_frag[UA] and acc[UC] from step F (same register slots).
         * Peak 6 live frags: a_frag[UA]+kb[UC]+acc[UC] = 4+4+4+4+8+8 = 32 regs.
         */
        {
            const int col[UC] = { warp_id % 2, warp_id % 2 + 2 }; /* {0,2} or {1,3} */
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> kb[UC];
            #pragma unroll
            for (int j = 0; j < UC; j++) fill_fragment(acc[j], 0.f);

            #pragma unroll 1
            for (int k_tile = 0; k_tile < K_DIM / 16; k_tile += UA) {
                #pragma unroll
                for (int i = 0; i < UA; i++)
                    load_matrix_sync(a_frag[i],
                        smem_q_bf16[warp_row * 16] + (k_tile + i) * 16,
                        K_DIM);
                #pragma unroll
                for (int i = 0; i < UA; i++) {
                    #pragma unroll
                    for (int j = 0; j < UC; j++)
                        load_matrix_sync(kb[j],
                            smem_k[col[j] * 16] + (k_tile + i) * 16,
                            K_DIM);
                    #pragma unroll
                    for (int j = 0; j < UC; j++)
                        mma_sync(acc[j], a_frag[i], kb[j], acc[j]);
                }
            }
            __syncthreads(); /* all warps done accumulating; safe to write smem_WY */
            #pragma unroll
            for (int j = 0; j < UC; j++)
                store_matrix_sync(
                    &smem_WY[warp_row * 16][col[j] * 16],
                    acc[j], C_DIM, mem_row_major);
        }
        __syncthreads(); /* (I) smem_WY = raw q@k^T fully written */

        /* Step J: causal gate (ref_prefill lines 108, 151)
         *   attn[c][s] *= gcum[c]/gcum[s]  for s <= c
         *   attn[c][s] = 0                  for s > c  (causal mask)
         */
        for (int idx = vi; idx < C_DIM * C_DIM; idx += V_DIM) {
            int c = idx / C_DIM, s = idx % C_DIM;
            if (s <= c)
                smem_WY[c][s] *= smem_gcum[c] / smem_gcum[s];
            else
                smem_WY[c][s] = 0.f;
        }
        __syncthreads(); /* (J) causal-gated attn in smem_WY */

        /* Step K: output (ref_prefill lines 150-152, 163)
         *   o_cross[c] = gcum[c] * dot(q_scaled[c,:], S_reg)  (ref line 150)
         *   o_intra[c] = dot(attn[c,:], dV[:,vi])              (ref line 152)
         *   out[b,h,chunk+c,vi] = mask[c] * (o_cross[c] + o_intra[c])
         * smem_q_bf16, smem_WY, smem_dV all stable from prior syncs.
         */
        for (int c = 0; c < C_DIM; c++) {
            int   t_abs  = chunk + c;
            float m_c    = smem_mask_s[c];
            float gcum_c = smem_gcum[c];

            float o_cross = 0.f;
            for (int ki = 0; ki < K_DIM; ki++)
                o_cross += __bfloat162float(smem_q_bf16[c][ki]) * gcum_c * S_reg[ki];

            /* intra-chunk: attn[c,:] @ dV[:,vi] (upper triangle zeroed in step J) */
            float o_intra = 0.f;
            for (int s = 0; s <= c; s++)
                o_intra += smem_WY[c][s] * smem_dV[s][vi];

            out[V_IDX(b, h, t_abs, vi)] = __float2bfloat16(m_c * (o_cross + o_intra));
        }

        /* Step L: state update (ref_prefill lines 155-157)
         *   S_new[ki][vi] = gcum[C-1]*S[ki][vi]
         *                 + sum_c k[c][ki]*(gcum[C-1]/gcum[c])*dV[c][vi]
         * Reads: smem_k, smem_dV[c][vi], smem_gcum (all stable).
         * Writes: S_reg (registers, no smem contention).
         */
        {
            float last_gcum = smem_gcum[C_DIM - 1];

            for (int ki = 0; ki < K_DIM; ki++)
                S_reg[ki] *= last_gcum;

            /* accumulate rank-1 updates from each chunk position */
            for (int c = 0; c < C_DIM; c++) {
                float pos_decay = last_gcum / smem_gcum[c];
                float dv_c      = smem_dV[c][vi];
                for (int ki = 0; ki < K_DIM; ki++)
                    S_reg[ki] += __bfloat162float(smem_k[c][ki]) * pos_decay * dv_c;
            }
        }

        __syncthreads(); /* end-of-chunk barrier before next iteration writes smem */
    } /* end chunk loop */

    /* Write final recurrent state */
    for (int ki = 0; ki < K_DIM; ki++)
        state_out[S_IDX(b, h, ki, vi)] = S_reg[ki];
}

/* ------------------------------------------------------------------ */
/* Host launcher                                                        */
/* ------------------------------------------------------------------ */
std::tuple<torch::Tensor, torch::Tensor> gdn_prefill(
    torch::Tensor q,          // [B,H,T,K]  bf16
    torch::Tensor k,          // [B,H,T,K]  bf16
    torch::Tensor v,          // [B,H,T,V]  bf16
    torch::Tensor A_log,      // [H]        fp32
    torch::Tensor a,          // [B,H,T]    bf16
    float         dt_bias,
    torch::Tensor b_logits,   // [B,H,T]    bf16
    torch::Tensor mask,       // [B,T]      fp32
    torch::Tensor state_in,   // [B,H,K,V]  fp32 (pass empty tensor for zero init)
    float         scale
) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.dtype() == torch::kBFloat16,
                "q: contiguous bfloat16 CUDA required");
    TORCH_CHECK(k.is_cuda() && k.is_contiguous() && k.dtype() == torch::kBFloat16,
                "k: contiguous bfloat16 CUDA required");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.dtype() == torch::kBFloat16,
                "v: contiguous bfloat16 CUDA required");
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dtype() == torch::kBFloat16,
                "a: contiguous bfloat16 CUDA required");
    TORCH_CHECK(b_logits.is_cuda() && b_logits.is_contiguous() &&
                b_logits.dtype() == torch::kBFloat16,
                "b_logits: contiguous bfloat16 CUDA required");

    const int B = q.size(0);
    const int T = q.size(2);
    const int N = T / C_DIM;

    TORCH_CHECK(T % C_DIM == 0,  "T must be divisible by C_DIM=", C_DIM);
    TORCH_CHECK(q.size(1) == H_DIM, "H must be ", H_DIM);
    TORCH_CHECK(q.size(3) == K_DIM, "K must be ", K_DIM);
    TORCH_CHECK(v.size(3) == V_DIM, "V must be ", V_DIM);

    auto opts_f32  = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());

    auto out       = torch::empty({B, H_DIM, T, V_DIM}, opts_bf16);
    auto state_out = torch::empty({B, H_DIM, K_DIM, V_DIM}, opts_f32);

    const float* state_in_ptr = (state_in.numel() > 0)
        ? state_in.data_ptr<float>() : nullptr;

    /* Dynamic shared memory size - must match the layout defined in the kernel. */
    constexpr size_t smem_bytes =
        (size_t)C_DIM * K_DIM * sizeof(__nv_bfloat16) +  // smem_k       16384
        (size_t)C_DIM          * sizeof(float)         +  // smem_gcum      256
        (size_t)C_DIM          * sizeof(float)         +  // smem_beta       256
        (size_t)C_DIM          * sizeof(float)         +  // smem_mask_s     256
        (size_t)C_DIM * C_DIM  * sizeof(float)         +  // smem_WY       16384
        (size_t)C_DIM * C_DIM  * sizeof(__nv_bfloat16) +  // smem_WY_bf16   8192
        (size_t)C_DIM * V_DIM  * sizeof(__nv_bfloat16) +  // smem_vq       32768
        (size_t)C_DIM * V_DIM  * sizeof(float);           // smem_dV       65536
    /* = 140032 bytes (~136.75 KB); exceeds 48 KB default, attribute required */

    TORCH_CHECK(
        cudaFuncSetAttribute(gdn_prefill_v2_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes) == cudaSuccess,
        "cudaFuncSetAttribute failed - GPU may not support ", smem_bytes, " B of shared memory");

    dim3 grid(B * H_DIM);
    dim3 block(V_DIM);

    gdn_prefill_v2_kernel<<<grid, block, smem_bytes>>>(
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

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "gdn_prefill_v2_kernel launch failed");

    return {out, state_out};
}

/* ------------------------------------------------------------------ */
/* Python bindings                                                      */
/* ------------------------------------------------------------------ */
PYBIND11_MODULE(gdn_prefill, m) {
    m.doc() = "GDN prefill v2 - fused kernel derived from ref_prefill";

    m.def("prefill", &gdn_prefill,
        "Args: q,k [B,H,T,K] bf16 | v [B,H,T,V] bf16 | A_log [H] fp32 | "
        "a,b_logits [B,H,T] bf16 | dt_bias float | mask [B,T] fp32 | "
        "state_in [B,H,K,V] fp32 (empty=zeros) | scale float\n"
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
