/*
 * fused_proj_conv_silu.cu
 * Fused Linear Projection + Causal Depthwise Conv1D + SiLU kernel.
 *
 * For each Q/K/V path:
 *   output[b,t,c] = SiLU( sum_{j=0}^{3} conv_w[c,j] * sum_d W[c,d] * input[b, t-3+j, d] )
 *
 * Fusion keeps the GEMM output in shared memory, eliminating the
 * [B, T, D_out] global memory round-trip between projection and conv.
 *
 * Block tile: [TILE_T x TILE_C] = [64 x 16] output elements.
 * Grid:       (ceil(T/TILE_T), D_out/TILE_C, B)
 * Threads:    128 (4 warps)
 *
 * Phase 1 (GEMM):  Compute projection for (TILE_T + CONV_K - 1) = 67 time
 *                   steps (3 extra for the causal boundary).  Accumulate in
 *                   registers, then flush to smem_proj[TILE_C][67].
 * Phase 2 (Conv+SiLU): Each thread reads a 4-element causal window from
 *                   smem_proj, dots with conv weights, applies SiLU, and
 *                   writes the final output to global memory.
 *
 * Shared-memory budget (float32):
 *   smem_proj  : TILE_C * GEMM_T            = 16 * 67  = 1072
 *   smem_A     : GEMM_T * BK                = 67 * 16  = 1072   (input tile)
 *   smem_B     : TILE_C * BK                = 16 * 16  = 256    (weight tile)
 *   smem_cw    : TILE_C * CONV_K            = 16 * 4   = 64     (conv weights)
 *   total      = 2464 floats = 9856 bytes (~10 KB, well within A100 48 KB limit)
 */

#include <cuda_runtime.h>
#include <torch/extension.h>

/* ── Compile-time constants ─────────────────────────────────────────────── */
#define TILE_T   64          /* output time-steps per block                 */
#define TILE_C   16          /* output channels per block                   */
#define BK       16          /* inner-dimension strip width for GEMM        */
#define CONV_K   4           /* causal depthwise conv kernel size           */
#define THREADS  128         /* threads per block (4 warps)                 */

/* ── Helpers ────────────────────────────────────────────────────────────── */
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

/* ── Main kernel ────────────────────────────────────────────────────────── */
__global__ void fused_proj_conv_silu_kernel(
    const float* __restrict__ input,      /* [B, T, D]           */
    const float* __restrict__ weight,     /* [D_out, D]          */
    const float* __restrict__ conv_w,     /* [D_out, CONV_K]     */
    float*       __restrict__ output,     /* [B, T, D_out]       */
    int T, int D, int D_out)
{
    /* ── Block / thread indices ──────────────────────────────────────── */
    const int tile_t = blockIdx.x;          /* tile index along T      */
    const int tile_c = blockIdx.y;          /* tile index along D_out  */
    const int b      = blockIdx.z;          /* batch index             */

    const int t_base = tile_t * TILE_T;     /* first output time-step  */
    const int c_base = tile_c * TILE_C;     /* first output channel    */
    const int tid    = threadIdx.x;

    const int GEMM_T = TILE_T + CONV_K - 1; /* 67 rows of projection  */

    /* ── Shared-memory layout ────────────────────────────────────────── */
    extern __shared__ float smem[];
    float* smem_proj = smem;                                        /* [TILE_C][GEMM_T]  */
    float* smem_A    = smem_proj + TILE_C * GEMM_T;                 /* [GEMM_T][BK]      */
    float* smem_B    = smem_A    + GEMM_T * BK;                     /* [TILE_C][BK]      */
    float* smem_cw   = smem_B    + TILE_C * BK;                     /* [TILE_C][CONV_K]  */

    /* ── Load conv weights (only 64 values, first 64 threads) ────── */
    if (tid < TILE_C * CONV_K) {
        int c_local = tid / CONV_K;
        int j       = tid % CONV_K;
        smem_cw[c_local * CONV_K + j] = conv_w[(c_base + c_local) * CONV_K + j];
    }

    /* ── Register accumulators for the GEMM (max 9 per thread) ────── */
    /* Total elements = GEMM_T * TILE_C = 67 * 16 = 1072              */
    /* ceil(1072 / 128) = 9                                            */
    #define GEMM_ELEMS 9
    float acc[GEMM_ELEMS];
    #pragma unroll
    for (int i = 0; i < GEMM_ELEMS; i++) acc[i] = 0.0f;

    /* first input time-step required for the causal boundary          */
    const int t_gemm_start = t_base - (CONV_K - 1);

    /* ── Phase 1: Tiled GEMM  proj = input @ weight^T ────────────── */
    for (int bk = 0; bk < D; bk += BK) {

        /* -- Cooperative load: input tile  smem_A[GEMM_T][BK] -------- */
        for (int i = tid; i < GEMM_T * BK; i += THREADS) {
            int gt = i / BK;
            int dk = i % BK;
            int abs_t = t_gemm_start + gt;
            if (abs_t >= 0 && abs_t < T && (bk + dk) < D)
                smem_A[gt * BK + dk] =
                    input[(size_t)b * T * D + (size_t)abs_t * D + bk + dk];
            else
                smem_A[gt * BK + dk] = 0.0f;
        }

        /* -- Cooperative load: weight tile smem_B[TILE_C][BK] -------- */
        for (int i = tid; i < TILE_C * BK; i += THREADS) {
            int c_local = i / BK;
            int dk      = i % BK;
            if ((bk + dk) < D)
                smem_B[c_local * BK + dk] =
                    weight[(size_t)(c_base + c_local) * D + bk + dk];
            else
                smem_B[c_local * BK + dk] = 0.0f;
        }

        __syncthreads();

        /* -- Accumulate  acc[e] += dot(smem_A[gt,:], smem_B[c,:]) ---- */
        for (int e = 0; e < GEMM_ELEMS; e++) {
            int idx = tid + e * THREADS;
            if (idx >= GEMM_T * TILE_C) break;
            int gt      = idx / TILE_C;
            int c_local = idx % TILE_C;

            float sum = 0.0f;
            #pragma unroll
            for (int dk = 0; dk < BK; dk++) {
                sum += smem_A[gt * BK + dk] * smem_B[c_local * BK + dk];
            }
            acc[e] += sum;
        }

        __syncthreads();   /* safe to overwrite smem_A / smem_B next iter */
    }

    /* ── Flush GEMM results to smem_proj[TILE_C][GEMM_T] ─────────── */
    for (int e = 0; e < GEMM_ELEMS; e++) {
        int idx = tid + e * THREADS;
        if (idx >= GEMM_T * TILE_C) break;
        int gt      = idx / TILE_C;
        int c_local = idx % TILE_C;
        smem_proj[c_local * GEMM_T + gt] = acc[e];
    }

    __syncthreads();   /* smem_proj visible to all threads for conv phase */

    /* ── Phase 2: Causal Conv1D + SiLU ───────────────────────────── */
    /* Each thread handles TILE_T * TILE_C / THREADS = 64*16/128 = 8  */
    #define OUT_ELEMS 8
    for (int e = 0; e < OUT_ELEMS; e++) {
        int idx     = tid + e * THREADS;
        int t_local = idx / TILE_C;
        int c_local = idx % TILE_C;

        /* 4-tap dot with causal window from smem_proj */
        float conv_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < CONV_K; j++) {
            conv_sum += smem_cw[c_local * CONV_K + j]
                      * smem_proj[c_local * GEMM_T + t_local + j];
        }

        float activated = silu_f(conv_sum);

        int abs_t = t_base + t_local;
        if (abs_t < T) {
            output[(size_t)b * T * D_out + (size_t)abs_t * D_out + c_base + c_local]
                = activated;
        }
    }
}

/* ── PyTorch wrapper ─────────────────────────────────────────────────────── */
torch::Tensor fused_proj_conv_silu(
    torch::Tensor input,      /* [B, T, D]          float32 */
    torch::Tensor weight,     /* [D_out, D]         float32 */
    torch::Tensor conv_w      /* [D_out, CONV_K]    float32 */
) {
    TORCH_CHECK(input.is_cuda(),        "input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(),       "weight must be on CUDA");
    TORCH_CHECK(conv_w.is_cuda(),       "conv_w must be on CUDA");
    TORCH_CHECK(input.is_contiguous(),  "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_w.is_contiguous(), "conv_w must be contiguous");

    const int B     = input.size(0);
    const int T     = input.size(1);
    const int D     = input.size(2);
    const int D_out = weight.size(0);

    TORCH_CHECK(weight.size(1) == D,      "weight dim-1 must equal D");
    TORCH_CHECK(conv_w.size(0) == D_out,  "conv_w dim-0 must equal D_out");
    TORCH_CHECK(conv_w.size(1) == CONV_K, "conv_w dim-1 must equal CONV_K");
    TORCH_CHECK(D_out % TILE_C == 0,      "D_out must be a multiple of TILE_C (16)");

    auto output = torch::empty({B, T, D_out}, input.options());

    const int GEMM_T = TILE_T + CONV_K - 1;

    dim3 grid((T + TILE_T - 1) / TILE_T,   /* tiles along T     */
              D_out / TILE_C,                /* tiles along D_out */
              B);                             /* batch             */
    dim3 block(THREADS);

    const int smem_bytes =
        (TILE_C * GEMM_T + GEMM_T * BK + TILE_C * BK + TILE_C * CONV_K)
        * (int)sizeof(float);

    fused_proj_conv_silu_kernel<<<grid, block, smem_bytes>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_w.data_ptr<float>(),
        output.data_ptr<float>(),
        T, D, D_out);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_proj_conv_silu,
          "Fused Linear Projection + Causal Depthwise Conv1D + SiLU");
}
