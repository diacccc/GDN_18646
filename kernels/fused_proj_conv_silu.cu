/*
 * fused_proj_conv_silu.cu
 * Fused Linear Projection + Causal Depthwise Conv1D + SiLU kernel.
 * GEMM phase accelerated with WMMA Tensor Core instructions (FP16×FP16→FP32).
 *
 * For each Q/K/V path:
 *   output[b,t,c] = SiLU( sum_{j=0}^{3} conv_w[c,j] * sum_d W[c,d] * input[b, t-3+j, d] )
 *
 * Fusion keeps the GEMM output in shared memory, eliminating the
 * [B, T, D_out] global memory round-trip between projection and conv.
 *
 * Block tile: [TILE_T × TILE_C] = [64 × 16] output elements.
 * Grid:       (ceil(T/TILE_T), D_out/TILE_C, B)
 * Threads:    160  (WARPS=5 warps × 32 threads)
 *
 * Phase 1 (GEMM via WMMA):
 *   GEMM_T = TILE_T + CONV_K - 1 = 67 rows projected (causal boundary).
 *   Padded to GEMM_T_PAD = 80 (5 × 16) so each of the 5 warps owns one
 *   contiguous 16-row M-tile of the WMMA 16×16×16 fragment triple.
 *   For each BK=16 inner-dim strip:
 *     1. All threads cooperatively load __half tiles into smem_A_fp16 and smem_B_fp16.
 *     2. Each warp calls wmma::mma_sync (FP16 in, FP32 acc).
 *   After the K loop each warp stores its fp32 C-fragment to smem_proj[GEMM_T_PAD, TILE_C].
 *
 * Phase 2 (Conv+SiLU):
 *   Same as the scalar version. Reads smem_proj[(t_local+j)*TILE_C + c_local].
 *
 * Shared-memory budget:
 *   smem_proj   (fp32) : GEMM_T_PAD × TILE_C × 4 B = 80 × 16 × 4 = 5120 B
 *   smem_cw     (fp32) : TILE_C × CONV_K × 4 B      = 16 ×  4 × 4 =  256 B
 *   smem_A_fp16 (fp16) : GEMM_T_PAD × BK × 2 B      = 80 × 16 × 2 = 2560 B
 *   smem_B_fp16 (fp16) : TILE_C × BK × 2 B           = 16 × 16 × 2 =  512 B
 *   total                                                            = 8448 B (~8.25 KB)
 *
 * WMMA fragment layout for the projection proj = input_tile × weight^T:
 *   frag_A [warp] : matrix_a, 16×16, fp16, row_major
 *                   ← smem_A_fp16[warp_row*BK :], leading_dim = BK
 *                   = input rows [warp_row, warp_row+16) for current K-strip
 *   frag_B        : matrix_b, 16×16, fp16, col_major
 *                   ← smem_B_fp16[0:], leading_dim = BK
 *                   smem_B stores weight_tile as [TILE_C, BK] = [N, K] row-major;
 *                   col_major reinterpretation reads element [k,n] from ptr[n*BK+k]
 *                   = weight_tile[n,k], giving the transposed weight slice  ✓
 *   frag_C [warp] : accumulator, 16×16, fp32, row_major
 *                   stored to smem_proj[warp_row*TILE_C:], leading_dim = TILE_C
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

using namespace nvcuda;

/* ── Compile-time constants ─────────────────────────────────────────────── */
#define TILE_T      64           /* output time-steps per block              */
#define TILE_C      16           /* output channels per block  (= WMMA N)   */
#define BK          16           /* inner-dimension strip      (= WMMA K)   */
#define CONV_K       4           /* causal depthwise conv kernel size        */
#define WARPS        5           /* one warp per 16-row M-tile               */
#define THREADS     (WARPS * 32) /* 160                                      */
/* GEMM_T = 67; pad to next multiple of 16 for WMMA tiling                  */
#define GEMM_T_PAD  80           /* ceil(67/16)*16 = 5*16                   */

/* ── Helpers ────────────────────────────────────────────────────────────── */
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

/* ── Main kernel ────────────────────────────────────────────────────────── */
__global__ void fused_proj_conv_silu_kernel(
    const float* __restrict__ input,   /* [B, T, D]        */
    const float* __restrict__ weight,  /* [D_out, D]       */
    const float* __restrict__ conv_w,  /* [D_out, CONV_K]  */
    float*       __restrict__ output,  /* [B, T, D_out]    */
    int T, int D, int D_out)
{
    /* ── Block / thread indices ──────────────────────────────────────── */
    const int tile_t  = blockIdx.x;
    const int tile_c  = blockIdx.y;
    const int b       = blockIdx.z;
    const int t_base  = tile_t * TILE_T;
    const int c_base  = tile_c * TILE_C;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;

    const int GEMM_T       = TILE_T + CONV_K - 1;  /* 67                */
    const int t_gemm_start = t_base - (CONV_K - 1);

    /* ── Shared-memory layout ────────────────────────────────────────── */
    /*  [ smem_proj fp32 | smem_cw fp32 | smem_A_fp16 | smem_B_fp16 ]   */
    extern __shared__ float smem[];
    float*  smem_proj   = smem;
    float*  smem_cw     = smem_proj + GEMM_T_PAD * TILE_C;
    __half* smem_A_fp16 = (__half*)(smem_cw + TILE_C * CONV_K);
    __half* smem_B_fp16 = smem_A_fp16 + GEMM_T_PAD * BK;

    /* ── Load conv weights (only TILE_C*CONV_K = 64 values) ─────────── */
    if (tid < TILE_C * CONV_K) {
        int c_local = tid / CONV_K;
        int j       = tid % CONV_K;
        smem_cw[c_local * CONV_K + j] = conv_w[(c_base + c_local) * CONV_K + j];
    }

    /* ── WMMA accumulator: each warp owns one 16×16 fp32 fragment ────── */
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C;
    wmma::fill_fragment(frag_C, 0.0f);

    const int warp_row = warp_id * 16;  /* first GEMM_T row for this warp */

    /* ── Phase 1: Tiled GEMM via WMMA  proj = input_tile × weight^T ─── */
    for (int bk = 0; bk < D; bk += BK) {

        /* -- Load smem_A_fp16[GEMM_T_PAD, BK] (1280 halves, 8 iters) -- */
        for (int i = tid; i < GEMM_T_PAD * BK; i += THREADS) {
            int gt    = i / BK;
            int dk    = i % BK;
            int abs_t = t_gemm_start + gt;
            float val = 0.0f;
            if (gt < GEMM_T && abs_t >= 0 && abs_t < T && (bk + dk) < D)
                val = input[(size_t)b * T * D + (size_t)abs_t * D + bk + dk];
            smem_A_fp16[gt * BK + dk] = __float2half(val);
        }

        /* -- Load smem_B_fp16[TILE_C, BK] (256 halves, 2 iters) -------- */
        for (int i = tid; i < TILE_C * BK; i += THREADS) {
            int c_local = i / BK;
            int dk      = i % BK;
            float val   = 0.0f;
            if ((bk + dk) < D)
                val = weight[(size_t)(c_base + c_local) * D + bk + dk];
            smem_B_fp16[c_local * BK + dk] = __float2half(val);
        }

        __syncthreads();

        /* -- Each warp: load fragments and accumulate ------------------- */
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag_A;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> frag_B;

        /* frag_A: rows [warp_row, warp_row+16), K=BK, lda=BK, row_major  */
        wmma::load_matrix_sync(frag_A, smem_A_fp16 + warp_row * BK, BK);

        /* frag_B: smem_B stored as [N=TILE_C, K=BK] row-major.
         * col_major interpretation reads [k,n] from ptr[n*lda+k]
         * = smem_B_fp16[n*BK+k] = weight_tile[n,k] → implements weight^T ✓ */
        wmma::load_matrix_sync(frag_B, smem_B_fp16, BK);

        wmma::mma_sync(frag_C, frag_A, frag_B, frag_C);

        __syncthreads();
    }

    /* ── Store WMMA result to smem_proj[GEMM_T_PAD, TILE_C] ─────────── */
    /* row_major store: element [m,n] → smem_proj[(warp_row+m)*TILE_C+n] */
    wmma::store_matrix_sync(smem_proj + warp_row * TILE_C, frag_C,
                            TILE_C, wmma::mem_row_major);

    __syncthreads();

    /* ── Phase 2: Causal Conv1D + SiLU ───────────────────────────────── */
    /* Total output elements = TILE_T*TILE_C = 1024; ceil(1024/160) = 7  */
    for (int e = 0; e < 7; e++) {
        int idx = tid + e * THREADS;
        if (idx >= TILE_T * TILE_C) break;
        int t_local = idx / TILE_C;
        int c_local = idx % TILE_C;

        float conv_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < CONV_K; j++) {
            /* smem_proj is [GEMM_T_PAD, TILE_C]: access [t_local+j, c_local] */
            conv_sum += smem_cw[c_local * CONV_K + j]
                      * smem_proj[(t_local + j) * TILE_C + c_local];
        }

        int abs_t = t_base + t_local;
        if (abs_t < T)
            output[(size_t)b * T * D_out + (size_t)abs_t * D_out + c_base + c_local]
                = silu_f(conv_sum);
    }
}

/* ── PyTorch wrapper ─────────────────────────────────────────────────────── */
torch::Tensor fused_proj_conv_silu(
    torch::Tensor input,   /* [B, T, D]       float32 */
    torch::Tensor weight,  /* [D_out, D]      float32 */
    torch::Tensor conv_w   /* [D_out, CONV_K] float32 */
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

    dim3 grid((T + TILE_T - 1) / TILE_T,
              D_out / TILE_C,
              B);
    dim3 block(THREADS);

    /* smem = fp32 region (proj + conv weights) + fp16 region (A + B tiles) */
    const int smem_bytes =
        (GEMM_T_PAD * TILE_C + TILE_C * CONV_K) * (int)sizeof(float) +
        (GEMM_T_PAD * BK     + TILE_C * BK)      * (int)sizeof(__half);

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
          "Fused Linear Projection + Causal Depthwise Conv1D + SiLU (Tensor Core GEMM)");
}
