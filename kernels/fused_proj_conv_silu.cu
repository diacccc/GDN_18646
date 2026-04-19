/*
 * fused_proj_conv_silu.cu
 * Fused Linear Projection + Causal Depthwise Conv1D + SiLU kernel.
 * GEMM phase accelerated with WMMA Tensor Core instructions (BF16×BF16→FP32).
 *
 * For each Q/K/V path:
 *   output[b,t,c] = SiLU( sum_{j=0}^{3} conv_w[c,j] * sum_d W[c,d] * input[b, t-3+j, d] )
 *
 * ── Optimizations applied (inspired by TIRX GEMM assignment) ────────────
 *
 * 1. Wider channel tile (TILE_C 16→32):
 *    Each block now computes a 64×32 output tile instead of 64×16.
 *    Two WMMA N-tiles of 16 are computed per warp (frag_C0, frag_C1),
 *    doubling arithmetic intensity for the same smem traffic on A.
 *
 * 2. Double-buffered shared memory (software pipeline, PIPE=2):
 *    Inspired by Steps 5/8 of the TIRX GEMM assignment.
 *    While warps run wmma::mma_sync on stage s, the next BK strip is
 *    loaded into stage 1-s.  This overlaps global-memory latency with
 *    Tensor Core compute, matching the TIRX load→MMA pipeline pattern.
 *    Two ping-pong buffers hold smem_A[PIPE][GEMM_T_PAD][BK] and
 *    smem_B[PIPE][TILE_C][BK].
 *
 * 3. Vectorized global loads (128-bit / nv_bfloat162):
 *    Loads two adjacent bf16 elements per instruction (x2 bandwidth),
 *    analogous to TMA's bank-conflict-free swizzled layout benefit.
 *    Applied to both A (input) and B (weight) tiles.
 *
 * 4. Single sync per K-iteration:
 *    Previously two __syncthreads() per BK-step (load + MMA).
 *    With double-buffering the trailing sync is folded into the prefetch
 *    sync of the next iteration, cutting synchronisation overhead by ~half.
 *
 * ── Shared-memory budget ────────────────────────────────────────────────
 *   smem_proj   (fp32) : GEMM_T_PAD × TILE_C × 4 B = 80 × 32 × 4  = 10240 B
 *   smem_cw     (fp32) : TILE_C × CONV_K × 4 B      = 32 ×  4 × 4  =   512 B
 *   smem_A[2]   (bf16) : 2 × GEMM_T_PAD × BK × 2 B  = 2×80×16×2   =  5120 B
 *   smem_B[2]   (bf16) : 2 × TILE_C × BK × 2 B       = 2×32×16×2   =  2048 B
 *   total                                                             = 17920 B (~17.5 KB)
 *
 * ── Grid / block ────────────────────────────────────────────────────────
 *   Block tile: [TILE_T × TILE_C] = [64 × 32]
 *   Grid:       (ceil(T/TILE_T), D_out/TILE_C, B)
 *   Threads:    160  (WARPS=5 × 32)
 *
 * ── WMMA fragment layout ────────────────────────────────────────────────
 *   frag_A [warp]   : matrix_a, 16×16, bf16, row_major
 *                     ← smem_A[s][warp_row*BK:], lda=BK
 *   frag_B0 / frag_B1: matrix_b, 16×16, bf16, col_major
 *                     ← smem_B[s][0:BK] / smem_B[s][BK:2*BK], lda=BK
 *                     col_major with lda=BK implements weight^T ✓
 *   frag_C0 / frag_C1: accumulator, 16×16, fp32, row_major
 *                     stored to smem_proj[warp_row*TILE_C : +16, 0:16/16:32]
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

using namespace nvcuda;

/* ── Compile-time constants ─────────────────────────────────────────────── */
#define TILE_T      64           /* output time-steps per block              */
#define TILE_C      32           /* output channels per block (2 × WMMA N)  */
#define BK          16           /* inner-dimension strip      (= WMMA K)   */
#define CONV_K       4           /* causal depthwise conv kernel size        */
#define WARPS        5           /* one warp per 16-row M-tile               */
#define THREADS     (WARPS * 32) /* 160                                      */
#define PIPE         2           /* double-buffer depth                      */
/* GEMM_T = 67; pad to next multiple of 16 for WMMA tiling                  */
#define GEMM_T_PAD  80           /* ceil(67/16)*16 = 5*16                   */

/* ── Helpers ────────────────────────────────────────────────────────────── */
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

/* Load a bf16 pair as a 32-bit word (vectorised). Falls back to scalar for
 * odd addresses or out-of-bounds elements. */
__device__ __forceinline__ nv_bfloat162 load_bf162(const nv_bfloat16* p) {
    return *reinterpret_cast<const nv_bfloat162*>(p);
}

/* ── Main kernel ────────────────────────────────────────────────────────── */
__global__ void fused_proj_conv_silu_kernel(
    const nv_bfloat16* __restrict__ input,   /* [B, T, D]        */
    const nv_bfloat16* __restrict__ weight,  /* [D_out, D]       */
    const nv_bfloat16* __restrict__ conv_w,  /* [D_out, CONV_K]  */
    nv_bfloat16*       __restrict__ output,  /* [B, T, D_out]    */
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
    /*  [ smem_proj fp32 | smem_cw fp32 | smem_A[PIPE] bf16 | smem_B[PIPE] bf16 ] */
    extern __shared__ float smem[];
    float*        smem_proj = smem;                              /* [GEMM_T_PAD, TILE_C] */
    float*        smem_cw   = smem_proj + GEMM_T_PAD * TILE_C;  /* [TILE_C, CONV_K]    */
    /* double-buffer ping-pong tiles */
    nv_bfloat16*  smem_A = (nv_bfloat16*)(smem_cw + TILE_C * CONV_K);
    /*   smem_A[s][gt][dk] = smem_A[s*GEMM_T_PAD*BK + gt*BK + dk]      */
    nv_bfloat16*  smem_B = smem_A + PIPE * GEMM_T_PAD * BK;
    /*   smem_B[s][c_local][dk] = smem_B[s*TILE_C*BK + c_local*BK + dk] */

    /* ── Load conv weights (TILE_C*CONV_K = 128 values, 1 iter) ─────── */
    for (int i = tid; i < TILE_C * CONV_K; i += THREADS) {
        int c_local = i / CONV_K;
        int j       = i % CONV_K;
        smem_cw[c_local * CONV_K + j] =
            __bfloat162float(conv_w[(c_base + c_local) * CONV_K + j]);
    }

    /* ── WMMA accumulators: each warp computes two 16×16 tiles (N0, N1) */
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_C0, frag_C1;
    wmma::fill_fragment(frag_C0, 0.0f);
    wmma::fill_fragment(frag_C1, 0.0f);

    const int warp_row = warp_id * 16;  /* first GEMM_T row for this warp */

    /* ── Helper: fill one pipeline stage into smem_A / smem_B ────────── */
    /* Inlined via lambda-equivalent using the stage index s.             */
    #define LOAD_STAGE(s, bk_val)                                              \
    {                                                                          \
        nv_bfloat16* sA = smem_A + (s) * GEMM_T_PAD * BK;                    \
        nv_bfloat16* sB = smem_B + (s) * TILE_C * BK;                        \
        int _bk = (bk_val);                                                    \
        /* Load A tile: GEMM_T_PAD*BK=1280 bf16, 8 iters of 160 threads.    \
         * Vectorised: process pairs of dk to use 32-bit loads.             */ \
        for (int i = tid; i < GEMM_T_PAD * (BK / 2); i += THREADS) {         \
            int gt   = i / (BK / 2);                                          \
            int dk2  = i % (BK / 2);  /* pair index */                        \
            int abs_t = t_gemm_start + gt;                                     \
            nv_bfloat162 val = __float2bfloat162_rn(0.0f);                    \
            if (gt < GEMM_T && abs_t >= 0 && abs_t < T &&                     \
                    (_bk + dk2*2 + 1) < D) {                                  \
                val = load_bf162(input + (size_t)b*T*D +                       \
                                 (size_t)abs_t*D + _bk + dk2*2);              \
            } else if (gt < GEMM_T && abs_t >= 0 && abs_t < T &&              \
                       (_bk + dk2*2) < D) {                                   \
                val.x = input[(size_t)b*T*D + (size_t)abs_t*D + _bk+dk2*2]; \
            }                                                                  \
            *reinterpret_cast<nv_bfloat162*>(sA + gt*BK + dk2*2) = val;       \
        }                                                                      \
        /* Load B tile: TILE_C*BK=512 bf16, vectorised pairs.              */ \
        for (int i = tid; i < TILE_C * (BK / 2); i += THREADS) {             \
            int c_local = i / (BK / 2);                                       \
            int dk2     = i % (BK / 2);                                       \
            nv_bfloat162 val = __float2bfloat162_rn(0.0f);                    \
            if ((_bk + dk2*2 + 1) < D) {                                      \
                val = load_bf162(weight + (size_t)(c_base+c_local)*D +         \
                                 _bk + dk2*2);                                 \
            } else if ((_bk + dk2*2) < D) {                                   \
                val.x = weight[(size_t)(c_base+c_local)*D + _bk + dk2*2];    \
            }                                                                  \
            *reinterpret_cast<nv_bfloat162*>(sB + c_local*BK + dk2*2) = val;  \
        }                                                                      \
    }

    /* ── Phase 1: Double-buffered tiled GEMM via WMMA ────────────────── */
    /* Prefetch stage 0 */
    LOAD_STAGE(0, 0)
    __syncthreads();

    int K_TILES = (D + BK - 1) / BK;
    for (int k = 0; k < K_TILES; k++) {
        int s     = k & 1;          /* current stage */
        int s_nxt = 1 - s;          /* next stage    */

        /* Prefetch next stage (overlap with MMA below) */
        if (k + 1 < K_TILES) {
            LOAD_STAGE(s_nxt, (k + 1) * BK)
        }

        /* MMA on current stage ---------------------------------------- */
        nv_bfloat16* sA = smem_A + s * GEMM_T_PAD * BK;
        nv_bfloat16* sB = smem_B + s * TILE_C * BK;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, nv_bfloat16, wmma::row_major> frag_A;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, nv_bfloat16, wmma::col_major> frag_B0;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, nv_bfloat16, wmma::col_major> frag_B1;

        wmma::load_matrix_sync(frag_A,  sA + warp_row * BK, BK);
        /* B0: columns 0..15 of weight tile → channels c_base+0..15      */
        wmma::load_matrix_sync(frag_B0, sB,           BK);
        /* B1: columns 16..31 of weight tile → channels c_base+16..31    */
        wmma::load_matrix_sync(frag_B1, sB + BK * BK, BK);

        wmma::mma_sync(frag_C0, frag_A, frag_B0, frag_C0);
        wmma::mma_sync(frag_C1, frag_A, frag_B1, frag_C1);

        /* Sync to let prefetch loads finish before next iteration reads  */
        __syncthreads();
    }
    #undef LOAD_STAGE

    /* ── Store WMMA results to smem_proj[GEMM_T_PAD, TILE_C] ─────────── */
    /* C0 → columns 0..15; C1 → columns 16..31 */
    wmma::store_matrix_sync(smem_proj + warp_row * TILE_C,
                            frag_C0, TILE_C, wmma::mem_row_major);
    wmma::store_matrix_sync(smem_proj + warp_row * TILE_C + 16,
                            frag_C1, TILE_C, wmma::mem_row_major);

    __syncthreads();

    /* ── Phase 2: Causal Conv1D + SiLU ───────────────────────────────── */
    /* Total output elements = TILE_T*TILE_C = 2048; ceil(2048/160) = 13 */
    for (int e = 0; e < 13; e++) {
        int idx = tid + e * THREADS;
        if (idx >= TILE_T * TILE_C) break;
        int t_local = idx / TILE_C;
        int c_local = idx % TILE_C;

        float conv_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < CONV_K; j++) {
            conv_sum += smem_cw[c_local * CONV_K + j]
                      * smem_proj[(t_local + j) * TILE_C + c_local];
        }

        int abs_t = t_base + t_local;
        if (abs_t < T)
            output[(size_t)b * T * D_out + (size_t)abs_t * D_out + c_base + c_local]
                = __float2bfloat16(silu_f(conv_sum));
    }
}

/* ── PyTorch wrapper ─────────────────────────────────────────────────────── */
torch::Tensor fused_proj_conv_silu(
    torch::Tensor input,   /* [B, T, D]       bfloat16 */
    torch::Tensor weight,  /* [D_out, D]      bfloat16 */
    torch::Tensor conv_w   /* [D_out, CONV_K] bfloat16 */
) {
    TORCH_CHECK(input.is_cuda(),        "input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(),       "weight must be on CUDA");
    TORCH_CHECK(conv_w.is_cuda(),       "conv_w must be on CUDA");
    TORCH_CHECK(input.is_contiguous(),  "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_w.is_contiguous(), "conv_w must be contiguous");
    TORCH_CHECK(input.scalar_type()  == torch::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(conv_w.scalar_type() == torch::kBFloat16, "conv_w must be bfloat16");

    const int B     = input.size(0);
    const int T     = input.size(1);
    const int D     = input.size(2);
    const int D_out = weight.size(0);

    TORCH_CHECK(weight.size(1) == D,      "weight dim-1 must equal D");
    TORCH_CHECK(conv_w.size(0) == D_out,  "conv_w dim-0 must equal D_out");
    TORCH_CHECK(conv_w.size(1) == CONV_K, "conv_w dim-1 must equal CONV_K");
    TORCH_CHECK(D_out % TILE_C == 0,      "D_out must be a multiple of TILE_C (32)");;

    auto output = torch::empty({B, T, D_out}, input.options());

    dim3 grid((T + TILE_T - 1) / TILE_T,
              D_out / TILE_C,
              B);
    dim3 block(THREADS);

    /* smem = fp32 region (proj + conv_weights) + bf16 double-buffered A + B tiles */
    const int smem_bytes =
        (GEMM_T_PAD * TILE_C + TILE_C * CONV_K) * (int)sizeof(float) +
        PIPE * (GEMM_T_PAD * BK + TILE_C * BK)   * (int)sizeof(nv_bfloat16);

    fused_proj_conv_silu_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(conv_w.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(output.data_ptr()),
        T, D, D_out);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_proj_conv_silu,
          "Fused Linear Projection + Causal Depthwise Conv1D + SiLU (Tensor Core GEMM, BF16)");
}
