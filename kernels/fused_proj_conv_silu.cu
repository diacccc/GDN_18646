/*
 * fused_proj_conv_silu.cu — 4-warp non-specialized (reverts warp-spec)
 *
 * For each Q/K/V path:
 *   output[b,t,c] = SiLU( sum_{j=0..3} conv_w[c,j] * sum_d W[c,d] * input[b, t-3+j, d] )
 *
 * ── Changes vs. the 1P+3C warp-specialized version ─────────────────────
 * 1. Warp specialization removed. All 4 warps cooperatively issue
 *    cp.async *and* perform MMA. On A100, 1 producer warp is severely
 *    underutilized (~14 cycles of cp.async issue per K-stage vs. ~128
 *    cycles of mma.sync in consumers) — the 25% tensor-core loss from
 *    giving up a warp isn't offset by any meaningful latency-hiding gain
 *    on plain cp.async hardware.
 *
 * 2. TILE_T = 125, so GEMM_T = 128 = 8 × WMMA_M.  Each warp computes
 *    M_FRAGS=2 M-tiles (32 rows) for 33% higher arithmetic intensity.
 *    ~48% fewer blocks vs. TILE_T=61, less barrier/launch overhead.
 *
 * 3. Single __syncthreads per K-iter, placed *after* wait_group and
 *    *before* MMA.  Loop order is:
 *       wait_group<PIPE-2>
 *       __syncthreads          <- visibility + ensures prev-iter MMA done
 *       MMA on stage k%PIPE
 *       issue stage (k+PIPE-1)%PIPE (if in range)
 *       cp_async_commit
 *    Because the issue is written *after* MMA within the same iter, and
 *    the next iter's __syncthreads gates every warp before the next
 *    issue can observe the next-iter commit, there is no same-slot
 *    read/write race between MMA and the producer path.
 *
 * ── Tile geometry ───────────────────────────────────────────────────────
 *   TILE_T     = 125;  TILE_C = 64 ;  BK = 32
 *   GEMM_T_PAD = 128;  PIPE   =  3  ;  M_FRAGS = 2
 *   SMEM_LDA   = SMEM_LDB = 40 bf16 (BK + 8 pad -> conflict-free ldmatrix)
 *
 * ── Shared memory (overlaid: pipeline bufs ↔ projection buffer) ────────
 *   smem_cw       : 64 ×  4 × 4                  =  1024 B  (always live)
 *   UNION {
 *     smem_A[3]   : 3 × 128 × 40 × 2            = 30720 B  (GEMM phase)
 *     smem_B[3]   : 3 ×  64 × 40 × 2            = 15360 B
 *   ─── OR ──────────────────────────────────────
 *     smem_proj   : 128 × 64 × 4                 = 32768 B  (conv phase)
 *   }                                    total  ≈ 47 KB
 *
 *   A tile: 128×32 bf16 -> 512 × 16-B chunks -> 4 passes × 128 threads.
 *   B tile:  64×32 bf16 -> 256 × 16-B chunks -> 2 passes × 128 threads.
 *
 * ── Requirements ────────────────────────────────────────────────────────
 *   • Compute Capability ≥ 8.0
 *   • D     % 32 == 0
 *   • D_out % 64 == 0
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

using namespace nvcuda;

/* ── Compile-time constants ─────────────────────────────────────────────── */
#define TILE_T            125
#define TILE_C             64
#define BK                 32
#define WMMA_M             16
#define WMMA_N             16
#define WMMA_K             16
#define CONV_K              4
#define WARPS               4
#define THREADS       (WARPS * 32)              /* 128        */
#define PIPE                3

#define GEMM_T        (TILE_T + CONV_K - 1)     /* 128        */
#define GEMM_T_PAD     GEMM_T                   /* 128 = 8·16 */
#define M_FRAGS       (GEMM_T_PAD / (WARPS * WMMA_M))  /* 2  */
#define N_FRAGS       (TILE_C / WMMA_N)         /* 4          */
#define K_STEPS       (BK / WMMA_K)             /* 2          */

#define SMEM_PAD       8
#define SMEM_LDA      (BK + SMEM_PAD)           /* 40 */
#define SMEM_LDB      (BK + SMEM_PAD)           /* 40 */

/* ── PTX helpers ────────────────────────────────────────────────────────── */
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__
void cp_async_16(void* smem_ptr, const void* gmem_ptr, int src_bytes) {
    unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_int), "l"(gmem_ptr), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

/* ── Main kernel ────────────────────────────────────────────────────────── */
__global__ __launch_bounds__(THREADS, 3)
void fused_proj_conv_silu_kernel(
    const nv_bfloat16* __restrict__ input,
    const nv_bfloat16* __restrict__ weight,
    const nv_bfloat16* __restrict__ conv_w,
    nv_bfloat16*       __restrict__ output,
    int T, int D, int D_out)
{
    const int tile_t       = blockIdx.x;
    const int tile_c       = blockIdx.y;
    const int b            = blockIdx.z;
    const int t_base       = tile_t * TILE_T;
    const int c_base       = tile_c * TILE_C;
    const int tid          = threadIdx.x;
    const int warp_id      = tid / 32;
    const int warp_row     = warp_id * (M_FRAGS * WMMA_M); /* 0, 32, 64, 96 */
    const int t_gemm_start = t_base - (CONV_K - 1);

    /* ── Shared memory layout (overlaid) ──────────────────────────────── */
    /* smem_cw always live; pipeline buffers & smem_proj share same region */
    extern __shared__ float smem[];
    float*       smem_cw = smem;                                       /* [64, 4]   */
    nv_bfloat16* smem_A  = (nv_bfloat16*)(smem_cw + TILE_C * CONV_K);
    nv_bfloat16* smem_B  = smem_A + PIPE * GEMM_T_PAD * SMEM_LDA;

    /* ── Load conv weights ───────────────────────────────────────────── */
    for (int i = tid; i < TILE_C * CONV_K; i += THREADS) {
        int c_local = i / CONV_K;
        int j       = i % CONV_K;
        smem_cw[c_local * CONV_K + j] =
            __bfloat162float(conv_w[(c_base + c_local) * CONV_K + j]);
    }

    /* ── WMMA accumulators (per-warp, M_FRAGS×N_FRAGS = 2×4 tiles) ──── */
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_C[M_FRAGS][N_FRAGS];
    #pragma unroll
    for (int m = 0; m < M_FRAGS; m++)
        #pragma unroll
        for (int n = 0; n < N_FRAGS; n++)
            wmma::fill_fragment(frag_C[m][n], 0.0f);

    const int K_TILES = D / BK;

    /* ── Cooperative load macro (all 128 threads) ────────────────────── *
     *   A tile: 128 × 32 bf16 = 8192 B = 512 × 16-B chunks -> 4 passes.
     *   B tile:  64 × 32 bf16 = 4096 B = 256 × 16-B chunks -> 2 passes.
     *   Rows outside the valid time range get nb=0 -> cp.async zero-fills. */
    #define ISSUE_STAGE(stage, k_val)                                           \
    {                                                                           \
        nv_bfloat16* sA = smem_A + (stage) * GEMM_T_PAD * SMEM_LDA;            \
        nv_bfloat16* sB = smem_B + (stage) * TILE_C    * SMEM_LDB;             \
        int _bk = (k_val) * BK;                                                \
        /* A-tile: 128 rows × 4 chunks/row -> 4 passes of 128 threads */        \
        _Pragma("unroll")                                                       \
        for (int pass = 0; pass < 4; pass++) {                                  \
            int cid    = tid + pass * THREADS;     /* 0..511            */      \
            int row    = cid >> 2;                 /* 0..127            */      \
            int off    = (cid & 3) << 3;           /* 0, 8, 16, 24 bf16 */      \
            int abs_t  = t_gemm_start + row;                                    \
            bool okA   = (abs_t >= 0) && (abs_t < T);                           \
            int nbA    = okA ? 16 : 0;                                          \
            int safe_t = okA ? abs_t : 0;                                       \
            cp_async_16(                                                        \
                sA + row * SMEM_LDA + off,                                      \
                input + ((size_t)b * T + safe_t) * D + _bk + off,              \
                nbA);                                                           \
        }                                                                       \
        /* B-tile: 64 rows × 4 chunks/row -> 2 passes of 128 threads */         \
        _Pragma("unroll")                                                       \
        for (int pass = 0; pass < 2; pass++) {                                  \
            int cid    = tid + pass * THREADS;     /* 0..255            */      \
            int row    = cid >> 2;                 /* 0..63             */      \
            int off    = (cid & 3) << 3;                                        \
            cp_async_16(                                                        \
                sB + row * SMEM_LDB + off,                                      \
                weight + (size_t)(c_base + row) * D + _bk + off,               \
                16);                                                            \
        }                                                                       \
    }

    /* ── Pre-issue PIPE-1 stages (K = 0..PIPE-2) ─────────────────────── */
    #pragma unroll
    for (int s = 0; s < PIPE - 1; s++) {
        if (s < K_TILES) { ISSUE_STAGE(s, s); }
        cp_async_commit();
    }

    /* ── Main loop ───────────────────────────────────────────────────── *
     *  Invariant at the top of iter k:
     *    - Committed groups = (PIPE-1) + k
     *    - wait_group<PIPE-2> ⇒ completed ≥ k+1 ⇒ group k is done.
     *  Race analysis: iter k issues at the *tail* to stage (k+PIPE-1)%PIPE,
     *  which is NOT stage k%PIPE.  Iter (k+1) will issue to stage
     *  (k+PIPE)%PIPE = k%PIPE — same slot as iter k's MMA.  But iter
     *  (k+1)'s __syncthreads comes *before* its issue, so all warps are
     *  guaranteed to have finished iter k's MMA (and its load_matrix_sync
     *  reads) before any warp's cp.async writes to that slot.
     */
    for (int k = 0; k < K_TILES; k++) {
        cp_async_wait_group<PIPE - 2>();
        __syncthreads();

        int s_cur = k % PIPE;
        nv_bfloat16* sA = smem_A + s_cur * GEMM_T_PAD * SMEM_LDA;
        nv_bfloat16* sB = smem_B + s_cur * TILE_C    * SMEM_LDB;

        #pragma unroll
        for (int kk = 0; kk < K_STEPS; kk++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           nv_bfloat16, wmma::row_major> frag_A[M_FRAGS];
            #pragma unroll
            for (int m = 0; m < M_FRAGS; m++) {
                wmma::load_matrix_sync(
                    frag_A[m],
                    sA + (warp_row + m * WMMA_M) * SMEM_LDA + kk * WMMA_K,
                    SMEM_LDA);
            }

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           nv_bfloat16, wmma::col_major> frag_B[N_FRAGS];
            #pragma unroll
            for (int n = 0; n < N_FRAGS; n++) {
                wmma::load_matrix_sync(
                    frag_B[n],
                    sB + n * WMMA_N * SMEM_LDB + kk * WMMA_K,
                    SMEM_LDB);
            }
            #pragma unroll
            for (int m = 0; m < M_FRAGS; m++) {
                #pragma unroll
                for (int n = 0; n < N_FRAGS; n++) {
                    wmma::mma_sync(frag_C[m][n], frag_A[m], frag_B[n], frag_C[m][n]);
                }
            }
        }

        /* Issue the (k+PIPE-1)-th K-tile load for a future iter. */
        int k_next = k + PIPE - 1;
        if (k_next < K_TILES) {
            ISSUE_STAGE(k_next % PIPE, k_next);
        }
        cp_async_commit();   /* empty commit if k_next >= K_TILES */
    }
    #undef ISSUE_STAGE

    /* Drain all outstanding cp.async groups, then sync warps before
     * reusing the pipeline-buffer region as smem_proj (fp32).         */
    cp_async_wait_group<0>();
    __syncthreads();

    float* smem_proj = (float*)(smem_cw + TILE_C * CONV_K);  /* overlays smem_A/B */

    /* ── Store accumulators to smem_proj ─────────────────────────────── */
    #pragma unroll
    for (int m = 0; m < M_FRAGS; m++) {
        #pragma unroll
        for (int n = 0; n < N_FRAGS; n++) {
            wmma::store_matrix_sync(
                smem_proj + (warp_row + m * WMMA_M) * TILE_C + n * WMMA_N,
                frag_C[m][n], TILE_C, wmma::mem_row_major);
        }
    }
    __syncthreads();

    /* ── Phase 2: Causal Conv1D + SiLU ───────────────────────────────── */
    constexpr int PHASE2_ITERS = (TILE_T * TILE_C + THREADS - 1) / THREADS; /* 63 */
    #pragma unroll 4
    for (int e = 0; e < PHASE2_ITERS; e++) {
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
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_w)
{
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && conv_w.is_cuda(),
                "all tensors must be on CUDA");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && conv_w.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(input.scalar_type()  == torch::kBFloat16 &&
                weight.scalar_type() == torch::kBFloat16 &&
                conv_w.scalar_type() == torch::kBFloat16,
                "all tensors must be bfloat16");

    const int B     = input.size(0);
    const int T     = input.size(1);
    const int D     = input.size(2);
    const int D_out = weight.size(0);

    TORCH_CHECK(weight.size(1) == D,      "weight dim-1 must equal D");
    TORCH_CHECK(conv_w.size(0) == D_out,  "conv_w dim-0 must equal D_out");
    TORCH_CHECK(conv_w.size(1) == CONV_K, "conv_w dim-1 must equal CONV_K");
    TORCH_CHECK(D_out % TILE_C == 0,      "D_out must be a multiple of TILE_C (64)");
    TORCH_CHECK(D % BK == 0,              "D must be a multiple of BK (32)");

    auto output = torch::empty({B, T, D_out}, input.options());

    dim3 grid((T + TILE_T - 1) / TILE_T, D_out / TILE_C, B);
    dim3 block(THREADS);

    /* smem = smem_cw (always) + max(pipeline_buffers, smem_proj) */
    const int cw_bytes   = TILE_C * CONV_K * (int)sizeof(float);
    const int pipe_bytes = PIPE * (GEMM_T_PAD * SMEM_LDA + TILE_C * SMEM_LDB)
                           * (int)sizeof(nv_bfloat16);
    const int proj_bytes = GEMM_T_PAD * TILE_C * (int)sizeof(float);
    const int smem_bytes = cw_bytes + (pipe_bytes > proj_bytes ? pipe_bytes : proj_bytes);

    static bool smem_attr_set = false;
    if (!smem_attr_set) {
        cudaError_t err = cudaFuncSetAttribute(
            fused_proj_conv_silu_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
        TORCH_CHECK(err == cudaSuccess,
            "cudaFuncSetAttribute failed to grant ", smem_bytes,
            " B of dynamic smem: ", cudaGetErrorString(err));
        smem_attr_set = true;
    }

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
          "Fused Linear Projection + Causal Depthwise Conv1D + SiLU "
          "(4-warp non-specialized, BF16)");
}
