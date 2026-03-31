// add_rmsnorm.cu — Optimized fused Add + RMSNorm + NVFP4 quantize
// Fixes applied over naive baseline:
//   1. BLOCK_THREADS=128 — eliminates 50% Phase-2 thread waste
//   2. uint4 vectorized loads — 8 bf16 per transaction
//   3. Warp shuffle reduction — replaces 9x __syncthreads tree
//   4. smem bf16 cache — eliminates Phase-2 global re-read of residual_out
//   5. quantize_block_nvfp4 — uses __nv_cvt_float2_to_fp4x2 hardware instruction
//      on sm_100a instead of 7-branch scalar if-else chain

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#endif

#ifndef __CUDA_FP8_TYPES_EXIST__
typedef unsigned char __nv_fp8_storage_t;
#endif

#include "nvfp4_utils.cuh"

// ── Constants ───────────────────────────────────────────────────────────────

#define BLOCK_THREADS    128
#define HIDDEN_SIZE      2048
#define NUM_QUANT_BLOCKS (HIDDEN_SIZE / NVFP4_BLOCK_SIZE)   // 128
#define NUM_WARPS        (BLOCK_THREADS / 32)                // 4

// smem layout (4112 bytes total):
//   [0 .. 2047] : __nv_bfloat16  smem_cache  — Phase-1 residual values
//   [4096..4111]: float[4]        smem_ws     — warp partial sums

// ── Kernel ───────────────────────────────────────────────────────────────────

__launch_bounds__(BLOCK_THREADS, 16)
__global__ void fused_add_rmsnorm_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16*       __restrict__ residual_out,
    uint8_t*             __restrict__ quant_out,
    __nv_fp8_storage_t*  __restrict__ quant_scales,
    int   hidden_size,
    float eps)
{
    extern __shared__ __nv_bfloat16 smem_cache[];
    float* smem_ws = reinterpret_cast<float*>(smem_cache + HIDDEN_SIZE);

    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int base = row * hidden_size;

    // ── Phase 1: vectorized add + accumulate ss + write residual_out + smem ──
    // 128 threads × 2 uint4 loads each = 256 uint4 = 2048 bf16 (hidden_size)
    // One uint4 = 16 bytes = 8 bf16

    const uint4* __restrict__ in_vec  = reinterpret_cast<const uint4*>(input   + base);
    const uint4* __restrict__ res_vec = reinterpret_cast<const uint4*>(residual + base);
    uint4* __restrict__ ro_vec        = reinterpret_cast<uint4*>(residual_out   + base);
    uint4* __restrict__ sm_vec        = reinterpret_cast<uint4*>(smem_cache);

    float local_ss = 0.0f;

    #pragma unroll
    for (int v = 0; v < 2; ++v) {
        int vi = tid * 2 + v;
        uint4 a = __ldg(in_vec  + vi);
        uint4 b = __ldg(res_vec + vi);

        uint4 out_val;
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
        __nv_bfloat162*       o2 = reinterpret_cast<__nv_bfloat162*>(&out_val);

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 s2 = __hadd2(a2[k], b2[k]);
            o2[k] = s2;
            float x0 = __bfloat162float(s2.x);
            float x1 = __bfloat162float(s2.y);
            local_ss += x0 * x0 + x1 * x1;
        }

        ro_vec[vi] = out_val;   // write residual_out
        sm_vec[vi] = out_val;   // cache in smem — avoids Phase-2 global re-read
    }

    // ── Warp shuffle reduction ──────────────────────────────────────────────
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        local_ss += __shfl_xor_sync(0xFFFFFFFFu, local_ss, mask);

    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    if (lane_id == 0)
        smem_ws[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? smem_ws[lane_id] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFFu, val, mask);
        if (lane_id == 0)
            smem_ws[0] = val;
    }
    __syncthreads();

    const float rms_inv = rsqrtf(smem_ws[0] / hidden_size + eps);

    // ── Phase 2: one quant block per thread, reads from smem (no global re-read)
    // tid 0..127 → qb 0..127, exactly 1 block each, no idle threads

    const int qb        = tid;   // one quant block per thread
    const int elem_base = qb * NVFP4_BLOCK_SIZE;

    float block_vals[NVFP4_BLOCK_SIZE];

    #pragma unroll
    for (int j = 0; j < NVFP4_BLOCK_SIZE; ++j) {
        float x = __bfloat162float(smem_cache[elem_base + j]);
        float w = __bfloat162float(rms_weight[elem_base + j]);
        block_vals[j] = x * rms_inv * w;
    }

    // quantize_block_nvfp4 uses __nv_cvt_float2_to_fp4x2 on sm_100a —
    // one hardware instruction per pair instead of the 7-branch if-else chain
    uint8_t           packed[NVFP4_BLOCK_SIZE / 2];
    __nv_fp8_storage_t scale;
    quantize_block_nvfp4(block_vals, packed, &scale);

    const int packed_base = (row * NUM_QUANT_BLOCKS + qb) * (NVFP4_BLOCK_SIZE / 2);
    #pragma unroll
    for (int j = 0; j < NVFP4_BLOCK_SIZE / 2; ++j)
        quant_out[packed_base + j] = packed[j];

    quant_scales[row * NUM_QUANT_BLOCKS + qb] = scale;
}

// ── Host launch wrapper — signature must not change ──────────────────────────

void launch_fused_add_rmsnorm_nvfp4(
    const __nv_bfloat16* input, const __nv_bfloat16* residual,
    const __nv_bfloat16* rms_weight, __nv_bfloat16* residual_out,
    uint8_t* quant_out, __nv_fp8_storage_t* quant_scales,
    int num_rows, int hidden_size, cudaStream_t stream)
{
    dim3 grid(num_rows);
    dim3 block(BLOCK_THREADS);
    // smem: 2048 bf16 cache (4096 bytes) + 4 float warp sums (16 bytes)
    size_t smem = HIDDEN_SIZE * sizeof(__nv_bfloat16) + NUM_WARPS * sizeof(float);
    fused_add_rmsnorm_nvfp4_kernel<<<grid, block, smem, stream>>>(
        input, residual, rms_weight, residual_out,
        quant_out, quant_scales, hidden_size, 1e-6f);
}
