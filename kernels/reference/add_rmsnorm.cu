// add_rmsnorm.cu — FlashInfer-style fused Add + RMSNorm baseline kernel
// Reference (unoptimized) implementation for WaferBench target.
// Operation: output = RMSNorm(input + residual) * weight
// RMSNorm: x / sqrt(mean(x^2) + eps) * weight

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include "../common/nvfp4_utils.cuh"
#include "../common/b200_intrinsics.cuh"

#define RMSNORM_BLOCK_SIZE  256
#define RMSNORM_WARP_SIZE   32
#define RMSNORM_EPSILON     1e-6f

// Baseline add + RMSNorm kernel (scalar, no vectorization)
// Grid:  (num_rows,)  Block: (RMSNORM_BLOCK_SIZE,)
// Each block handles one row of [hidden_size] elements
__global__ void add_rmsnorm_kernel_baseline(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float*       __restrict__ output,
    float*       __restrict__ residual_out,
    int hidden_size,
    float eps)
{
    extern __shared__ float smem[];
    float* partial_sums = smem;

    int row  = blockIdx.x;
    int tid  = threadIdx.x;
    int base = row * hidden_size;

    // Phase 1: add input + residual, compute sum of squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = input[base + i] + residual[base + i];
        residual_out[base + i] = x;
        local_ss += x * x;
    }

    // Block-level reduction (naive — intentionally unoptimized baseline)
    partial_sums[tid] = local_ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial_sums[tid] += partial_sums[tid + stride];
        __syncthreads();
    }

    float rms_inv = rsqrtf(partial_sums[0] / hidden_size + eps);

    // Phase 2: normalize and apply weight
    for (int i = tid; i < hidden_size; i += blockDim.x)
        output[base + i] = residual_out[base + i] * rms_inv * weight[i];
}

// Warp-optimized add + RMSNorm with warp reductions
__global__ void add_rmsnorm_kernel_warp(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16*       __restrict__ output,
    __nv_bfloat16*       __restrict__ residual_out,
    int hidden_size,
    float eps)
{
    extern __shared__ float smem_warp[];

    int row       = blockIdx.x;
    int tid       = threadIdx.x;
    int wid       = tid / RMSNORM_WARP_SIZE;
    int lane      = tid % RMSNORM_WARP_SIZE;
    int num_warps = blockDim.x / RMSNORM_WARP_SIZE;
    int base      = row * hidden_size;

    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float a = __bfloat162float(input[base + i]);
        float r = __bfloat162float(residual[base + i]);
        float x = a + r;
        residual_out[base + i] = __float2bfloat16(x);
        local_ss += x * x;
    }

    local_ss = warp_reduce_sum(local_ss);
    if (lane == 0) smem_warp[wid] = local_ss;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < num_warps) ? smem_warp[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) smem_warp[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(smem_warp[0] / hidden_size + eps);

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = __bfloat162float(residual_out[base + i]);
        float w = __bfloat162float(weight[i]);
        output[base + i] = __float2bfloat16(x * rms_inv * w);
    }
}

// Fused Add+RMSNorm+NVFP4 quantize — the primary WaferBench target
#define FUSED_BLOCK_THREADS 128

__global__ void fused_add_rmsnorm_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16*       __restrict__ residual_out,
    uint8_t*             __restrict__ quant_out,
    __nv_bfloat16*       __restrict__ quant_scales,
    int hidden_size,
    float eps)
{
    extern __shared__ float smem_fused[];
    float* warp_sums = smem_fused;

    int row       = blockIdx.x;
    int tid       = threadIdx.x;
    int lane      = tid % 32;
    int wid       = tid / 32;
    int num_warps = blockDim.x / 32;
    int base      = row * hidden_size;

    // Phase 1: Add + accumulate sum-of-squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float a = __bfloat162float(input[base + i]);
        float r = __bfloat162float(residual[base + i]);
        float x = a + r;
        residual_out[base + i] = __float2bfloat16(x);
        local_ss += x * x;
    }

    local_ss = warp_reduce_sum(local_ss);
    if (lane == 0) warp_sums[wid] = local_ss;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(warp_sums[0] / hidden_size + eps);

    // Phase 2: Normalize + weight + quantize to NVFP4
    int num_quant_blocks = hidden_size / NVFP4_BLOCK_SIZE;
    for (int qb = tid; qb < num_quant_blocks; qb += blockDim.x) {
        int elem_base = qb * NVFP4_BLOCK_SIZE;
        float block_vals[NVFP4_BLOCK_SIZE];
        #pragma unroll
        for (int j = 0; j < NVFP4_BLOCK_SIZE; ++j) {
            float x = __bfloat162float(residual_out[base + elem_base + j]);
            float w = __bfloat162float(rms_weight[elem_base + j]);
            block_vals[j] = x * rms_inv * w;
        }

        uint8_t packed_out[NVFP4_BLOCK_SIZE / 2];
        __nv_bfloat16 scale_out;
        quantize_block_nvfp4(block_vals, packed_out, &scale_out);

        int packed_base = (row * num_quant_blocks + qb) * (NVFP4_BLOCK_SIZE / 2);
        #pragma unroll
        for (int j = 0; j < NVFP4_BLOCK_SIZE / 2; ++j)
            quant_out[packed_base + j] = packed_out[j];
        quant_scales[row * num_quant_blocks + qb] = scale_out;
    }
}

// Host launch wrapper
void launch_fused_add_rmsnorm_nvfp4(
    const __nv_bfloat16* input, const __nv_bfloat16* residual,
    const __nv_bfloat16* rms_weight, __nv_bfloat16* residual_out,
    uint8_t* quant_out, __nv_bfloat16* quant_scales,
    int num_rows, int hidden_size, cudaStream_t stream)
{
    int num_warps = FUSED_BLOCK_THREADS / 32;
    dim3 grid(num_rows);
    dim3 block(FUSED_BLOCK_THREADS);
    size_t smem = num_warps * sizeof(float);
    fused_add_rmsnorm_nvfp4_kernel<<<grid, block, smem, stream>>>(
        input, residual, rms_weight, residual_out,
        quant_out, quant_scales, hidden_size, 1e-6f);
}
