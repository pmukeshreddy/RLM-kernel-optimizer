// silu_mul.cu — Fused SiLU × Mul + NVFP4 quantize kernel
// Operation: output = NVFP4_quantize(SiLU(gate) * up)
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../common/nvfp4_utils.cuh"
#include "../common/b200_intrinsics.cuh"

#define SILU_BLOCK_SIZE 256

// Scalar SiLU × gate kernel (baseline, no quant)
__global__ void silu_mul_kernel_baseline(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float*       __restrict__ output,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float g = gate[idx];
    float u = up[idx];
    float silu = g / (1.0f + expf(-g));
    output[idx] = silu * u;
}

__device__ __forceinline__ float silu_f32(float x) {
    return x * __frcp_rn(1.0f + expf(-x));
}

// ── Fused SiLU × Mul + NVFP4 quantize (primary WaferBench target) ──────────
// Each thread handles one 16-element NVFP4 block:
//   1. Load 16 bf16 gate + 16 bf16 up values
//   2. Compute SiLU(gate) * up in fp32
//   3. Block-quantize to NVFP4 with E4M3 scale
#define FUSED_SILU_BLOCK_THREADS 128

__global__ void silu_mul_fp4quant_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    uint8_t*             __restrict__ quant_out,
    __nv_fp8_storage_t*  __restrict__ quant_scales,
    int N)
{
    int num_quant_blocks = N / NVFP4_BLOCK_SIZE;
    int qb = blockIdx.x * blockDim.x + threadIdx.x;
    if (qb >= num_quant_blocks) return;

    int elem_base = qb * NVFP4_BLOCK_SIZE;
    float block_vals[NVFP4_BLOCK_SIZE];

    #pragma unroll
    for (int j = 0; j < NVFP4_BLOCK_SIZE; ++j) {
        float g = __bfloat162float(gate[elem_base + j]);
        float u = __bfloat162float(up[elem_base + j]);
        block_vals[j] = silu_f32(g) * u;
    }

    uint8_t packed_out[NVFP4_BLOCK_SIZE / 2];
    __nv_fp8_storage_t scale_out;
    quantize_block_nvfp4(block_vals, packed_out, &scale_out);

    int packed_base = qb * (NVFP4_BLOCK_SIZE / 2);
    #pragma unroll
    for (int j = 0; j < NVFP4_BLOCK_SIZE / 2; ++j)
        quant_out[packed_base + j] = packed_out[j];
    quant_scales[qb] = scale_out;
}

void launch_silu_mul_baseline(
    const float* gate, const float* up, float* output,
    int N, cudaStream_t stream)
{
    dim3 grid((N + SILU_BLOCK_SIZE - 1) / SILU_BLOCK_SIZE);
    dim3 block(SILU_BLOCK_SIZE);
    silu_mul_kernel_baseline<<<grid, block, 0, stream>>>(gate, up, output, N);
}

// Primary launch function for WaferBench: fused SiLU*Mul + NVFP4 quantize
void launch_silu_mul_fp4quant(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    uint8_t* quant_out, __nv_fp8_storage_t* quant_scales,
    int N, cudaStream_t stream)
{
    int num_quant_blocks = N / NVFP4_BLOCK_SIZE;
    dim3 grid((num_quant_blocks + FUSED_SILU_BLOCK_THREADS - 1) / FUSED_SILU_BLOCK_THREADS);
    dim3 block(FUSED_SILU_BLOCK_THREADS);
    silu_mul_fp4quant_kernel<<<grid, block, 0, stream>>>(
        gate, up, quant_out, quant_scales, N);
}
