// nvfp4_quantize.cu — Block-wise NVFP4 quantization kernel (reference)
// Quantizes bf16 activations to NVFP4 for B200 TMEM/tensor ops.
// Layout: [num_blocks * 8 bytes data] + [num_blocks * 2 bytes scales]

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../common/nvfp4_utils.cuh"
#include "../common/b200_intrinsics.cuh"

#define QUANT_BLOCK_THREADS 256

// BFloat16 → NVFP4 quantization (1 thread = 1 block of 16 elements)
__global__ void nvfp4_quantize_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t*             __restrict__ packed,
    __nv_bfloat16*       __restrict__ scales,
    int N)
{
    int block_id  = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = N / NVFP4_BLOCK_SIZE;
    if (block_id >= num_blocks) return;

    int base = block_id * NVFP4_BLOCK_SIZE;
    float x[NVFP4_BLOCK_SIZE];
    #pragma unroll
    for (int i = 0; i < NVFP4_BLOCK_SIZE; ++i)
        x[i] = __bfloat162float(input[base + i]);

    uint8_t packed_out[NVFP4_BLOCK_SIZE / 2];
    __nv_bfloat16 scale_out;
    quantize_block_nvfp4(x, packed_out, &scale_out);

    int packed_base = block_id * (NVFP4_BLOCK_SIZE / 2);
    #pragma unroll
    for (int i = 0; i < NVFP4_BLOCK_SIZE / 2; ++i)
        packed[packed_base + i] = packed_out[i];
    scales[block_id] = scale_out;
}

// NVFP4 → BFloat16 dequantization
__global__ void nvfp4_dequantize_bf16_kernel(
    const uint8_t*       __restrict__ packed,
    const __nv_bfloat16* __restrict__ scales,
    __nv_bfloat16*       __restrict__ output,
    int N)
{
    int block_id  = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = N / NVFP4_BLOCK_SIZE;
    if (block_id >= num_blocks) return;

    int packed_base = block_id * (NVFP4_BLOCK_SIZE / 2);
    __nv_bfloat16 scale = scales[block_id];

    float out[NVFP4_BLOCK_SIZE];
    dequantize_block_nvfp4(&packed[packed_base], scale, out);

    int base = block_id * NVFP4_BLOCK_SIZE;
    #pragma unroll
    for (int i = 0; i < NVFP4_BLOCK_SIZE; ++i)
        output[base + i] = __float2bfloat16(out[i]);
}

void launch_nvfp4_quantize_bf16(
    const __nv_bfloat16* input, uint8_t* packed, __nv_bfloat16* scales,
    int N, cudaStream_t stream)
{
    int num_blocks = N / NVFP4_BLOCK_SIZE;
    dim3 grid((num_blocks + QUANT_BLOCK_THREADS - 1) / QUANT_BLOCK_THREADS);
    nvfp4_quantize_bf16_kernel<<<grid, QUANT_BLOCK_THREADS, 0, stream>>>(
        input, packed, scales, N);
}

void launch_nvfp4_dequantize_bf16(
    const uint8_t* packed, const __nv_bfloat16* scales,
    __nv_bfloat16* output, int N, cudaStream_t stream)
{
    int num_blocks = N / NVFP4_BLOCK_SIZE;
    dim3 grid((num_blocks + QUANT_BLOCK_THREADS - 1) / QUANT_BLOCK_THREADS);
    nvfp4_dequantize_bf16_kernel<<<grid, QUANT_BLOCK_THREADS, 0, stream>>>(
        packed, scales, output, N);
}
