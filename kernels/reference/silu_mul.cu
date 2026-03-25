// silu_mul.cu — SiLU × gate (gated MLP activation) baseline kernel
// Operation: output = SiLU(gate) * up
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../common/b200_intrinsics.cuh"

#define SILU_BLOCK_SIZE 256

// Scalar SiLU × gate kernel (baseline)
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

// BFloat16 SiLU × gate with float4 vectorization (4 elements per thread)
__device__ __forceinline__ float silu_f32(float x) {
    return x * __frcp_rn(1.0f + expf(-x));
}

__global__ void silu_mul_kernel_bf16_vec4(
    const __nv_bfloat162* __restrict__ gate,
    const __nv_bfloat162* __restrict__ up,
    __nv_bfloat162*       __restrict__ output,
    int half_N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 >= half_N) return;

    __nv_bfloat162 g0 = gate[idx * 2];
    __nv_bfloat162 g1 = gate[idx * 2 + 1];
    __nv_bfloat162 u0 = up[idx * 2];
    __nv_bfloat162 u1 = up[idx * 2 + 1];

    float g0x = silu_f32(__bfloat162float(g0.x));
    float g0y = silu_f32(__bfloat162float(g0.y));
    float g1x = silu_f32(__bfloat162float(g1.x));
    float g1y = silu_f32(__bfloat162float(g1.y));

    output[idx * 2]     = __floats2bfloat162_rn(
        g0x * __bfloat162float(u0.x), g0y * __bfloat162float(u0.y));
    output[idx * 2 + 1] = __floats2bfloat162_rn(
        g1x * __bfloat162float(u1.x), g1y * __bfloat162float(u1.y));
}

void launch_silu_mul_baseline(
    const float* gate, const float* up, float* output,
    int N, cudaStream_t stream)
{
    dim3 grid((N + SILU_BLOCK_SIZE - 1) / SILU_BLOCK_SIZE);
    dim3 block(SILU_BLOCK_SIZE);
    silu_mul_kernel_baseline<<<grid, block, 0, stream>>>(gate, up, output, N);
}

void launch_silu_mul_bf16(
    const __nv_bfloat162* gate, const __nv_bfloat162* up,
    __nv_bfloat162* output, int N, cudaStream_t stream)
{
    int half_N  = N / 2;
    int threads = SILU_BLOCK_SIZE;
    int blocks  = (half_N / 2 + threads - 1) / threads;
    silu_mul_kernel_bf16_vec4<<<blocks, threads, 0, stream>>>(gate, up, output, half_N);
}
