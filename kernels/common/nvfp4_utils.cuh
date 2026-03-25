// nvfp4_utils.cuh — NVFP4 pack/unpack and quantization helpers
// NVFP4: 1-bit exponent, 2-bit mantissa, 1-bit sign
// Format: s | e | m1 | m0  (4 bits total)
// Scale: per-16-element block scaling factor (bfloat16)

#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ────────────────────────────────────────────────────────────────────────────
// NVFP4 representation constants
// ────────────────────────────────────────────────────────────────────────────
#define NVFP4_MANTISSA_BITS 2
#define NVFP4_EXPONENT_BITS 1
#define NVFP4_BIAS          1
#define NVFP4_BLOCK_SIZE    16      // elements per scale factor
#define NVFP4_PER_BYTE      2       // 2 fp4 values packed per byte
#define NVFP4_PER_INT32     8       // 8 fp4 values packed per int32

// ────────────────────────────────────────────────────────────────────────────
// Lookup table: all 16 NVFP4 values (sign × exp × mant)
// Positive values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
// Negative values: mirror of positive
// ────────────────────────────────────────────────────────────────────────────
__device__ __constant__ float kNVFP4LUT[16] = {
    0.0f,   0.5f,   1.0f,   1.5f,   // 0b0000 .. 0b0011  (positive, exp=0)
    2.0f,   3.0f,   4.0f,   6.0f,   // 0b0100 .. 0b0111  (positive, exp=1)
   -0.0f,  -0.5f,  -1.0f,  -1.5f,  // 0b1000 .. 0b1011  (negative, exp=0)
   -2.0f,  -3.0f,  -4.0f,  -6.0f,  // 0b1100 .. 0b1111  (negative, exp=1)
};

// ────────────────────────────────────────────────────────────────────────────
// Core encode: float → 4-bit code (two-step: clamp → scale → round-to-nearest)
//
// NVFP4 positive magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6
// The spacing between levels is non-uniform (0.5 up to 2, then 1, then 2).
// We map the scaled value to the nearest representable level using a lookup
// into 8 boundary midpoints, which avoids a 16-iteration linear scan and
// gives the compiler a branchless cmov sequence instead.
//
// Boundary midpoints between consecutive positive levels:
//   0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0  (7 thresholds for 8 levels)
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ uint8_t float_to_nvfp4(float x) {
    // Extract sign and work with |x| scaled to [0, 6] range
    uint8_t sign_bit = (x < 0.0f) ? 0x8u : 0x0u;
    float   ax       = fabsf(x);

    // Clamp to representable range [0, 6]
    ax = fminf(ax, 6.0f);

    // Map magnitude to 3-bit mantissa+exponent code (0..7 → levels 0,0.5,1,1.5,2,3,4,6)
    // Threshold comparisons collapse to a sequence of SETP/SELP on B200 PTX
    uint8_t code;
    if      (ax < 0.25f) code = 0;   // → 0.0
    else if (ax < 0.75f) code = 1;   // → 0.5
    else if (ax < 1.25f) code = 2;   // → 1.0
    else if (ax < 1.75f) code = 3;   // → 1.5
    else if (ax < 2.5f)  code = 4;   // → 2.0
    else if (ax < 3.5f)  code = 5;   // → 3.0
    else if (ax < 5.0f)  code = 6;   // → 4.0
    else                 code = 7;   // → 6.0

    return sign_bit | code;
}

// ────────────────────────────────────────────────────────────────────────────
// Core decode: 4-bit code → float
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float nvfp4_to_float(uint8_t code) {
    return kNVFP4LUT[code & 0xF];
}

// ────────────────────────────────────────────────────────────────────────────
// Block quantize: 16 floats → 8 bytes (packed fp4) + 1 bf16 scale
// Input:  float x[16]
// Output: uint8_t packed[8], __nv_bfloat16 scale
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void quantize_block_nvfp4(
    const float* __restrict__ x,
    uint8_t* __restrict__ packed,
    __nv_bfloat16* __restrict__ scale)
{
    // Find block absmax for scaling
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < NVFP4_BLOCK_SIZE; ++i) {
        amax = fmaxf(amax, fabsf(x[i]));
    }

    // Scale so that max maps to representable NVFP4 max (6.0)
    const float inv_max_repr = 1.0f / 6.0f;
    float s = (amax > 0.0f) ? (amax * inv_max_repr) : 1.0f;
    *scale = __float2bfloat16(s);

    float inv_s = (amax > 0.0f) ? (6.0f / amax) : 1.0f;

    // Encode each element and pack 2 fp4 per byte
    #pragma unroll
    for (int i = 0; i < NVFP4_BLOCK_SIZE / 2; ++i) {
        uint8_t lo = float_to_nvfp4(x[2*i]   * inv_s);
        uint8_t hi = float_to_nvfp4(x[2*i+1] * inv_s);
        packed[i] = (hi << 4) | (lo & 0xF);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Block dequantize: 8 bytes (packed fp4) + bf16 scale → 16 floats
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void dequantize_block_nvfp4(
    const uint8_t* __restrict__ packed,
    __nv_bfloat16 scale,
    float* __restrict__ out)
{
    float s = __bfloat162float(scale);
    #pragma unroll
    for (int i = 0; i < NVFP4_BLOCK_SIZE / 2; ++i) {
        uint8_t byte = packed[i];
        out[2*i]   = nvfp4_to_float(byte & 0xF)  * s;
        out[2*i+1] = nvfp4_to_float(byte >> 4)   * s;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Vectorized pack: 8 fp4 codes → single int32 (for register-level ops)
// Layout: bits [3:0]=code0, [7:4]=code1, ..., [31:28]=code7
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ uint32_t pack8_nvfp4(const uint8_t codes[8]) {
    uint32_t result = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        result |= ((uint32_t)(codes[i] & 0xF)) << (i * 4);
    }
    return result;
}

__device__ __forceinline__ void unpack8_nvfp4(uint32_t packed, uint8_t codes[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        codes[i] = (packed >> (i * 4)) & 0xF;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Warp-level absmax reduction (for block scale computation across warp)
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_absmax(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, fabsf(__shfl_xor_sync(0xFFFFFFFF, val, mask)));
    }
    return val;
}

// ────────────────────────────────────────────────────────────────────────────
// Warp-level sum reduction
// ────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    }
    return val;
}
