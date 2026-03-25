"""
combiner.py — Merge top-K beam variants into a single optimized kernel.
Used as fallback when LLM-based combination fails.
"""

from __future__ import annotations
import re
import logging
from rlm.environment import KernelCandidate

logger = logging.getLogger(__name__)


def extract_kernel_body(cuda_src: str) -> str:
    match = re.search(r"__global__[^{]+\{", cuda_src, re.DOTALL)
    if not match:
        return cuda_src
    start = match.end()
    depth = 1
    pos   = start
    while pos < len(cuda_src) and depth > 0:
        if cuda_src[pos] == "{":   depth += 1
        elif cuda_src[pos] == "}": depth -= 1
        pos += 1
    return cuda_src[start:pos-1].strip()


def extract_shared_memory_decls(cuda_src: str) -> list:
    return re.findall(r"__shared__[^;]+;", cuda_src)


def naive_merge(variant_a: KernelCandidate, variant_b: KernelCandidate) -> str:
    """Naive merge: combine shared memory and loop bodies from both variants."""
    body_a = extract_kernel_body(variant_a.code)
    body_b = extract_kernel_body(variant_b.code)
    smem_a = extract_shared_memory_decls(variant_a.code)
    smem_b = extract_shared_memory_decls(variant_b.code)
    all_smem = list(dict.fromkeys(smem_a + smem_b))

    return f"""// Auto-merged: {variant_a.strategy} + {variant_b.strategy}
// Variant A speedup: {variant_a.speedup:.3f}x ({variant_a.strategy})
// Variant B speedup: {variant_b.speedup:.3f}x ({variant_b.strategy})
// WARNING: Naive merge — review for correctness before production use

#include "../common/nvfp4_utils.cuh"
#include "../common/b200_intrinsics.cuh"

__global__ void fused_add_rmsnorm_nvfp4_optimized(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ rms_weight,
    __nv_bfloat16*       __restrict__ residual_out,
    uint8_t*             __restrict__ quant_out,
    __nv_bfloat16*       __restrict__ quant_scales,
    int hidden_size,
    float eps)
{{
    // Shared memory (merged from both variants)
    {chr(10).join('    ' + s for s in all_smem)}

    // === From Variant A ({variant_a.strategy}) ===
    {{
        {body_a[:2000]}
    }}

    // === From Variant B ({variant_b.strategy}) ===
    {{
        {body_b[:2000]}
    }}
}}
"""
