"""
sub_prompts.py — Strategy-specific prompts for sub-LLMs.
Sub-LLMs only see the relevant kernel slice, not the full source.
"""

from __future__ import annotations


def vectorize_loads_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    mem_bw = hw_spec["memory"]["hbm_bandwidth_tbs"]
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent NCU metrics:\n"
            f"  Memory throughput: {current_metrics.get('mem_throughput_pct', 'N/A')}%\n"
            f"  DRAM stall rate: {current_metrics.get('stall_memory', 'N/A')}%\n"
            f"  L2 hit rate: {current_metrics.get('l2_hit_rate', 'N/A')}%\n"
        )
    return f"""\
## Optimization Task: Vectorize Global Memory Loads

Hardware: NVIDIA B200 ({mem_bw} TB/s HBM3e bandwidth)
{metrics_str}
Target kernel section:
```cuda
{kernel_slice}
```

Apply float4 / uint4 vectorized loads (128-bit transactions):

Requirements:
1. Replace scalar `float` loads with `float4` where 16-byte aligned
2. For bfloat16: use `uint4` loads then reinterpret as `__nv_bfloat162 x4`
3. Ensure pointer alignment with `__builtin_assume_aligned(ptr, 16)`
4. Same output semantics — no change to computed values
5. Do NOT add new __syncthreads() calls
6. Add `#pragma unroll 4` hints where appropriate

Return ONLY the complete optimized CUDA code in a single ```cuda code block.
No explanations, no markdown headings, no prose — just the code block.
"""


def tma_prefetch_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent NCU metrics:\n"
            f"  Memory throughput: {current_metrics.get('mem_throughput_pct', 'N/A')}%\n"
            f"  Long scoreboard stalls: {current_metrics.get('stall_memory', 'N/A')}%\n"
        )
    return f"""\
## Optimization Task: TMA Async Prefetch

Hardware: NVIDIA B200 (Blackwell) with TMA support
{metrics_str}
Target kernel section:
```cuda
{kernel_slice}
```

Add TMA async prefetch with double-buffering:

Requirements:
1. Use `tma_load_1d()` from b200_intrinsics.cuh for async copy
2. Double-buffered shared memory: two ping-pong buffers
3. Initialize mbarrier with `mbar_init()`, wait with `mbar_wait()`
4. Issue prefetch 1 tile ahead
5. Elect threadIdx.x == 0 to issue TMA
6. __syncthreads() or mbar_wait() before consuming prefetched data

Shared memory layout:
  __shared__ T smem_buf[2][TILE_SIZE];  // double buffer
  __shared__ uint64_t mbar[2];

Return ONLY the complete optimized CUDA code in a single ```cuda code block.
No explanations, no markdown headings, no prose — just the code block.
"""


def warp_reduction_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent NCU metrics:\n"
            f"  Barrier stall rate: {current_metrics.get('stall_barrier', 'N/A')}%\n"
            f"  SM occupancy: {current_metrics.get('sm_occupancy', 'N/A')}%\n"
        )
    return f"""\
## Optimization Task: Replace Block Reduction with Warp Shuffles

Hardware: NVIDIA B200 (sm_100a, __shfl_xor_sync supported)
{metrics_str}
Target kernel section (contains __syncthreads-based reduction):
```cuda
{kernel_slice}
```

Replace shared-memory block reduction with warp-level shuffles:

Requirements:
1. Use `warp_reduce_sum()` from nvfp4_utils.cuh (__shfl_xor_sync internally)
2. Pattern: each warp reduces locally → warp leaders write to shared mem
   → warp 0 reduces the warp partial sums
3. Only ONE __syncthreads() instead of O(log N) naive syncs
4. For RMSNorm: accumulate `x*x` locally, warp reduce, then combine warps
5. Use `0xFFFFFFFF` as the warp mask (full warp participation)
6. Preserve numerical equivalence

Return ONLY the complete optimized CUDA code in a single ```cuda code block.
No explanations, no markdown headings, no prose — just the code block.
"""


def fuse_passes_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent NCU metrics:\n"
            f"  Memory throughput: {current_metrics.get('mem_throughput_pct', 'N/A')}%\n"
            f"  Achieved occupancy: {current_metrics.get('sm_occupancy', 'N/A')}%\n"
        )
    return f"""\
## Optimization Task: Fuse Multiple Memory Passes into One

Hardware: NVIDIA B200 ({hw_spec['memory']['hbm_bandwidth_tbs']} TB/s HBM3e)
{metrics_str}
Target kernel (currently makes multiple passes over data):
```cuda
{kernel_slice}
```

Fuse separate read passes into a single loop:

Requirements:
1. Identify values computed in Pass 1 needed in Pass 2
2. Use registers or shared memory to carry values across phases
3. Single loop: load → compute all phases → store results
4. Register budget: B200 has 255 registers/thread — use ~64
5. Use `__ldg()` for read-only data (L1 texture cache)
6. Mark read-only pointers with `__restrict__` and `const`

Return ONLY the complete optimized CUDA code in a single ```cuda code block.
No explanations, no markdown headings, no prose — just the code block. Add register count as an inline comment.
"""


def register_tiling_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent NCU metrics:\n"
            f"  Compute throughput: {current_metrics.get('compute_throughput_pct', 'N/A')}%\n"
            f"  Achieved occupancy: {current_metrics.get('sm_occupancy', 'N/A')}%\n"
        )
    return f"""\
## Optimization Task: Register-Level Tiling for Compute ILP

Hardware: NVIDIA B200 (255 registers/thread, 4-wide SIMD fp32)
{metrics_str}
Target kernel section:
```cuda
{kernel_slice}
```

Apply register tiling to increase ILP:

Requirements:
1. Unroll inner loop by 4x — process 4 independent elements per iteration
2. Use separate register variables (avoid array subscripts in hot path)
3. Interleave independent computations to hide latency
4. For RMSNorm: compute 4 partial sums simultaneously before reducing
5. Preserve #pragma unroll for the compiler
6. Do not exceed 96 registers/thread

Return ONLY the complete optimized CUDA code in a single ```cuda code block.
No explanations, no markdown headings, no prose — just the code block.
"""


def async_pipeline_prompt(kernel_slice: str, hw_spec: dict, current_metrics: dict = None) -> str:
    metrics_str = ""
    if current_metrics:
        metrics_str = (
            f"\nCurrent NCU metrics:\n"
            f"  Long scoreboard stalls: {current_metrics.get('stall_memory', 'N/A')}%\n"
            f"  Memory throughput: {current_metrics.get('mem_throughput_pct', 'N/A')}%\n"
        )
    return f"""\
## Optimization Task: Async Software Pipeline (cp.async)

Hardware: NVIDIA B200 (Blackwell, cp.async.bulk supported)
{metrics_str}
Target kernel section:
```cuda
{kernel_slice}
```

Implement a software pipeline using cp.async:

Requirements:
1. Use `cuda::pipeline` from <cuda/pipeline>
2. Pattern: issue cp.async for next tile → compute on current tile → commit + wait
3. Pipeline depth = 2 (double buffer)
4. Shared memory: allocate 2x tile_size for ping-pong
5. Use `__pipeline_memcpy_async()` or TMA for bulk async copy
6. Commit with `__pipeline_commit()`
7. Wait with `__pipeline_wait_prior(1)` — allow 1 outstanding stage
8. Handle prologue (first tile) and epilogue (drain) correctly

Return ONLY the complete optimized CUDA code in a single ```cuda code block.
No explanations, no markdown headings, no prose — just the code block.
"""


STRATEGY_PROMPTS = {
    "vectorize_loads":  vectorize_loads_prompt,
    "tma_prefetch":     tma_prefetch_prompt,
    "warp_reduction":   warp_reduction_prompt,
    "fuse_passes":      fuse_passes_prompt,
    "register_tiling":  register_tiling_prompt,
    "async_pipeline":   async_pipeline_prompt,
}


def get_prompt_for_strategy(
    strategy: str,
    kernel_slice: str,
    hw_spec: dict,
    current_metrics: dict = None,
) -> str:
    if strategy not in STRATEGY_PROMPTS:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(STRATEGY_PROMPTS)}")
    return STRATEGY_PROMPTS[strategy](kernel_slice, hw_spec, current_metrics)
