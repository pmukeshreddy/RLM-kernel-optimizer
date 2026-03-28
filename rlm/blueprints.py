"""
blueprints.py — Structural kernel blueprints for each kernel type.

These are concrete code skeletons showing the OPTIMAL kernel structure,
derived from analysis of hackathon-winning kernels. The LLM should write
code that follows this structure, not just tweak the naive kernel.

The key insight: the naive kernels waste cycles on unnecessary HBM round-trips
and scalar operations. The blueprints show the correct data flow.
"""

from __future__ import annotations


def _memory_analysis(kernel_type: str, problem_shape: tuple) -> str:
    """Calculate theoretical minimum latency and actual bandwidth utilization."""
    if kernel_type == "add_rmsnorm":
        rows, hidden = problem_shape
        n = rows * hidden
        # Reads: input(bf16) + residual(bf16) + rms_weight(4KB, L2-cached)
        # Writes: residual_out(bf16) + quant_out(FP4 packed) + quant_scales(FP8)
        # Naive adds: re-read residual_out in Phase 2 = extra 524KB
        read_bytes = n * 2 + n * 2 + hidden * 2  # input + residual + weight
        write_bytes = n * 2 + n // 2 + n // 16  # residual_out + quant + scales
        naive_extra = n * 2  # re-read residual_out in Phase 2
        total_min = read_bytes + write_bytes
        total_naive = total_min + naive_extra
        return (
            f"MEMORY TRAFFIC ANALYSIS (shape={problem_shape}):\n"
            f"  Minimum HBM traffic (single-pass):  {total_min:,} bytes ({total_min/1024:.0f} KB)\n"
            f"  Naive HBM traffic (2-pass):          {total_naive:,} bytes ({total_naive/1024:.0f} KB)\n"
            f"  WASTED by re-reading residual_out:   {naive_extra:,} bytes ({naive_extra/1024:.0f} KB)\n"
            f"  At 8 TB/s peak HBM: theoretical min = {total_min / 8e6:.3f} us\n"
            f"  At 4 TB/s effective: realistic min  = {total_min / 4e6:.3f} us\n"
            f"  Current ~4.1 us => bandwidth util    = {total_naive / 4.1 / 1e6:.1f} TB/s "
            f"({total_naive / 4.1 / 8e6 * 100:.0f}% of peak)\n"
            f"  ELIMINATING the residual_out re-read saves ~{naive_extra / 0.55e6:.1f} us"
        )

    elif kernel_type == "silu_mul":
        if len(problem_shape) == 3:
            b, m, k = problem_shape
            n = b * m * k
        else:
            n = problem_shape[0]
        # Reads: gate(bf16) + up(bf16)
        # Writes: quant_out(FP4 packed) + scales(FP8)
        read_bytes = n * 2 + n * 2
        write_bytes = n // 2 + n // 16
        total = read_bytes + write_bytes
        return (
            f"MEMORY TRAFFIC ANALYSIS (N={n:,}):\n"
            f"  Total HBM traffic:       {total:,} bytes ({total/1024:.0f} KB)\n"
            f"  At 8 TB/s peak HBM: theoretical min = {total / 8e6:.3f} us\n"
            f"  At 4 TB/s effective: realistic min  = {total / 4e6:.3f} us\n"
            f"  This kernel is SINGLE-PASS — no wasted re-reads.\n"
            f"  Win comes from: vectorized loads/stores, thread coarsening, fast math."
        )

    elif kernel_type == "nvfp4_quantize":
        if len(problem_shape) == 2:
            m, k = problem_shape
            n = m * k
        else:
            n = problem_shape[0]
        read_bytes = n * 2
        write_bytes = n // 2 + n // 16
        total = read_bytes + write_bytes
        return (
            f"MEMORY TRAFFIC ANALYSIS (N={n:,}):\n"
            f"  Total HBM traffic:       {total:,} bytes ({total/1024:.0f} KB)\n"
            f"  At 8 TB/s peak HBM: theoretical min = {total / 8e6:.3f} us\n"
            f"  At 4 TB/s effective: realistic min  = {total / 4e6:.3f} us\n"
            f"  This kernel is SINGLE-PASS — no wasted re-reads.\n"
            f"  Win comes from: vectorized loads/stores, thread coarsening, fewer instructions."
        )

    return ""


def get_structural_blueprint(kernel_type: str, problem_shape: tuple) -> str:
    """Return the structural blueprint for a kernel type.

    This tells the LLM EXACTLY what code structure to write,
    not just vague optimization names.
    """
    memory_analysis = _memory_analysis(kernel_type, problem_shape)

    if kernel_type == "add_rmsnorm":
        rows, hidden = problem_shape
        items_per_thread = hidden // 256  # 2048/256 = 8
        num_quant_blocks = hidden // 16   # 128
        return f"""\
## STRUCTURAL BLUEPRINT — Follow this architecture exactly

{memory_analysis}

THE #1 OPTIMIZATION: Single-pass register tiling.
The naive kernel writes residual_out to HBM in Phase 1, then re-reads it in Phase 2.
This wastes {hidden * rows * 2:,} bytes of HBM bandwidth.
Keep the add results in REGISTERS across phases — never write then re-read.

### Required kernel structure (pseudo-code):
```
#define HIDDEN {hidden}
#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD {items_per_thread}  // {hidden}/256

__global__ void kernel(input, residual, weight, residual_out, quant_out, scales) {{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * HIDDEN;

    // ── STEP 1: Load + add into REGISTERS (NOT global memory) ──
    float vals[ITEMS_PER_THREAD];
    float local_ss = 0.0f;

    // Use uint4 loads: 8 bf16 per 128-bit load (2 loads = 16 bf16 = 8 float vals)
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {{
        int idx = tid + i * BLOCK_SIZE;  // strided for coalescing
        float a = __bfloat162float(input[base + idx]);
        float r = __bfloat162float(residual[base + idx]);
        vals[i] = a + r;
        local_ss += vals[i] * vals[i];
        residual_out[base + idx] = __float2bfloat16(vals[i]);  // write once, never re-read
    }}

    // ── STEP 2: Warp shuffle reduction (NOT shared memory tree) ──
    // Intra-warp: __shfl_xor_sync (5 steps, 0 shared memory, 0 barriers)
    for (int mask = 16; mask > 0; mask >>= 1)
        local_ss += __shfl_xor_sync(0xffffffff, local_ss, mask);
    // Cross-warp: only 8 warps, use tiny shared memory
    __shared__ float warp_sums[8];
    if (tid % 32 == 0) warp_sums[tid / 32] = local_ss;
    __syncthreads();
    if (tid < 8) {{
        float v = warp_sums[tid];
        for (int mask = 4; mask > 0; mask >>= 1)
            v += __shfl_xor_sync(0xff, v, mask);
        if (tid == 0) warp_sums[0] = v;
    }}
    __syncthreads();
    float rms_inv = rsqrtf(warp_sums[0] / HIDDEN + 1e-6f);

    // ── STEP 3: Normalize FROM REGISTERS + quantize ──
    // vals[] still holds the add results — NO re-read from HBM!
    // Process in 16-element quant blocks
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {{
        int idx = tid + i * BLOCK_SIZE;
        float w = __bfloat162float(weight[idx]);  // weight is only 4KB, L2-cached
        vals[i] = vals[i] * rms_inv * w;          // normalize from register
    }}
    // ... then quantize vals[] to FP4 with absmax per 16-element block ...
    // ... pack into bytes, write quant_out and scales ...
}}
```

### Key points:
1. vals[{items_per_thread}] stays in registers across Phase 1 → Phase 2 (only {items_per_thread} floats = {items_per_thread * 4} bytes/thread)
2. residual_out is WRITTEN ONCE, NEVER RE-READ — saves {hidden * rows * 2 // 1024} KB of HBM traffic
3. Warp shuffle reduction: 5 __shfl_xor + 1 __syncthreads (vs 8 __syncthreads in naive)
4. Weight vector ({hidden} bf16 = {hidden * 2 // 1024} KB) stays in L2 cache — negligible cost
5. Use #pragma unroll on all loops with known trip count
6. Use __launch_bounds__(256, 4) to target 32-48 registers for max occupancy
7. The quantization loop should process each thread's {items_per_thread} normalized values
"""

    elif kernel_type == "silu_mul":
        if len(problem_shape) == 3:
            b, m, k = problem_shape
            n = b * m * k
        else:
            n = problem_shape[0]
        return f"""\
## STRUCTURAL BLUEPRINT — Follow this architecture exactly

{memory_analysis}

### Required kernel structure:
```
#define QUANT_BLOCK 16
#define BLOCKS_PER_THREAD 4  // thread coarsening: 4 quant blocks per thread

__global__ __launch_bounds__(256, 4)
void kernel(gate, up, quant_out, scales, int N) {{
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int qb_start = tid_global * BLOCKS_PER_THREAD;

    #pragma unroll
    for (int b = 0; b < BLOCKS_PER_THREAD; b++) {{
        int qb = qb_start + b;
        if (qb >= N / QUANT_BLOCK) return;
        int elem_base = qb * QUANT_BLOCK;

        // Vectorized load: 2x uint4 = 16 bf16 values for gate, same for up
        // (128-bit loads, 8 bf16 per load)
        float vals[16];
        // Load gate[elem_base..+16] via uint4, convert to float, apply SiLU
        // Load up[elem_base..+16] via uint4, multiply with SiLU result
        for (int j = 0; j < 16; j++) {{
            float g = __bfloat162float(gate[elem_base + j]);
            float u = __bfloat162float(up[elem_base + j]);
            float silu = g * __frcp_rn(1.0f + __expf(-g));  // fast math SiLU
            vals[j] = silu * u;
        }}

        // Absmax, scale, quantize, pack (same as naive but use vectorized stores)
        // Pack 8 bytes into uint2, store once
    }}
}}
```

### Key points:
1. Thread coarsening (4 quant blocks/thread) — amortizes launch overhead
2. Fast math: __expf + __frcp_rn for SiLU (~4 cycles vs ~20)
3. Vectorized loads: uint4 for 8 bf16 values per load instruction
4. Vectorized stores: pack FP4 bytes into uint2, single 64-bit write
5. __launch_bounds__(256, 4) for controlled register budget
6. Grid size = N / (QUANT_BLOCK * BLOCKS_PER_THREAD * 256)
"""

    elif kernel_type == "nvfp4_quantize":
        if len(problem_shape) == 2:
            m, k = problem_shape
            n = m * k
        else:
            n = problem_shape[0]
        return f"""\
## STRUCTURAL BLUEPRINT — Follow this architecture exactly

{memory_analysis}

### Required kernel structure:
```
#define QUANT_BLOCK 16
#define BLOCKS_PER_THREAD 4  // thread coarsening

__global__ __launch_bounds__(256, 4)
void kernel(input, packed_out, scales, int N) {{
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    int qb_start = tid_global * BLOCKS_PER_THREAD;

    #pragma unroll
    for (int b = 0; b < BLOCKS_PER_THREAD; b++) {{
        int qb = qb_start + b;
        if (qb >= N / QUANT_BLOCK) return;
        int elem_base = qb * QUANT_BLOCK;

        // Vectorized load: 2x uint4 = 16 bf16 values (128-bit each)
        float x[16];
        // Load input[elem_base..+16] via uint4, convert bf16->float

        // Absmax, E4M3 scale, quantize, pack into uint2, vectorized store
    }}
}}
```

### Key points:
1. Thread coarsening (4 quant blocks/thread)
2. Vectorized loads: uint4 for 8 bf16 values per transaction
3. Vectorized stores: pack 8 FP4 bytes into uint2, single write
4. __launch_bounds__(256, 4) for register budget control
5. #pragma unroll on inner loops
6. Grid = N / (QUANT_BLOCK * BLOCKS_PER_THREAD * 256)
"""

    return ""


def get_roofline_feedback(kernel_type: str, problem_shape: tuple,
                          timing_us: float, speedup: float) -> str:
    """Calculate roofline efficiency for profiler feedback."""
    if kernel_type == "add_rmsnorm":
        rows, hidden = problem_shape
        n = rows * hidden
        # Minimum bytes (single-pass, no re-read)
        min_bytes = n * 2 + n * 2 + hidden * 2 + n * 2 + n // 2 + n // 16
        # Naive bytes (with re-read)
        naive_bytes = min_bytes + n * 2
    elif kernel_type == "silu_mul":
        if len(problem_shape) == 3:
            n = problem_shape[0] * problem_shape[1] * problem_shape[2]
        else:
            n = problem_shape[0]
        min_bytes = n * 2 * 2 + n // 2 + n // 16
        naive_bytes = min_bytes
    elif kernel_type == "nvfp4_quantize":
        if len(problem_shape) == 2:
            n = problem_shape[0] * problem_shape[1]
        else:
            n = problem_shape[0]
        min_bytes = n * 2 + n // 2 + n // 16
        naive_bytes = min_bytes
    else:
        return ""

    if timing_us <= 0:
        return ""

    achieved_bw = naive_bytes / timing_us / 1e6  # TB/s
    peak_bw = 8.0  # TB/s for B200
    bw_util = achieved_bw / peak_bw * 100
    theoretical_min = min_bytes / (peak_bw * 1e6)
    realistic_min = min_bytes / (4.0 * 1e6)  # ~50% of peak for small kernels
    efficiency = theoretical_min / timing_us * 100

    lines = [
        f"Roofline: {achieved_bw:.2f} TB/s achieved ({bw_util:.0f}% of 8 TB/s peak HBM)",
        f"  Theoretical min (peak BW):     {theoretical_min:.3f} us",
        f"  Realistic min (~50% peak):     {realistic_min:.3f} us",
        f"  Your timing:                   {timing_us:.3f} us",
        f"  Roofline efficiency:           {efficiency:.0f}%",
    ]

    if kernel_type == "add_rmsnorm" and timing_us > realistic_min * 1.5:
        lines.append(
            f"  ** You are {timing_us / realistic_min:.1f}x above realistic minimum. "
            f"The #1 fix: eliminate the residual_out re-read (keep values in registers)."
        )

    return "\n".join(lines)
