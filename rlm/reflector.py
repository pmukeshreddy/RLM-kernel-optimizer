"""
reflector.py — Reflection / iteration logic for the RLM kernel-optimization pipeline.

After each profiling round, analyses the KernelCandidate and generates a structured
reflection prompt that feeds back into the LLM for the next attempt.

Modelled on AMD Apex's reflector pattern: separate templates per failure mode,
kernel-type-specific hints, dual speedup thresholds, raw profiler data as code blocks.
"""

from __future__ import annotations

import logging
from textwrap import dedent

logger = logging.getLogger(__name__)


# ── Reflection templates ─────────────────────────────────────────────────────

COMPILE_REFLECTION = dedent("""\
    ## Reflection -- Iteration {iteration}: Compilation Failure

    Your previous solution **failed to compile**.

    ### Error context
    - Strategy: {strategy}
    - Common failure modes: missing #include, undefined helper functions,
      changed launch_* signature, syntax errors in template/macro code.

    ### Your previous solution
    ```cuda
    {solution}
    ```

    ### What to fix
    - Check for syntax errors and missing #include directives.
    - Only call functions defined in the included headers -- do NOT invent helpers.
    - The launch_* wrapper signature must not change.
    - Verify template parameters and CUDA type casts are correct.
    {hints}

    ### Instructions
    Write a corrected kernel that compiles successfully.
    Return the COMPLETE .cu file in a single ```cuda code block.
""")


CORRECTNESS_REFLECTION = dedent("""\
    ## Reflection -- Iteration {iteration}: Correctness Failure

    Your solution compiled but **produced wrong results** (atol > {atol}).

    ### Your previous solution
    ```cuda
    {solution}
    ```

    ### What to fix
    - Check index calculations carefully (off-by-one, stride, base offset errors).
    - Verify vectorized loads/stores unpack in the correct element order.
    - Ensure reductions accumulate ALL elements -- no partial sums lost at boundaries.
    - Check NVFP4 quantization block alignment (blocks of 16 elements).
    - Verify bf16 <-> float conversions round correctly.
    {hints}

    ### Instructions
    Fix the correctness bug while keeping the optimization approach.
    Return the COMPLETE .cu file in a single ```cuda code block.
""")


PERF_REGRESSION_REFLECTION = dedent("""\
    ## Reflection -- Iteration {iteration}: Performance Regression

    Your solution is correct but **slower than baseline** ({speedup:.2f}x).

    ### Speedup thresholds
    | Threshold | Value | Status |
    |-----------|-------|--------|
    | Baseline  | 1.00x | REGRESSED |
    | Integration minimum | {min_speedup:.2f}x | NOT MET |
    | Target | {target_speedup:.1f}x | NOT MET |

    ### Timing
    - Baseline: {baseline_us:.3f} us
    - Your solution: {optimized_us:.3f} us
    - Speedup: {speedup:.2f}x

    ### Your previous solution
    ```cuda
    {solution}
    ```
    {profile_section}

    ### What to fix
    - Your optimization made things worse. Check for:
      * Unnecessary global memory round-trips (writing then re-reading)
      * Excessive __syncthreads() barriers
      * Register spills from too many local variables
      * Uncoalesced memory access patterns
    - Consider reverting to a simpler approach and optimizing incrementally.
    {hints}

    ### Instructions
    Write an optimized kernel that is faster than the baseline.
    Correctness must still pass. Return the COMPLETE .cu file in a single ```cuda code block.
""")


BELOW_THRESHOLD_REFLECTION = dedent("""\
    ## Reflection -- Iteration {iteration}: Below Target

    Your solution is correct with {speedup:.2f}x speedup -- progress, but not enough.

    ### Speedup thresholds
    | Threshold | Value | Status |
    |-----------|-------|--------|
    | Baseline  | 1.00x | PASSED |
    | Integration minimum | {min_speedup:.2f}x | {min_status} |
    | Target | {target_speedup:.1f}x | NOT MET |

    ### Timing
    - Baseline: {baseline_us:.3f} us
    - Your solution: {optimized_us:.3f} us
    - Speedup: {speedup:.2f}x (need >= {target_speedup:.1f}x)

    ### Your previous solution
    ```cuda
    {solution}
    ```
    {profile_section}
    {delta_section}

    ### What to fix
    - You need a more aggressive approach. Consider:
      * Fuse ALL passes into a single pass over the data
      * Cache intermediate values in registers, not global memory
      * Use 128-bit vectorized loads/stores (uint4 for 8 bf16 values)
      * Replace shared memory reductions with warp shuffles
      * Increase per-thread work (each thread handles 16+ elements)
    {hints}

    ### Instructions
    Write a significantly faster kernel targeting {target_speedup:.1f}x+ speedup.
    Correctness must still pass. Return the COMPLETE .cu file in a single ```cuda code block.
""")


IMPROVEMENT_REFLECTION = dedent("""\
    ## Reflection -- Iteration {iteration}: Good Progress

    Your solution achieves **{speedup:.2f}x speedup** -- solid work.

    ### Speedup thresholds
    | Threshold | Value | Status |
    |-----------|-------|--------|
    | Baseline  | 1.00x | PASSED |
    | Integration minimum | {min_speedup:.2f}x | PASSED |
    | Target | {target_speedup:.1f}x | {target_status} |

    ### Timing
    - Baseline: {baseline_us:.3f} us
    - Your solution: {optimized_us:.3f} us
    - Speedup: {speedup:.2f}x
    {profile_section}
    {delta_section}

    ### Push further
    - Current: {speedup:.2f}x -> target: {target_speedup:.1f}x+
    - Look at the profiler data above for the remaining bottleneck.
    - Consider combining multiple techniques (vectorized loads + warp shuffles + register caching).
    - Reduce total instruction count -- simpler code often runs faster on GPUs.
    {hints}

    ### Instructions
    Write an improved kernel pushing closer to {target_speedup:.1f}x+ speedup.
    Correctness must still pass. Return the COMPLETE .cu file in a single ```cuda code block.
""")


# ── Kernel-type-specific hints ───────────────────────────────────────────────

def _get_hints(kernel_type: str) -> str:
    hints_map = {
        "add_rmsnorm": (
            "- add+RMSNorm+quantize is memory-bound: maximize bandwidth utilization.\n"
            "- Fuse add, norm, and FP4 quantize into ONE pass -- avoid writing residual_out then re-reading it.\n"
            "- Use uint4 loads for 8 bf16 values per 128-bit transaction.\n"
            "- Use warp shuffles (__shfl_xor_sync) for the sum-of-squares reduction.\n"
            "- Each thread should own one NVFP4 block (16 elements) and cache in registers."
        ),
        "silu_mul": (
            "- SiLU(x)*y is pure elementwise -- should run at near memory bandwidth.\n"
            "- Use vectorized loads (uint4 = 8 bf16) and stores.\n"
            "- Use __expf() instead of expf() for the sigmoid (fast math SFU).\n"
            "- Fuse SiLU activation with the FP4 quantization in one pass.\n"
            "- Thread coarsening: each thread processes multiple elements."
        ),
        "nvfp4_quantize": (
            "- Quantize-only is bandwidth-bound: read once, write packed FP4.\n"
            "- Use 128-bit vectorized loads for input bf16 data.\n"
            "- Pack 8 FP4 pairs into uint64_t for a single 8-byte coalesced store.\n"
            "- Each thread handles one 16-element quantization block.\n"
            "- Minimize register pressure -- keep the quantization logic simple."
        ),
    }
    return hints_map.get(kernel_type, "")


# ── Hardware context builder ─────────────────────────────────────────────────

def _build_hw_context(hw_spec: dict) -> str:
    mem = hw_spec.get("memory", {})
    sm = hw_spec.get("sm", {})
    name = hw_spec.get("hardware", {}).get("name", "GPU")
    bw_tbs = mem.get("hbm_bandwidth_tbs", 8.0)
    sm_count = sm.get("count", 142)
    smem_kb = mem.get("shared_memory_per_sm_kb", 228)
    max_threads = sm.get("max_threads_per_sm", 2048)
    warp_size = sm.get("warp_size", 32)

    return dedent(f"""\
        ### {name} Hardware
        - HBM bandwidth: {bw_tbs:.1f} TB/s
        - {sm_count} SMs, {smem_kb} KB shared memory per SM, 255 registers per thread
        - Warp size: {warp_size}, max {max_threads} threads per SM
        - 128-bit load/store transactions (uint4 = 8 bf16 values)
        - Fast math SFU: __expf ~4 cycles vs expf ~20 cycles
        - Warp shuffle: __shfl_xor_sync ~2 cycles vs shared mem reduction ~10+ cycles
    """)


# ── Profile data formatter ───────────────────────────────────────────────────

def _format_profile_section(metrics: dict, iteration: int) -> str:
    if not metrics:
        return ""

    lines = [f"\n### Profiler Data (Iteration {iteration})"]
    lines.append("```")

    # Runtime metrics
    mem_pct = metrics.get("mem_throughput_pct", 0)
    compute_pct = metrics.get("compute_throughput_pct", 0)
    occupancy = metrics.get("sm_occupancy", 0)
    stall_mem = metrics.get("stall_memory", 0)
    stall_bar = metrics.get("stall_barrier", 0)
    l2_hit = metrics.get("l2_hit_rate", 0)

    lines.append(f"Memory bandwidth utilization:  {mem_pct:.1f}%")
    lines.append(f"Compute utilization:           {compute_pct:.1f}%")
    lines.append(f"SM occupancy:                  {occupancy:.1f}%")
    lines.append(f"Memory stall rate:             {stall_mem:.1f}%")
    lines.append(f"Barrier stall rate:            {stall_bar:.1f}%")
    lines.append(f"L2 cache hit rate:             {l2_hit:.1f}%")

    # Compiler metrics
    cm = metrics.get("_compiler", {})
    if cm:
        regs = cm.get("registers_per_thread", 0)
        spill_total = cm.get("spill_stores_bytes", 0) + cm.get("spill_loads_bytes", 0)
        smem = cm.get("static_smem_bytes", 0)

        lines.append("")
        lines.append(f"Registers per thread:          {regs}")
        lines.append(f"Register spills:               {spill_total} bytes{' *** SPILLING ***' if spill_total > 0 else ''}")
        lines.append(f"Shared memory:                 {smem} bytes")

        # SASS instruction breakdown
        sass_total = cm.get("sass_total_instructions", 0)
        if sass_total > 0:
            ldg_128 = cm.get("sass_ldg_128", 0)
            ldg_64 = cm.get("sass_ldg_64", 0)
            ldg_32 = cm.get("sass_ldg_32", 0)
            stg_128 = cm.get("sass_stg_128", 0)
            stg_32 = cm.get("sass_stg_32", 0)
            ldl = cm.get("sass_ldl", 0)
            stl = cm.get("sass_stl", 0)
            ffma = cm.get("sass_ffma", 0)
            hfma2 = cm.get("sass_hfma2", 0)
            mufu = cm.get("sass_mufu", 0)
            bar = cm.get("sass_bar", 0)
            shfl = cm.get("sass_shfl", 0)

            total_ldg = ldg_32 + ldg_64 + ldg_128
            vec_pct = (ldg_128 / total_ldg * 100) if total_ldg > 0 else 0

            lines.append("")
            lines.append(f"SASS total instructions:       {sass_total}")
            lines.append(f"Global loads:                  {total_ldg}  (128-bit: {ldg_128}, 64-bit: {ldg_64}, 32-bit: {ldg_32})  vectorization: {vec_pct:.0f}%")
            lines.append(f"Global stores:                 {stg_32 + stg_128}  (128-bit: {stg_128}, 32-bit: {stg_32})")
            lines.append(f"Spill load/store instructions: {ldl + stl}{' *** REGISTER SPILLS IN BINARY ***' if ldl + stl > 0 else ''}")
            lines.append(f"Compute: FFMA={ffma}  HFMA2={hfma2}  MUFU(SFU)={mufu}")
            lines.append(f"Sync: barriers={bar}  shuffles={shfl}")

    lines.append("```")
    return "\n".join(lines)


# ── Round-over-round delta ───────────────────────────────────────────────────

def _format_delta_section(current: dict, previous: dict) -> str:
    if not previous or not current:
        return ""

    lines = ["\n### Changes vs Previous Round"]
    lines.append("```")

    _DELTA_KEYS = [
        ("mem_throughput_pct", "Memory bandwidth"),
        ("compute_throughput_pct", "Compute utilization"),
        ("sm_occupancy", "SM occupancy"),
        ("stall_memory", "Memory stalls"),
        ("stall_barrier", "Barrier stalls"),
    ]

    for key, label in _DELTA_KEYS:
        cur = current.get(key, 0)
        prev = previous.get(key, 0)
        if prev == 0 and cur == 0:
            continue
        delta = cur - prev
        arrow = "^" if delta > 0 else "v" if delta < 0 else "="
        # For stalls, down is good; for utilization, up is good
        lines.append(f"{label:30s}  {prev:6.1f}% -> {cur:6.1f}%  ({arrow} {abs(delta):.1f})")

    # Compiler metric deltas
    cur_cm = current.get("_compiler", {})
    prev_cm = previous.get("_compiler", {})
    if cur_cm and prev_cm:
        cur_regs = cur_cm.get("registers_per_thread", 0)
        prev_regs = prev_cm.get("registers_per_thread", 0)
        if cur_regs != prev_regs:
            lines.append(f"{'Registers/thread':30s}  {prev_regs:6d}  -> {cur_regs:6d}")

        cur_spills = cur_cm.get("spill_stores_bytes", 0) + cur_cm.get("spill_loads_bytes", 0)
        prev_spills = prev_cm.get("spill_stores_bytes", 0) + prev_cm.get("spill_loads_bytes", 0)
        if cur_spills != prev_spills:
            warn = "  *** REGRESSED ***" if cur_spills > prev_spills else "  (improved)" if cur_spills < prev_spills else ""
            lines.append(f"{'Register spill bytes':30s}  {prev_spills:6d}  -> {cur_spills:6d}{warn}")

        cur_sass = cur_cm.get("sass_total_instructions", 0)
        prev_sass = prev_cm.get("sass_total_instructions", 0)
        if cur_sass != prev_sass and cur_sass > 0 and prev_sass > 0:
            lines.append(f"{'SASS instructions':30s}  {prev_sass:6d}  -> {cur_sass:6d}")

        # Spill instruction delta
        cur_spill_insts = cur_cm.get("sass_ldl", 0) + cur_cm.get("sass_stl", 0)
        prev_spill_insts = prev_cm.get("sass_ldl", 0) + prev_cm.get("sass_stl", 0)
        if cur_spill_insts != prev_spill_insts:
            warn = "  *** REGRESSED ***" if cur_spill_insts > prev_spill_insts else "  (improved)"
            lines.append(f"{'Spill instructions':30s}  {prev_spill_insts:6d}  -> {cur_spill_insts:6d}{warn}")

    lines.append("```")

    # Only return if we actually have deltas (more than header + backticks)
    if len(lines) <= 3:
        return ""
    return "\n".join(lines)


# ── Critical rules footer ────────────────────────────────────────────────────

CRITICAL_RULES = dedent("""\
    ### Critical Rules
    1. Return the COMPLETE .cu file in a single ```cuda code block.
    2. Keep all #includes (use the original #include directives, NOT expanded content).
    3. Keep ALL kernel functions and the launch_* wrapper function.
    4. Do NOT change the launch_* function signature.
    5. Output must match reference within atol=1e-2.
    6. No explanations -- just the code block.
    7. Only call functions defined in the included headers. Do NOT invent helper functions.
""")


# ── Main reflection dispatcher ───────────────────────────────────────────────

def reflect(
    candidate,
    iteration: int,
    hw_spec: dict,
    kernel_type: str = "",
    target_speedup: float = 1.5,
    min_speedup: float = 1.05,
    baseline_us: float = 0.0,
    prev_metrics: dict = None,
    atol: float = 1e-2,
) -> str:
    """Generate a reflection prompt based on the candidate's performance.

    Args:
        candidate: KernelCandidate with compile_ok, correct, speedup, metrics, code.
        iteration: Current refinement round number.
        hw_spec: Hardware specification dict (from b200_spec.yaml).
        kernel_type: Kernel type for type-specific hints.
        target_speedup: Stretch goal speedup.
        min_speedup: Minimum speedup for the result to be considered useful.
        baseline_us: Baseline kernel timing in microseconds.
        prev_metrics: Previous round's metrics dict for delta comparison.
        atol: Correctness tolerance.
    """
    solution = candidate.code or "# (no code generated)"
    metrics = candidate.metrics or {}
    speedup = candidate.speedup
    hints = _get_hints(kernel_type)
    hw_context = _build_hw_context(hw_spec)
    profile_section = _format_profile_section(metrics, iteration)
    delta_section = _format_delta_section(metrics, prev_metrics)

    optimized_us = baseline_us / speedup if speedup > 0 else 0.0

    if not candidate.compile_ok:
        prompt = COMPILE_REFLECTION.format(
            iteration=iteration,
            strategy=candidate.strategy,
            solution=solution,
            hints=hints,
        )
        return prompt + "\n" + hw_context + "\n" + CRITICAL_RULES

    if not candidate.correct:
        prompt = CORRECTNESS_REFLECTION.format(
            iteration=iteration,
            solution=solution,
            atol=atol,
            hints=hints,
        )
        return prompt + "\n" + hw_context + "\n" + CRITICAL_RULES

    if speedup < 1.0:
        prompt = PERF_REGRESSION_REFLECTION.format(
            iteration=iteration,
            solution=solution,
            speedup=speedup,
            baseline_us=baseline_us,
            optimized_us=optimized_us,
            min_speedup=min_speedup,
            target_speedup=target_speedup,
            profile_section=profile_section,
            hints=hints,
        )
        return prompt + "\n" + hw_context + "\n" + CRITICAL_RULES

    if speedup < min_speedup:
        prompt = BELOW_THRESHOLD_REFLECTION.format(
            iteration=iteration,
            solution=solution,
            speedup=speedup,
            baseline_us=baseline_us,
            optimized_us=optimized_us,
            min_speedup=min_speedup,
            min_status="NOT MET",
            target_speedup=target_speedup,
            profile_section=profile_section,
            delta_section=delta_section,
            hints=hints,
        )
        return prompt + "\n" + hw_context + "\n" + CRITICAL_RULES

    # speedup >= min_speedup: good progress, push further
    min_status = "PASSED" if speedup >= min_speedup else "NOT MET"
    target_status = "PASSED" if speedup >= target_speedup else "NOT MET"
    prompt = IMPROVEMENT_REFLECTION.format(
        iteration=iteration,
        solution=solution,
        speedup=speedup,
        baseline_us=baseline_us,
        optimized_us=optimized_us,
        min_speedup=min_speedup,
        target_speedup=target_speedup,
        target_status=target_status,
        profile_section=profile_section,
        delta_section=delta_section,
        hints=hints,
    )
    return prompt + "\n" + hw_context + "\n" + CRITICAL_RULES
