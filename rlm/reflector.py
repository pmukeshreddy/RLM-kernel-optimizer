"""
reflector.py — Reflection + reward system for the RLM kernel-optimization pipeline.

After each profiling round, computes a numerical reward score and generates a
reflection prompt with real profiler data for the next LLM attempt.

3 templates only:
  1. Compile failure  → show error, fix it
  2. Correctness failure → show code, fix the math
  3. Performance → show reward score + profiler data, make it faster

Reward scoring:
  +20  if kernel compiles
  +100 if correctness passes
  +(baseline_us / optimized_us) × 100  for speedup (e.g. 1.5x = 150 pts)
"""

from __future__ import annotations

import logging
from textwrap import dedent

logger = logging.getLogger(__name__)


# ── Reward computation ────────────────────────────────────────────────────────

def compute_reward(compile_ok: bool, correct: bool, speedup: float) -> tuple[float, str]:
    """Compute numerical reward score.
    Returns (total_score, breakdown_string)."""
    score = 0.0
    parts = []

    if compile_ok:
        score += 20
        parts.append("compile: +20")
    else:
        parts.append("compile: +0 (FAILED)")
        return score, " | ".join(parts)

    if correct:
        score += 100
        parts.append("correctness: +100")
    else:
        parts.append("correctness: +0 (FAILED)")
        return score, " | ".join(parts)

    perf_score = speedup * 100
    score += perf_score
    parts.append(f"speedup: +{perf_score:.0f} ({speedup:.3f}x)")

    return score, " | ".join(parts)


# ── Reflection templates ─────────────────────────────────────────────────────

COMPILE_REFLECTION = dedent("""\
    ## Reflection -- Iteration {iteration}: Compilation Failure

    **Reward: {reward:.0f}** ({reward_breakdown})

    Your previous solution **failed to compile**.

    ### Compiler error
    ```
    {error}
    ```

    ### Your previous solution
    ```cuda
    {solution}
    ```

    ### What to fix
    - Read the compiler error above carefully and fix the exact issue.
    - Do NOT invent header files (e.g. "fp4_utils.cuh") -- only use the original #includes.
    - Only call functions defined in the included headers -- do NOT invent helpers.
    - If you see "undefined reference" to the launch function, you changed its signature.
    {hints}

    ### Instructions
    Write a corrected kernel that compiles successfully.
    Return the COMPLETE .cu file in a single ```cuda code block.
""")


CORRECTNESS_REFLECTION = dedent("""\
    ## Reflection -- Iteration {iteration}: Correctness Failure

    **Reward: {reward:.0f}** ({reward_breakdown})

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


PERFORMANCE_REFLECTION = dedent("""\
    ## Reflection -- Iteration {iteration}

    **Reward: {reward:.0f}** ({reward_breakdown})

    ### Timing
    - Baseline: {baseline_us:.3f} us
    - Your solution: {optimized_us:.3f} us
    - Speedup: {speedup:.3f}x

    ### Your previous solution
    ```cuda
    {solution}
    ```
    {profile_section}
    {delta_section}

    ### How to improve
    - Study the profiler data above. Use it to guide your next optimization.
    - Higher reward = better. Maximize speedup while keeping correctness.
    {hints}

    ### Instructions
    Write a faster kernel. Correctness must still pass (atol < 1e-2).
    Return the COMPLETE .cu file in a single ```cuda code block.
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
    """Format REAL profiler data only: timing, occupancy, compiler metrics, SASS."""
    if not metrics:
        return ""

    lines = [f"\n### Profiler Data (Iteration {iteration})"]
    lines.append("```")

    # Real runtime metrics
    occupancy = metrics.get("sm_occupancy", 0)
    duration = metrics.get("duration_us", 0)
    speedup = metrics.get("speedup", 1.0)

    lines.append(f"Kernel timing:                 {duration:.3f} us")
    lines.append(f"Speedup:                       {speedup:.3f}x")
    lines.append(f"SM occupancy:                  {occupancy:.1f}%")

    # Compiler metrics (from nvcc -Xptxas -v)
    cm = metrics.get("_compiler", {})
    if cm:
        regs = cm.get("registers_per_thread", 0)
        spill_total = cm.get("spill_stores_bytes", 0) + cm.get("spill_loads_bytes", 0)
        smem = cm.get("static_smem_bytes", 0)

        lines.append("")
        lines.append(f"Registers per thread:          {regs}")
        lines.append(f"Register spills:               {spill_total} bytes{' *** SPILLING ***' if spill_total > 0 else ''}")
        lines.append(f"Shared memory:                 {smem} bytes")

        # SASS instruction breakdown (from cuobjdump -sass)
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
    """Format round-over-round deltas using only real measured data."""
    if not previous or not current:
        return ""

    lines = ["\n### Changes vs Previous Round"]
    lines.append("```")

    # Timing delta
    cur_timing = current.get("duration_us", 0)
    prev_timing = previous.get("duration_us", 0)
    if cur_timing > 0 and prev_timing > 0:
        delta_us = cur_timing - prev_timing
        arrow = "v" if delta_us < 0 else "^" if delta_us > 0 else "="
        better = "  (faster)" if delta_us < 0 else "  (slower)" if delta_us > 0 else ""
        lines.append(f"{'Kernel timing (us)':30s}  {prev_timing:9.3f} -> {cur_timing:9.3f}  ({arrow} {abs(delta_us):.3f}){better}")

    # Speedup delta
    cur_spd = current.get("speedup", 1.0)
    prev_spd = previous.get("speedup", 1.0)
    if cur_spd != prev_spd:
        lines.append(f"{'Speedup':30s}  {prev_spd:9.3f}x -> {cur_spd:9.3f}x")

    # Occupancy delta
    cur_occ = current.get("sm_occupancy", 0)
    prev_occ = previous.get("sm_occupancy", 0)
    if cur_occ != prev_occ and (cur_occ > 0 or prev_occ > 0):
        delta = cur_occ - prev_occ
        arrow = "^" if delta > 0 else "v" if delta < 0 else "="
        lines.append(f"{'SM occupancy':30s}  {prev_occ:6.1f}% -> {cur_occ:6.1f}%  ({arrow} {abs(delta):.1f})")

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

        # Vectorization delta
        cur_ldg128 = cur_cm.get("sass_ldg_128", 0)
        prev_ldg128 = prev_cm.get("sass_ldg_128", 0)
        cur_ldg_total = cur_cm.get("sass_ldg_32", 0) + cur_cm.get("sass_ldg_64", 0) + cur_ldg128
        prev_ldg_total = prev_cm.get("sass_ldg_32", 0) + prev_cm.get("sass_ldg_64", 0) + prev_ldg128
        cur_vec = (cur_ldg128 / cur_ldg_total * 100) if cur_ldg_total > 0 else 0
        prev_vec = (prev_ldg128 / prev_ldg_total * 100) if prev_ldg_total > 0 else 0
        if cur_vec != prev_vec and (cur_ldg_total > 0 or prev_ldg_total > 0):
            lines.append(f"{'Load vectorization':30s}  {prev_vec:5.0f}%  -> {cur_vec:5.0f}%")

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


# ── Launch function signatures (must match harness exactly) ──────────────────

def _get_launch_signature(kernel_type: str) -> str:
    """Return the EXACT launch function signature the harness expects.
    If the kernel changes this, it gets 'undefined reference' linker errors."""
    sigs = {
        "add_rmsnorm": dedent("""\
            ### Required Launch Function (DO NOT CHANGE)
            The benchmark harness calls this exact function. Your code MUST define it with this exact signature:
            ```c
            void launch_fused_add_rmsnorm_nvfp4(
                const __nv_bfloat16* input,
                const __nv_bfloat16* residual,
                const __nv_bfloat16* weight,
                __nv_bfloat16* residual_out,
                unsigned char* quant_out,
                __nv_fp8_storage_t* scale_out,
                int rows, int hidden,
                cudaStream_t stream);
            ```
            - `unsigned char*` for quant_out, NOT `uint8_t*`
            - `__nv_fp8_storage_t*` for scale_out
            - Exactly 9 parameters in this order
        """),
        "silu_mul": dedent("""\
            ### Required Launch Function (DO NOT CHANGE)
            The benchmark harness calls this exact function. Your code MUST define it with this exact signature:
            ```c
            void launch_silu_mul_fp4quant(
                const __nv_bfloat16* gate,
                const __nv_bfloat16* up,
                uint8_t* quant_out,
                __nv_fp8_storage_t* scale_out,
                int N,
                cudaStream_t stream);
            ```
            - Exactly 6 parameters in this order
        """),
        "nvfp4_quantize": dedent("""\
            ### Required Launch Function (DO NOT CHANGE)
            The benchmark harness calls this exact function. Your code MUST define it with this exact signature:
            ```c
            void launch_nvfp4_quantize_bf16(
                const __nv_bfloat16* input,
                uint8_t* packed_out,
                __nv_fp8_storage_t* scale_out,
                int N,
                cudaStream_t stream);
            ```
            - Exactly 5 parameters in this order
        """),
    }
    return sigs.get(kernel_type, "")


# ── Critical rules footer ────────────────────────────────────────────────────

CRITICAL_RULES = dedent("""\
    ### Critical Rules
    1. Return the COMPLETE .cu file in a single ```cuda code block.
    2. Keep all #includes (use the original #include directives, NOT expanded content).
    3. Do NOT use torch headers (torch/extension.h, ATen, c10) -- this is standalone CUDA.
    4. Keep ALL kernel functions and the launch_* wrapper function.
    5. The launch_* function signature MUST match the "Required Launch Function" section EXACTLY.
       If you change it, the code will fail with "undefined reference" linker errors.
    6. Output must match reference within atol=1e-2.
    7. No explanations -- just the code block.
    8. Only call functions defined in the included headers. Do NOT invent helper functions.
""")


# ── Main reflection dispatcher ───────────────────────────────────────────────

def reflect(
    candidate,
    iteration: int,
    hw_spec: dict,
    kernel_type: str = "",
    baseline_us: float = 0.0,
    prev_metrics: dict = None,
    atol: float = 1e-2,
    **kwargs,  # absorb unused args (target_speedup, min_speedup) from callers
) -> str:
    """Generate a reflection prompt based on the candidate's result.

    3 paths:
      1. compile_ok=False → COMPILE_REFLECTION + compiler error
      2. correct=False    → CORRECTNESS_REFLECTION
      3. otherwise        → PERFORMANCE_REFLECTION + reward score + profiler data
    """
    solution = candidate.code or "# (no code generated)"
    metrics = candidate.metrics or {}
    speedup = candidate.speedup
    hints = _get_hints(kernel_type)
    hw_context = _build_hw_context(hw_spec)
    launch_sig = _get_launch_signature(kernel_type)
    profile_section = _format_profile_section(metrics, iteration)
    delta_section = _format_delta_section(metrics, prev_metrics)

    reward, reward_breakdown = compute_reward(
        candidate.compile_ok, candidate.correct, speedup
    )

    # Common footer: launch signature + hw context + rules
    footer = "\n" + launch_sig + "\n" + hw_context + "\n" + CRITICAL_RULES

    if not candidate.compile_ok:
        error = getattr(candidate, 'compile_error', '') or "Unknown compilation error"
        prompt = COMPILE_REFLECTION.format(
            iteration=iteration,
            reward=reward,
            reward_breakdown=reward_breakdown,
            error=error,
            solution=solution,
            hints=hints,
        )
        return prompt + footer

    if not candidate.correct:
        prompt = CORRECTNESS_REFLECTION.format(
            iteration=iteration,
            reward=reward,
            reward_breakdown=reward_breakdown,
            solution=solution,
            atol=atol,
            hints=hints,
        )
        return prompt + footer

    optimized_us = baseline_us / speedup if speedup > 0 else 0.0
    prompt = PERFORMANCE_REFLECTION.format(
        iteration=iteration,
        reward=reward,
        reward_breakdown=reward_breakdown,
        solution=solution,
        speedup=speedup,
        baseline_us=baseline_us,
        optimized_us=optimized_us,
        profile_section=profile_section,
        delta_section=delta_section,
        hints=hints,
    )
    return prompt + footer
