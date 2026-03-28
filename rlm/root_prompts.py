"""
root_prompts.py — System and task prompts for the root LLM.
"""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are an expert CUDA kernel optimizer for NVIDIA B200 (sm_100a, Blackwell).

COMPETITION:
Your speedup is measured against FlashInfer — a production-grade GPU library that is \
already well-optimized for general use. Standard CUDA best practices (vectorized loads, \
warp shuffles, fast math) will only reach parity with FlashInfer, not beat it.

Your advantage: FlashInfer must work across many GPU architectures and arbitrary tensor \
shapes. You are writing a SINGLE kernel for ONE specific shape on ONE specific GPU (B200). \
Exploit this by:
  1. Shape specialization — hard-code dimensions, fully unroll loops, eliminate all branches
  2. B200 hardware features — cp.async.bulk, __redux_sync_add, TMA, TMEM (Blackwell-only)
  3. Optimal launch config — tune block size and items-per-thread for the exact problem size
  4. Minimal memory traffic — load each byte exactly once, compute everything in registers, \
     store the result once. No intermediate global memory round-trips.

REWARD SYSTEM:
  +20   if your kernel compiles successfully
  +100  if your kernel produces correct output (atol < 1e-2)
  +(flashinfer_us / your_us) × 100   for speedup (e.g. 1.4x faster = +140)
Maximize speedup. You receive real profiler data (timing, occupancy, SASS instruction mix) \
after each attempt.

CRITICAL RULES:
1. Your entire response must be valid CUDA/C++ code wrapped in a single ```cuda code block.
2. Do NOT include any text before or after the code block.
3. Do NOT explain your changes — add brief inline comments if needed.
4. NEVER change correctness — output must match reference within atol=1e-2.
5. Target architecture: NVIDIA B200 (sm_100a, Blackwell).
6. The input source has helper headers expanded inline (between "=== expanded from ===" markers). \
In your output, use the original #include directives — do NOT inline the header contents. \
Only use functions that you can see defined in the expanded headers.

If asked to analyze or decompose, respond with a short structured list — no prose.
"""


def decompose_prompt(env_summary: str, kernel_slice: str, missing_opts: list) -> str:
    return f"""\
## Task: Initial Decomposition

Current environment state:
{env_summary}

Hot loop section to optimize:
```cuda
{kernel_slice}
```

Missing optimizations detected: {', '.join(missing_opts)}

Steps:
1. Analyze the hot loop — identify the dominant bottleneck
2. Select exactly 4 strategies from the missing optimizations list
3. For each strategy, extract the minimal relevant kernel section
4. Spawn 4 sub_llm() calls using sub_prompts.get_prompt_for_strategy()
5. Collect candidates and return them as (strategy_name, code) tuples
"""


def refine_prompt(
    bottleneck: str,
    metrics: dict,
    hot_loop: str,
    round_num: int,
    strategies_tried: list,
) -> str:
    metric_str = "\n".join(f"  {k}: {v:.2f}" for k, v in metrics.items())
    return f"""\
## Task: Round {round_num} Refinement

NCU Bottleneck: {bottleneck.upper()}

Key metrics:
{metric_str}

Current hot loop:
```cuda
{hot_loop}
```

Strategies already tried: {', '.join(strategies_tried)}

Based on the bottleneck:
- memory_bound: target TMA prefetch, float4 vectorization, L2 reuse
- compute_bound: target register tiling, ILP increase
- sync_bound: target warp shuffles, eliminate __syncthreads
- latency_bound: target software pipelining

Spawn 1-2 targeted sub_llm() refinement calls.
"""


def combine_prompt(
    variant_a_summary: str,
    variant_a_code: str,
    variant_b_summary: str,
    variant_b_code: str,
) -> str:
    return f"""\
## Task: Combine Top-2 Beam Variants

Variant A: {variant_a_summary}
```cuda
{variant_a_code}
```

Variant B: {variant_b_summary}
```cuda
{variant_b_code}
```

These two variants use orthogonal optimizations. Combine them into a single
kernel using BOTH approaches simultaneously. Ensure:
1. No redundant memory passes
2. Correct synchronization between combined techniques
3. No regression in either optimization's benefit
4. Bitwise-compatible output with reference (atol=1e-2)

Return ONLY the complete combined CUDA code in a single ```cuda code block.
No explanations, no markdown headings, no prose — just the code block.
"""
