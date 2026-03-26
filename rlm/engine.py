"""
engine.py — RLM core engine.
Orchestrates root LLM decomposition, parallel sub-LLM beam generation, and refinement.
"""

from __future__ import annotations
import asyncio
import logging
import re
from typing import Optional

import anthropic
from anthropic import AsyncAnthropic

from .environment import RLMEnvironment, KernelCandidate
from .root_prompts import SYSTEM_PROMPT, decompose_prompt, refine_prompt, combine_prompt
from .sub_prompts import get_prompt_for_strategy

logger = logging.getLogger(__name__)


class RLMEngine:
    """
    Main orchestrator for the RLM beam search loop.
    Handles: decomposition → beam generation → NCU-guided refinement → combination.
    """

    def __init__(self, env: RLMEnvironment):
        self.env = env
        cfg = env.search_config
        # Sync client for root/combine calls (sequential); async client for parallel beams
        self.client       = anthropic.Anthropic()
        self.async_client = AsyncAnthropic()
        self._loop = None  # persistent event loop for async calls

        self.root_model    = cfg["models"]["root_model"]
        self.sub_model     = cfg["models"]["sub_model"]
        self.combine_model = cfg["models"]["combine_model"]
        self.beam_width    = cfg["beam"]["width"]
        self.refine_rounds = cfg["beam"]["refine_rounds"]
        self.combine_top_k = cfg["beam"]["combine_top_k"]
        self.max_tokens    = cfg["cost_control"]["max_tokens_per_sub_call"]

    # ── Low-level LLM call ────────────────────────────────────────────────────

    def _call_llm(
        self,
        prompt: str,
        model: str,
        system: str = SYSTEM_PROMPT,
        temperature: float = 0.3,
    ) -> tuple:
        if self.env.over_budget():
            raise RuntimeError(
                f"Budget exhausted: ${self.env.total_api_cost_usd:.4f} spent"
            )

        response = self.client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        text       = response.content[0].text
        tokens_in  = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = self.env.record_api_cost(tokens_in, tokens_out, model)
        logger.info(
            "LLM call: model=%s in=%d out=%d cost=$%.4f",
            model, tokens_in, tokens_out, cost,
        )
        return text, tokens_in, tokens_out

    async def _call_llm_async(
        self,
        prompt: str,
        model: str,
        system: str = SYSTEM_PROMPT,
        temperature: float = 0.3,
    ) -> tuple:
        """True async API call — all 4 beam coroutines run concurrently."""
        if self.env.over_budget():
            raise RuntimeError(
                f"Budget exhausted: ${self.env.total_api_cost_usd:.4f} spent"
            )

        response = await self.async_client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        text       = response.content[0].text
        tokens_in  = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = self.env.record_api_cost(tokens_in, tokens_out, model)
        logger.info(
            "LLM call (async): model=%s in=%d out=%d cost=$%.4f",
            model, tokens_in, tokens_out, cost,
        )
        return text, tokens_in, tokens_out

    # ── Round 0: Decomposition ────────────────────────────────────────────────

    def decompose(self) -> list:
        """Freeform strategy selection — LLM proposes optimizations from scratch.

        No predefined menu. The LLM analyzes the kernel and picks the best
        strategies on its own. Returns list of {name, what} dicts for codegen.
        Falls back to kernel-aware defaults if parsing fails.
        """
        import json
        env = self.env
        kernel_src = env.kernel_src  # expanded includes

        prompt = f"""\
You are a CUDA optimization expert. Analyze this kernel and propose exactly {self.beam_width}
optimization techniques that would give the biggest speedup.

You are NOT limited to any predefined list. Propose whatever CUDA optimizations
you think are most impactful for THIS SPECIFIC kernel. Be specific and concrete.

Kernel type: {env.kernel_type}
Target: NVIDIA B200 (Blackwell, sm_100a, 8 TB/s HBM3e, 142 SMs)

```cuda
{kernel_src}
```

For each optimization, give a short name and one-line description of what to do.

Return as a JSON array of objects, most impactful first:
[
  {{"name": "short_name", "what": "one line description of the concrete change"}},
  {{"name": "short_name", "what": "one line description of the concrete change"}},
  {{"name": "short_name", "what": "one line description of the concrete change"}},
  {{"name": "short_name", "what": "one line description of the concrete change"}}
]

Respond with ONLY the JSON array, nothing else."""

        logger.info("Round 0: freeform strategy selection for kernel_type=%s",
                     env.kernel_type)
        response, _, _ = self._call_llm(prompt, model=self.root_model, temperature=0.2)
        logger.debug("Freeform response: %s", response[:300])

        # Parse JSON response
        strategies = []
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                raw = json.loads(json_match.group())
                strategies = [s for s in raw if isinstance(s, dict) and "name" in s]
            except (json.JSONDecodeError, TypeError):
                pass

        if strategies:
            names = [s["name"] for s in strategies[:self.beam_width]]
            logger.info("LLM-proposed strategies: %s", names)
            return strategies[:self.beam_width]

        # Fallback: use kernel-aware defaults from strategy bank
        from search.strategy_bank import select_for_kernel
        fallback = select_for_kernel(
            kernel_type=env.kernel_type, tried=[], beam_width=self.beam_width)
        logger.info("Fallback to kernel-aware strategies: %s", fallback)
        return [{"name": s, "what": ""} for s in fallback]

    # ── Sub-LLM beam generation (parallel) ───────────────────────────────────

    async def _generate_single_beam(
        self,
        strategy,
        kernel_slice: str,
        current_metrics: dict = None,
        round_num: int = 0,
    ) -> KernelCandidate:
        # Support both freeform dicts {name, what} and plain strategy name strings
        if isinstance(strategy, dict):
            strat_name = strategy.get("name", "unknown")
            strat_desc = strategy.get("what", "")
        else:
            strat_name = strategy
            strat_desc = ""

        if strat_desc:
            # Freeform mode: use the LLM's own description as the instruction
            prompt = f"""\
You are an expert CUDA kernel optimizer targeting NVIDIA B200 (sm_100a, Blackwell).

Apply this optimization to the kernel below:

## Optimization: {strat_name}
{strat_desc}

## B200 Hardware (use these numbers to guide your optimization):
- HBM3e: 8 TB/s bandwidth — use float4/uint4 (128-bit) loads to saturate it
- 142 SMs, 228 KB shared memory per SM, 255 registers per thread
- Warp shuffle (__shfl_xor_sync) costs ~2 cycles vs shared mem reduction ~10+ cycles
- Fast math SFU: __expf ~4 cycles vs expf ~20 cycles; __frcp_rn for reciprocal
- For bf16: load 8 values at once via reinterpret_cast<uint4*>, unpack to float for compute

## Full kernel source (headers expanded inline between === markers):
```cuda
{kernel_slice}
```

CRITICAL RULES:
1. Return the COMPLETE .cu file in a single ```cuda code block
2. Keep all #includes (use the original #include directives, NOT the expanded content)
3. Keep ALL kernel functions and the launch_* wrapper function
4. Do NOT change the launch_* function signature
5. Output must match reference within atol=1e-2
6. No explanations — just the code block
7. You may call any function defined in the expanded headers above. Do NOT invent
   helper functions that aren't defined in the headers.
"""
        else:
            # Legacy mode: use strategy bank prompts for refinement rounds
            prompt = get_prompt_for_strategy(
                strategy=strat_name,
                kernel_slice=kernel_slice,
                hw_spec=self.env.hw_spec,
                current_metrics=current_metrics,
            )

        try:
            response, _, _ = await self._call_llm_async(
                prompt, model=self.sub_model, temperature=0.4
            )
        except RuntimeError as e:
            logger.error("Budget exceeded during beam %s: %s", strat_name, e)
            return KernelCandidate(code="", strategy=strat_name, round_num=round_num)

        code = self._extract_cuda_code(response)
        if not code:
            logger.warning("No CUDA code extracted for strategy=%s (response starts: %s)",
                           strat_name, response[:100])
        return KernelCandidate(
            code=code,
            strategy=strat_name,
            round_num=round_num,
            compile_ok=bool(code),
        )

    async def generate_beams(
        self,
        strategies: list,
        kernel_slice: str,
        current_metrics: dict = None,
        round_num: int = 0,
    ) -> list:
        tasks = [
            self._generate_single_beam(s, kernel_slice, current_metrics, round_num)
            for s in strategies
        ]
        return list(await asyncio.gather(*tasks))

    # ── Refinement rounds (reflection-based, inspired by Apex) ────────────────

    def _build_reflection_prompt(self, candidate) -> str:
        """Build a targeted refinement prompt based on candidate's performance profile.

        Classifies the result into failure modes and gives specific hints,
        rather than just picking a new strategy name.
        """
        env = self.env
        code = candidate.code if candidate.code else env.kernel_src
        metrics = candidate.metrics or {}
        speedup = candidate.speedup
        bottleneck = candidate.bottleneck

        # ── Classify result into failure mode ──
        if not candidate.compile_ok:
            mode = "compile_fail"
            feedback = (
                "The previous attempt FAILED TO COMPILE. Common causes:\n"
                "- Missing #include directives\n"
                "- Using undefined functions (only use functions from the included headers)\n"
                "- Changed the launch_* function signature\n"
                "- Syntax errors in template code\n"
                "Focus on producing CORRECT, COMPILABLE code first."
            )
        elif not candidate.correct:
            mode = "correctness_fail"
            feedback = (
                "The previous attempt COMPILED but PRODUCED WRONG RESULTS.\n"
                "- Check index calculations carefully (off-by-one, stride errors)\n"
                "- Verify vectorized loads unpack in the correct order\n"
                "- Ensure reductions accumulate all elements (no partial sums lost)\n"
                "- Check that the NVFP4 quantization block alignment is correct\n"
                "Fix the correctness bug while keeping the optimization approach."
            )
        elif speedup < 1.05:
            mode = "no_improvement"
            feedback = (
                f"The previous attempt compiled and was correct but achieved only {speedup:.3f}x "
                f"speedup (essentially no improvement). The optimization was too conservative "
                f"or didn't target the actual bottleneck.\n"
                f"Try a FUNDAMENTALLY DIFFERENT approach — don't just tweak, restructure."
            )
        elif speedup < 1.3:
            mode = "below_target"
            feedback = (
                f"The previous attempt achieved {speedup:.3f}x speedup — decent but not enough. "
                f"Target is >1.5x over FlashInfer baseline.\n"
                f"Build on what worked but apply MORE AGGRESSIVE optimizations."
            )
        else:
            mode = "good_but_improve"
            feedback = (
                f"The previous attempt achieved {speedup:.3f}x speedup — good! "
                f"Push further by combining multiple techniques or finding the remaining bottleneck."
            )

        # ── Build profiler metrics feedback ──
        metrics_section = ""
        if metrics:
            mem_pct = metrics.get("mem_throughput_pct", 0)
            compute_pct = metrics.get("compute_throughput_pct", 0)
            occupancy = metrics.get("sm_occupancy", 0)
            stall_mem = metrics.get("stall_memory", 0)
            stall_bar = metrics.get("stall_barrier", 0)
            l2_hit = metrics.get("l2_hit_rate", 0)
            bw_gbps = metrics.get("dram_read_bw_gbps", 0)

            metrics_section = f"""
## Profiler Results from Previous Attempt:
### Timing (MEASURED):
- Speedup: {speedup:.3f}x
- Achieved bandwidth: {bw_gbps:.1f} GB/s ({mem_pct:.1f}% of 6.0 TB/s peak)
- Compute utilization: {compute_pct:.1f}% of peak
- SM occupancy: {occupancy:.1f}%
- Memory stall rate: {stall_mem:.1f}%
- Barrier stall rate: {stall_bar:.1f}%
"""

            # ── Compiler metrics (registers, spills, SASS) ──
            cm = metrics.get("_compiler", {})
            if cm:
                regs = cm.get("registers_per_thread", 0)
                spill_st = cm.get("spill_stores_bytes", 0)
                spill_ld = cm.get("spill_loads_bytes", 0)
                smem = cm.get("static_smem_bytes", 0)
                sass_total = cm.get("sass_total_instructions", 0)

                metrics_section += f"""
### Binary Analysis (FROM COMPILER — exact values):
- Registers per thread: {regs} {'(>32 limits occupancy!)' if regs > 32 else '(good)'}
- Register spills: {spill_ld + spill_st} bytes {'⚠ SPILLING TO SLOW LOCAL MEMORY' if spill_ld + spill_st > 0 else '(none — good)'}
- Shared memory: {smem} bytes
"""
                if sass_total > 0:
                    ldg_128 = cm.get("sass_ldg_128", 0)
                    ldg_32 = cm.get("sass_ldg_32", 0)
                    ldg_64 = cm.get("sass_ldg_64", 0)
                    stg_128 = cm.get("sass_stg_128", 0)
                    stg_32 = cm.get("sass_stg_32", 0)
                    lds = cm.get("sass_lds", 0)
                    sts = cm.get("sass_sts", 0)
                    ldl = cm.get("sass_ldl", 0)
                    stl = cm.get("sass_stl", 0)
                    ffma = cm.get("sass_ffma", 0)
                    hfma2 = cm.get("sass_hfma2", 0)
                    mufu = cm.get("sass_mufu", 0)
                    bar = cm.get("sass_bar", 0)
                    shfl = cm.get("sass_shfl", 0)

                    total_ldg = ldg_32 + ldg_64 + ldg_128
                    vec_pct = (ldg_128 / total_ldg * 100) if total_ldg > 0 else 0

                    metrics_section += f"""
### SASS Instruction Mix (FROM DISASSEMBLY — ground truth):
- Total instructions: {sass_total}
- Global loads: {total_ldg} [128-bit: {ldg_128}, 64-bit: {ldg_64}, 32-bit: {ldg_32}] → vectorization: {vec_pct:.0f}%
- Global stores: {stg_32 + stg_128} [128-bit: {stg_128}, 32-bit: {stg_32}]
- Shared mem: {lds} loads, {sts} stores
- Spill instructions: {ldl + stl} {'⚠ REGISTER SPILLS IN BINARY' if ldl + stl > 0 else '(none)'}
- Compute: FFMA={ffma} HFMA2={hfma2} MUFU(SFU)={mufu}
- Barriers: {bar}  Shuffles: {shfl}
"""

            # Add bottleneck-specific hints
            regs = cm.get("registers_per_thread", 0) if cm else 0
            has_spills = (cm.get("spill_stores_bytes", 0) + cm.get("spill_loads_bytes", 0)) > 0 if cm else False

            if has_spills:
                metrics_section += (
                    "\n⚠ CRITICAL: Kernel has REGISTER SPILLS. This means the compiler ran out of\n"
                    "registers and is using slow local memory. This is likely your #1 performance issue.\n"
                    "- Reduce local variables and arrays\n"
                    "- Use fewer registers per thread (simpler per-thread logic)\n"
                    "- Consider smaller block size to give each thread more registers\n\n"
                )
            elif regs > 32 and occupancy < 60:
                metrics_section += (
                    f"\nBOTTLENECK: Register pressure ({regs} regs/thread → occupancy={occupancy:.0f}%).\n"
                    f"With {regs} registers, fewer warps can be active simultaneously.\n"
                    "- Reduce register usage: fewer local variables, recompute instead of caching\n"
                    "- Target ≤32 registers for 100% occupancy on B200\n\n"
                )
            elif mem_pct < 30 and compute_pct < 30:
                metrics_section += (
                    "\nBOTTLENECK: Latency-bound (both bandwidth and compute underutilized).\n"
                    "Priorities:\n"
                    "- REDUCE PASSES: fuse all operations into a single pass over data\n"
                    "- MINIMIZE GLOBAL READS: cache in registers, not residual_out rewrite\n"
                    "- MAXIMIZE ILP: each thread processes 8-16 elements with unrolling\n"
                    "- Use warp shuffles for reductions instead of shared memory + __syncthreads\n"
                    "- Process multiple rows per block to increase work per launch\n\n"
                )
            elif stall_mem > 30 and mem_pct > 30:
                metrics_section += (
                    "\nBOTTLENECK: Memory-bound. Priorities:\n"
                    "- Use float4/uint4 vectorized loads (128-bit per transaction)\n"
                    "- Fuse passes to reduce global memory round-trips\n"
                    "- Use __ldg() for read-only data\n\n"
                )
            elif stall_bar > 20:
                metrics_section += (
                    "\nBOTTLENECK: Sync-bound (high barrier stalls). Priorities:\n"
                    "- Replace __syncthreads() reductions with warp shuffles\n"
                    "- Use __shfl_xor_sync, __shfl_down_sync\n\n"
                )
            elif compute_pct > 60:
                metrics_section += (
                    "\nBOTTLENECK: Compute-bound. Priorities:\n"
                    "- Use fast math (__expf, __frcp_rn, __frsqrt_rn)\n"
                    "- Increase ILP with register tiling\n\n"
                )

        # ── B200 hardware context ──
        hw_context = (
            "## B200 Hardware (use these numbers to guide optimization):\n"
            "- HBM3e: 8 TB/s bandwidth → for 128×2048 bf16 (512 KB input), theoretical min = 0.06 us\n"
            "- 142 SMs, 228 KB shared memory per SM, 255 registers per thread\n"
            "- Warp size: 32, max 2048 threads per SM\n"
            "- 128-bit load/store transactions (float4 = 4×fp32, uint4 = 8×bf16)\n"
            "- Fast math SFU: __expf ~4 cycles vs expf ~20 cycles\n"
            "- Warp shuffle: __shfl_xor_sync costs ~2 cycles vs shared mem reduction ~10+ cycles\n"
        )

        return f"""\
You are an expert CUDA kernel optimizer. You are refining a previous optimization attempt.

## Status: {mode.upper().replace('_', ' ')}
{feedback}
{metrics_section}
{hw_context}

## Previous kernel code (your starting point — improve it):
```cuda
{code}
```

CRITICAL RULES:
1. Return the COMPLETE .cu file in a single ```cuda code block
2. Keep all #includes (use the original #include directives, NOT the expanded content)
3. Keep ALL kernel functions and the launch_* wrapper function
4. Do NOT change the launch_* function signature
5. Output must match reference within atol=1e-2
6. No explanations — just the code block
7. You may call any function defined in the included headers. Do NOT invent
   helper functions that aren't defined in the headers.
"""

    async def refine_beams(self, survivors: list, round_num: int) -> list:
        tasks = []
        for candidate in survivors:
            prompt = self._build_reflection_prompt(candidate)
            tasks.append(self._refine_single_beam(prompt, candidate, round_num))
        return list(await asyncio.gather(*tasks))

    async def _refine_single_beam(self, prompt: str, parent: 'KernelCandidate', round_num: int) -> 'KernelCandidate':
        """Run a reflection-based refinement for a single candidate."""
        try:
            response, _, _ = await self._call_llm_async(
                prompt, model=self.sub_model, temperature=0.4
            )
        except RuntimeError as e:
            logger.error("Budget exceeded during refine %s: %s", parent.strategy, e)
            return KernelCandidate(code="", strategy=parent.strategy, round_num=round_num)

        code = self._extract_cuda_code(response)
        if not code:
            logger.warning("No CUDA code extracted for refinement of %s", parent.strategy)
        return KernelCandidate(
            code=code,
            strategy=f"{parent.strategy}_r{round_num}",
            round_num=round_num,
            compile_ok=bool(code),
        )

    # ── Combination step ──────────────────────────────────────────────────────

    def combine(self, top_candidates: list) -> KernelCandidate:
        if len(top_candidates) < 2:
            return top_candidates[0]

        a, b   = top_candidates[0], top_candidates[1]
        hot_a  = a.code
        hot_b  = b.code

        prompt = combine_prompt(
            variant_a_summary=a.summary(),
            variant_a_code=hot_a,
            variant_b_summary=b.summary(),
            variant_b_code=hot_b,
        )

        response, _, _ = self._call_llm(prompt, model=self.combine_model, temperature=0.2)
        combined_code  = self._extract_cuda_code(response)

        if not combined_code:
            logger.warning("No CUDA code extracted from combine step, using best candidate")
            return top_candidates[0]
        return KernelCandidate(
            code=combined_code,
            strategy=f"combined_{a.strategy}+{b.strategy}",
            round_num=self.refine_rounds + 1,
            compile_ok=True,
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _extract_cuda_code(self, text: str) -> str:
        # Try code blocks first (with and without language tags)
        for pattern in [r"```cuda\s*\n(.*?)```", r"```cpp\s*\n(.*?)```",
                        r"```c\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        # If no code block but contains CUDA keywords, extract from first CUDA line
        if "__global__" in text or "__device__" in text or "#include" in text:
            lines = text.split("\n")
            start = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if (stripped.startswith("#include") or stripped.startswith("__global__")
                        or stripped.startswith("__device__") or stripped.startswith("//")
                        or stripped.startswith("/*") or stripped.startswith("typedef")
                        or stripped.startswith("template") or stripped.startswith("static")
                        or stripped.startswith("namespace") or stripped.startswith("extern")):
                    start = i
                    break
            return "\n".join(lines[start:]).strip()
        return ""

    def _extract_hot_loop_from_code(self, code: str) -> str:
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if "for" in line and ("idx" in line or "tid" in line or "blockIdx" in line):
                return "\n".join(lines[i:min(i + 25, len(lines))])
        return code[:800]

    def _bottleneck_to_strategy(self, bottleneck: str, current_strategy: str) -> str:
        """Select next strategy based on bottleneck AND kernel type.

        Uses kernel-type-aware strategy bank to pick strategies that are
        actually applicable, rather than a static mapping that suggests
        TMA for elementwise kernels.
        """
        from search.strategy_bank import select_for_kernel, STRATEGY_BANK

        tried = [current_strategy] + self.env.optimization_history.strategies_tried()
        kernel_type = self.env.kernel_type

        # Get kernel-aware candidates, excluding already-tried strategies
        candidates = select_for_kernel(
            kernel_type=kernel_type,
            tried=tried,
            beam_width=4,
        )

        # Prefer strategies that target the identified bottleneck
        for name in candidates:
            s = STRATEGY_BANK.get(name)
            if s and s.applies_to(bottleneck):
                return name

        # If nothing matches the bottleneck, just use the top kernel-aware pick
        if candidates:
            return candidates[0]

        # Ultimate fallback — cycle through applicable strategies
        for name, s in STRATEGY_BANK.items():
            if name not in tried and s.applies_to_kernel(kernel_type):
                return name

        return "vectorize_loads"

    # ── Synchronous wrappers (for beam_search.py) ─────────────────────────────

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Reuse a single event loop to avoid 'Event loop is closed' errors
        from AsyncAnthropic's httpx connection pool cleanup."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def close(self):
        """Properly close the async client and event loop."""
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.run_until_complete(self.async_client.close())
            except Exception:
                pass
            self._loop.close()
        self._loop = None

    def __del__(self):
        self.close()

    def run_decompose(self) -> list:
        return self.decompose()

    def run_generate_beams(
        self, strategies: list, kernel_slice: str,
        current_metrics: dict = None, round_num: int = 0,
    ) -> list:
        loop = self._get_or_create_loop()
        return loop.run_until_complete(
            self.generate_beams(strategies, kernel_slice, current_metrics, round_num)
        )

    def run_refine_beams(self, survivors: list, round_num: int) -> list:
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self.refine_beams(survivors, round_num))
