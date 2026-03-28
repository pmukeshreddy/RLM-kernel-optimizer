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
from .root_prompts import SYSTEM_PROMPT, decompose_prompt, combine_prompt
from .sub_prompts import get_prompt_for_strategy
from .reflector import (
    _get_launch_signature, _format_profile_section,
    _format_suggestions_section, _format_react_trace,
    _format_last_error_section, _format_delta_section,
    _compute_proven_ineffective,
)

logger = logging.getLogger(__name__)


# ── Refinement: tool-use agent loop ────────────────────────────────────────

MAX_INNER_TURNS = 3

SUBMIT_KERNEL_TOOL = {
    "name": "submit_kernel",
    "description": (
        "Submit optimized CUDA kernel for compilation, correctness checking, "
        "and profiling.\n\n"
        "Returns one of:\n"
        "- COMPILE ERROR: first error with file:line plus surrounding context\n"
        "- CORRECTNESS FAILURE: max error magnitude and which check failed\n"
        "- Result verdict (IMPROVED / REGRESSION / NO CHANGE) with:\n"
        "  timing_us, speedup vs baseline, SM occupancy,\n"
        "  SASS breakdown (load vectorization %, loads by width, "
        "stores, barriers, shuffles, register spills),\n"
        "  delta from your previous submission,\n"
        "  remaining optimization suggestions from profiler data"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "cuda_code": {
                "type": "string",
                "description": (
                    "Complete .cu file content with all #includes, "
                    "kernel functions, and the launch_* wrapper."
                ),
            }
        },
        "required": ["cuda_code"],
    },
}

REFINE_SYSTEM_PROMPT = f"""\
You are a CUDA kernel optimization agent. You have {{turns}} submit_kernel calls.

Target hardware — NVIDIA B200 (sm_100a, Blackwell):
- HBM3e: 8 TB/s bandwidth, 192 GB
- L2 cache: 126 MB — but benchmark uses L2 cache cycling (data is COLD every iteration)
- 142 SMs, 228 KB shared memory per SM, 255 registers per thread
- Warp size 32, max 2048 threads per SM
- 128-bit load/store = uint4 = 8 bf16 values per transaction
- Warp shuffle (__shfl_xor_sync): ~2 cycles vs shared mem reduction ~10+
- Fast math SFU: __expf/__rsqrtf ~4 cycles vs expf/rsqrtf ~20 cycles

CRITICAL — These kernels are MEMORY-BOUND:
- The benchmark forces L2 cache misses every iteration. Data comes from HBM, not L2.
- The GPU is IDLE most of the time, waiting for HBM data to arrive.
- Pure compute optimizations (faster math, ILP, register tiling) will NOT improve timing.
- To get real speedup you MUST either reduce total memory traffic or overlap memory with compute:
  * Software pipelining: load next tile via cp.async while computing on current tile
  * Double buffering: use 2 shared memory buffers to overlap HBM loads with computation
  * Fuse all operations into a single pass: load ONCE into registers, compute everything, store ONCE
  * Vectorized 128-bit loads (uint4): maximize bytes per memory transaction

Before EVERY submit_kernel call, you MUST first explain in 2-3 sentences:
1. What the profiler data tells you is the current bottleneck
2. What specific code change you will make and why you expect it to help

Rules:
{{constraint}}
- NEVER put __syncthreads() inside if/else branches (deadlock).
- The launch_* function signature must match exactly.
- Output must match reference within atol=1e-2.
- Keep original #include directives, not expanded headers.
- Do not use torch headers (torch/extension.h, ATen, c10).
"""


def _constraint_for_speedup(speedup: float) -> str:
    """Constraint guard: only restrict changes when kernel is already fast."""
    if speedup >= 1.6:
        return (
            "- Surgical changes only (5-15 lines). The kernel is already 1.6x+.\n"
            "- Keep all existing optimizations — do not regress.\n"
            "- Make targeted micro-optimizations, do not rewrite from scratch."
        )
    return "- Structural changes, algorithmic rewrites, and new optimization patterns are all allowed."


def _build_refine_system_prompt(speedup: float) -> str:
    """Build REFINE_SYSTEM_PROMPT with dynamic constraint for current speedup."""
    constraint = _constraint_for_speedup(speedup)
    return REFINE_SYSTEM_PROMPT.replace("{{turns}}", str(MAX_INNER_TURNS)).replace("{{constraint}}", constraint)


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
        self.async_client = AsyncAnthropic(max_retries=10)
        # Limit concurrent API calls to avoid 429 rate-limit errors
        self._api_semaphore = asyncio.Semaphore(2)
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

        async with self._api_semaphore:
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

    async def _call_llm_with_tools_async(
        self,
        messages: list,
        tools: list,
        model: str,
        system: str = SYSTEM_PROMPT,
        temperature: float = 0.4,
    ):
        """Async API call with tool use support. Returns full response object."""
        if self.env.over_budget():
            raise RuntimeError(
                f"Budget exhausted: ${self.env.total_api_cost_usd:.4f} spent"
            )

        async with self._api_semaphore:
            response = await self.async_client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
                tools=tools,
            )

        tokens_in  = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = self.env.record_api_cost(tokens_in, tokens_out, model)
        logger.info(
            "LLM tool call: model=%s in=%d out=%d cost=$%.4f stop=%s",
            model, tokens_in, tokens_out, cost, response.stop_reason,
        )
        return response

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

        num_strategies = self.beam_width * 2  # extra strategies held in reserve
        example_lines = "\n".join(
            f'  {{"name": "short_name", "what": "one line description of the concrete change"}},'
            for _ in range(num_strategies)
        )
        prompt = f"""\
You are a CUDA optimization expert. Analyze this kernel and propose exactly {num_strategies}
DIVERSE optimization techniques that would give the biggest speedup.

CRITICAL CONTEXT — This kernel is MEMORY-BOUND:
- Benchmark uses L2 cache cycling — data is COLD every iteration, fetched from HBM (not L2).
- The GPU is mostly IDLE, waiting for HBM data. Arithmetic is NOT the bottleneck.
- Pure compute optimizations (faster math, ILP, register tiling) alone will NOT help.
- To get real speedup, strategies MUST reduce total memory traffic or hide memory latency:
  * Software pipelining / double buffering (cp.async + shared memory ping-pong)
  * Fuse all passes into single-pass: load ONCE, compute everything in registers, store ONCE
  * Vectorized 128-bit loads/stores (uint4) to maximize memory bus utilization
  * Reduce total bytes: eliminate redundant loads/stores, compress intermediates
  * Overlap: load next tile while computing on current tile

You are NOT limited to any predefined list. Propose whatever CUDA optimizations
you think are most impactful for THIS SPECIFIC kernel. Be specific and concrete.
Each strategy should be FUNDAMENTALLY DIFFERENT — not variations of the same idea.
At least half your strategies MUST target memory access patterns, not compute.

Kernel type: {env.kernel_type}
Target: NVIDIA B200 (Blackwell, sm_100a, 8 TB/s HBM3e, 126 MB L2, 142 SMs)

```cuda
{kernel_src}
```

For each optimization, give a short name and one-line description of what to do.

Return as a JSON array of objects, most impactful first:
[
{example_lines}
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
            names = [s["name"] for s in strategies]
            logger.info("LLM-proposed strategies (%d): %s", len(names), names)
            return strategies  # return ALL — caller splits active vs reserve

        # Fallback: use kernel-aware defaults from strategy bank
        from search.strategy_bank import select_for_kernel
        fallback = select_for_kernel(
            kernel_type=env.kernel_type, tried=[], beam_width=num_strategies)
        logger.info("Fallback to kernel-aware strategies: %s", fallback)
        return [{"name": s, "what": ""} for s in fallback]

    # ── Sub-LLM beam generation (parallel) ───────────────────────────────────

    async def _generate_single_beam(
        self,
        strategy,
        kernel_slice: str,
        current_metrics: dict = None,
        round_num: int = 0,
        profile_fn=None,
    ) -> KernelCandidate:
        # Support both freeform dicts {name, what} and plain strategy name strings
        if isinstance(strategy, dict):
            strat_name = strategy.get("name", "unknown")
            strat_desc = strategy.get("what", "")
        else:
            strat_name = strategy
            strat_desc = ""

        launch_sig = _get_launch_signature(self.env.kernel_type)

        # ── Tool-use path: compile/test/iterate like refinement ──────────
        if profile_fn and strat_desc:
            constraint = _constraint_for_speedup(0.0)  # round 0 = aggressive
            system_prompt = _build_refine_system_prompt(0.0)

            prompt_parts = [
                f"You are beam \"{strat_name}\" — generate an optimized CUDA kernel from scratch.",
                f"Direction: {strat_desc}",
                "\nIMPORTANT: This kernel is MEMORY-BOUND. The benchmark uses L2 cache cycling "
                "so data comes from HBM every time. The GPU is idle waiting for memory. "
                "Compute-only optimizations won't help — you must reduce memory traffic or "
                "overlap loads with compute (software pipelining, cp.async, double buffering).",
                f"\n## Reference kernel (BASELINE — this is what you must beat):\n"
                f"```cuda\n{kernel_slice}\n```",
                f"\n{launch_sig}",
                "\nBefore calling submit_kernel, explain in 2-3 sentences:\n"
                "1. What optimization you're applying and why it should help\n"
                "2. What specific code changes you're making\n\n"
                "Then call submit_kernel with your complete .cu file.",
            ]
            initial_prompt = "\n\n".join(prompt_parts)

            messages = [{"role": "user", "content": initial_prompt}]
            best = None
            prev_inner_metrics = None

            for turn in range(MAX_INNER_TURNS):
                try:
                    response = await self._call_llm_with_tools_async(
                        messages=messages,
                        tools=[SUBMIT_KERNEL_TOOL],
                        model=self.sub_model,
                        system=system_prompt,
                        temperature=0.5,
                    )
                except RuntimeError as e:
                    logger.error("Budget exceeded gen %s turn %d: %s",
                                 strat_name, turn, e)
                    break

                messages.append({"role": "assistant", "content": response.content})

                tool_block = next(
                    (b for b in response.content
                     if b.type == "tool_use" and b.name == "submit_kernel"),
                    None,
                )
                if not tool_block:
                    logger.info("GEN [%s] turn %d: no tool call", strat_name, turn)
                    break

                code = tool_block.input.get("cuda_code", "")
                if not code:
                    messages.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": tool_block.id,
                         "content": "Error: empty cuda_code. Submit the complete .cu file.",
                         "is_error": True}
                    ]})
                    continue

                result = await asyncio.get_event_loop().run_in_executor(
                    None, profile_fn, code, strat_name, round_num)

                tool_result_text = self._format_tool_result(
                    result, 1.0, prev_inner_metrics)
                if result["compile_ok"] and result["correct"] and result.get("metrics"):
                    prev_inner_metrics = result["metrics"]
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tool_block.id,
                     "content": tool_result_text}
                ]})

                logger.info("GEN [%s] turn %d: compile=%s correct=%s speedup=%.3fx",
                            strat_name, turn,
                            result["compile_ok"], result["correct"],
                            result.get("speedup", 0))

                if result["compile_ok"] and result["correct"]:
                    if best is None or result["speedup"] > best.speedup:
                        best = KernelCandidate(
                            code=code,
                            strategy=strat_name,
                            round_num=round_num,
                            compile_ok=True,
                            correct=True,
                            speedup=result["speedup"],
                            metrics=result.get("metrics", {}),
                            bottleneck=result.get("bottleneck", "unknown"),
                        )
                        best.strategy_context = strat_desc

            if best:
                return best

            # All turns failed — return failure
            failed = KernelCandidate(
                code="", strategy=strat_name, round_num=round_num,
                compile_ok=False,
            )
            failed.strategy_context = strat_desc
            return failed

        # ── One-shot fallback (no profile_fn or no description) ──────────
        if strat_desc:
            prompt = f"""\
You are an expert CUDA kernel optimizer targeting NVIDIA B200 (sm_100a, Blackwell).

Apply this optimization to the kernel below:

## Optimization: {strat_name}
{strat_desc}

## B200 Hardware:
- HBM3e: 8 TB/s bandwidth — use float4/uint4 (128-bit) loads to saturate it
- L2 cache: 126 MB — but benchmark uses L2 cycling so data is COLD (from HBM)
- 142 SMs, 228 KB shared memory per SM, 255 registers per thread
- Warp shuffle (__shfl_xor_sync) costs ~2 cycles vs shared mem reduction ~10+ cycles
- Fast math SFU: __expf ~4 cycles vs expf ~20 cycles; __frcp_rn for reciprocal
- For bf16: load 8 values at once via reinterpret_cast<uint4*>, unpack to float for compute

## MEMORY-BOUND WARNING:
This kernel is memory-bound — the GPU is idle waiting for HBM data most of the time.
Compute tricks alone won't help. You MUST reduce memory traffic or overlap loads with compute.
Key techniques: software pipelining (cp.async + double buffer), single-pass fusion, vectorized loads.

## Full kernel source (headers expanded inline between === markers):
```cuda
{kernel_slice}
```

{launch_sig}

CRITICAL RULES:
1. Return the COMPLETE .cu file in a single ```cuda code block
2. Keep all #includes (use the original #include directives, NOT the expanded content)
3. Do NOT use torch headers (torch/extension.h, ATen, c10) — this is standalone CUDA
4. Keep ALL kernel functions and the launch_* wrapper function
5. The launch_* function signature MUST match the "Required Launch Function" section EXACTLY.
   If you change it, you will get "undefined reference" linker errors.
6. Output must match reference within atol=1e-2
7. NEVER put __syncthreads() inside an if/else branch — all threads in a block MUST hit the same barrier or the kernel will deadlock.
8. No explanations — just the code block
9. You may call any function defined in the expanded headers above. Do NOT invent
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
                prompt, model=self.sub_model, temperature=0.6
            )
        except RuntimeError as e:
            logger.error("Budget exceeded during beam %s: %s", strat_name, e)
            return KernelCandidate(code="", strategy=strat_name, round_num=round_num)

        code = self._extract_cuda_code(response)
        if not code:
            logger.warning("No CUDA code extracted for strategy=%s (response starts: %s)",
                           strat_name, response[:100])
        c = KernelCandidate(
            code=code,
            strategy=strat_name,
            round_num=round_num,
            compile_ok=bool(code),
        )
        c.strategy_context = strat_desc
        return c

    async def generate_beams(
        self,
        strategies: list,
        kernel_slice: str,
        current_metrics: dict = None,
        round_num: int = 0,
        profile_fn=None,
    ) -> list:
        tasks = [
            self._generate_single_beam(
                s, kernel_slice, current_metrics, round_num, profile_fn)
            for s in strategies
        ]
        return list(await asyncio.gather(*tasks))

    # ── Refinement: multi-turn tool-use loop ─────────────────────────────────

    async def refine_beams(self, survivors: list, round_num: int,
                           profile_fn=None) -> list:
        tasks = [
            self._refine_single_beam(candidate, round_num, profile_fn)
            for candidate in survivors
        ]
        return list(await asyncio.gather(*tasks))

    async def _refine_single_beam(
        self,
        parent: 'KernelCandidate',
        round_num: int,
        profile_fn=None,
    ) -> 'KernelCandidate':
        """Multi-turn refinement with tool use.

        Single conversation where the model sees profiler data AND code,
        reasons about the bottleneck, and calls submit_kernel to compile/profile.
        If it fails or regresses, it sees the error and can fix+resubmit
        (up to MAX_INNER_TURNS).

        Returns a fully profiled KernelCandidate (best from inner loop).
        """
        metrics = parent.metrics or {}
        prev_metrics = parent.prev_metrics
        has_fresh_data = bool(prev_metrics) and prev_metrics != metrics

        # Build profiler observation
        if has_fresh_data:
            profile_section = _format_profile_section(prev_metrics, round_num)
            ineffective, ineff_lines = _compute_proven_ineffective(prev_metrics, metrics)
            delta_section = _format_delta_section(prev_metrics, metrics,
                                                  title="Last Attempt vs Current Best")
            if ineff_lines:
                dead_ends = "\n### Proven Non-Bottlenecks (do NOT optimize these)\n"
                dead_ends += "\n".join(f"- {l}" for l in ineff_lines)
            else:
                dead_ends = ""
        else:
            profile_section = _format_profile_section(metrics, round_num)
            delta_section = ""
            dead_ends = ""

        last_error_section = _format_last_error_section(parent)
        react_trace = _format_react_trace(parent)
        strat_ctx = getattr(parent, 'strategy_context', '')
        strat_name = parent.strategy.split("_r")[0]

        logger.info("Refining [%s_r%d]: compile=%s correct=%s speedup=%.3fx",
                     parent.strategy, round_num, parent.compile_ok,
                     parent.correct, parent.speedup)

        # Build observation block
        obs_parts = [profile_section]
        if delta_section:
            obs_parts.append(delta_section)
        if dead_ends:
            obs_parts.append(dead_ends)
        if last_error_section:
            obs_parts.append(last_error_section)
        observation = "\n".join(p for p in obs_parts if p)

        # Build initial prompt: profiler data + code in ONE message
        prompt_parts = []

        if strat_ctx:
            prompt_parts.append(
                f"You are beam \"{strat_name}\" optimizing a CUDA kernel.\n"
                f"Direction: {strat_ctx}")
        else:
            prompt_parts.append(
                f"You are beam \"{strat_name}\" optimizing a CUDA kernel.")

        if has_fresh_data:
            last_spd = prev_metrics.get("speedup", 0)
            prompt_parts.append(
                f"Best so far: {parent.speedup:.3f}x. Last attempt: {last_spd:.3f}x.")
        else:
            prompt_parts.append(f"Current: {parent.speedup:.3f}x.")

        if react_trace:
            prompt_parts.append(react_trace)

        prompt_parts.append(f"Observation:\n{observation}")

        # Code + launch signature — model sees BOTH data and code
        base_code = parent.best_code or parent.code
        base_speedup = parent.best_speedup or parent.speedup
        launch_sig = _get_launch_signature(self.env.kernel_type)
        prompt_parts.append(
            f"Current kernel ({base_speedup:.3f}x):\n```cuda\n{base_code}\n```"
            f"\n\n{launch_sig}")

        prompt_parts.append(
            "Before calling submit_kernel, explain what the profiler shows is the "
            "bottleneck and what code change you'll make. Example:\n\n"
            "\"SASS shows 18 scalar LDG.32 loads (0% vectorized) but occupancy "
            "is 96%, so the bottleneck is memory transaction efficiency not "
            "parallelism. I'll pack the scalar bf16 loads in the hot loop into "
            "uint4 loads via reinterpret_cast<uint4*>, cutting 18 loads to 3 "
            "and using the full 128-bit bus width.\"\n\n"
            "Then call submit_kernel with your complete .cu file.")

        initial_prompt = "\n\n".join(prompt_parts)
        logger.info("REFINE PROMPT [%s]: %d chars", parent.strategy, len(initial_prompt))

        # ── Multi-turn tool-use conversation ──────────────────────────────
        messages = [{"role": "user", "content": initial_prompt}]
        best = None              # best KernelCandidate from inner loop
        last_error = ""
        prev_inner_metrics = None  # for SASS deltas between submissions

        for turn in range(MAX_INNER_TURNS):
            try:
                system_prompt = _build_refine_system_prompt(parent.speedup)
                response = await self._call_llm_with_tools_async(
                    messages=messages,
                    tools=[SUBMIT_KERNEL_TOOL],
                    model=self.sub_model,
                    system=system_prompt,
                    temperature=0.4,
                )
            except RuntimeError as e:
                logger.error("Budget exceeded refine %s turn %d: %s",
                             parent.strategy, turn, e)
                break

            # Append assistant message to conversation history
            messages.append({"role": "assistant", "content": response.content})

            # Find submit_kernel tool call
            tool_block = next(
                (b for b in response.content
                 if b.type == "tool_use" and b.name == "submit_kernel"),
                None,
            )

            if not tool_block:
                text_parts = [b.text for b in response.content
                              if hasattr(b, 'text')]
                logger.info("REFINE [%s] turn %d: no tool call: %s",
                            parent.strategy, turn,
                            " ".join(text_parts)[:200])
                break

            code = tool_block.input.get("cuda_code", "")
            if not code:
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tool_block.id,
                     "content": "Error: empty cuda_code. Submit the complete .cu file.",
                     "is_error": True}
                ]})
                continue

            # Profile the submission via callback
            if profile_fn:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, profile_fn, code, parent.strategy, round_num)
            else:
                result = {"compile_ok": False, "correct": False, "speedup": 0,
                          "metrics": {}, "error": "No profiler available",
                          "bottleneck": "unknown"}

            # Feed result back as tool_result (with delta from prev submission)
            tool_result_text = self._format_tool_result(
                result, parent.speedup, prev_inner_metrics)
            if result["compile_ok"] and result["correct"] and result.get("metrics"):
                prev_inner_metrics = result["metrics"]
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_block.id,
                 "content": tool_result_text}
            ]})

            logger.info("REFINE [%s] turn %d: compile=%s correct=%s speedup=%.3fx",
                         parent.strategy, turn,
                         result["compile_ok"], result["correct"],
                         result.get("speedup", 0))

            # Track best viable result
            if result["compile_ok"] and result["correct"]:
                if best is None or result["speedup"] > best.speedup:
                    best = KernelCandidate(
                        code=code,
                        strategy=f"{parent.strategy}_r{round_num}",
                        round_num=round_num,
                        compile_ok=True,
                        correct=True,
                        speedup=result["speedup"],
                        metrics=result.get("metrics", {}),
                        bottleneck=result.get("bottleneck", "unknown"),
                        prev_metrics=parent.metrics,
                    )
                    best.strategy_context = strat_ctx
                    text_parts = [b.text for b in response.content
                                  if hasattr(b, 'text')]
                    best.strategy_desc = (
                        " ".join(text_parts)[:300] if text_parts else "")
            else:
                last_error = result.get("error", "")

        # Return best viable candidate, or a failure
        if best:
            return best

        failed = KernelCandidate(
            code="",
            strategy=f"{parent.strategy}_r{round_num}",
            round_num=round_num,
            compile_ok=False,
            prev_metrics=parent.metrics,
        )
        failed.compile_error = last_error or "All inner refinement attempts failed"
        failed.strategy_context = strat_ctx
        failed.strategy_desc = ""
        return failed

    def _format_tool_result(self, result: dict, parent_speedup: float,
                            prev_inner_metrics: dict = None) -> str:
        """Format profiling result as tool_result content for the model.

        Follows Anthropic guidance: error responses must communicate specific
        and actionable improvements, success responses include delta + suggestions.
        """
        if not result["compile_ok"]:
            error = result.get("error", "Unknown compilation error")
            # Extract first actionable error line from nvcc output
            lines = error.split('\n')
            error_lines = [l for l in lines if 'error' in l.lower()
                           and ('(' in l or ':' in l)]
            if error_lines:
                first = error_lines[0].strip()
                return (f"COMPILE ERROR: {first}\n\n"
                        f"Full output ({len(error_lines)} error(s)):\n"
                        f"{error[:500]}")
            return f"COMPILE ERROR:\n{error[:600]}"

        if not result["correct"]:
            error = result.get("error", "Output mismatch (atol=1e-2)")
            return f"CORRECTNESS FAILURE: {error[:400]}"

        # Viable — verdict + delta + suggestions (no code echo)
        metrics = result.get("metrics", {})
        speedup = result.get("speedup", 0)

        if speedup > parent_speedup + 0.02:
            verdict = f"IMPROVED: {speedup:.3f}x (was {parent_speedup:.3f}x)"
        elif speedup < parent_speedup - 0.01:
            verdict = f"REGRESSION: {speedup:.3f}x (was {parent_speedup:.3f}x)"
        else:
            verdict = f"NO CHANGE: {speedup:.3f}x (was {parent_speedup:.3f}x)"

        parts = [verdict]

        # SASS delta from previous submission — model sees what its change did
        if prev_inner_metrics:
            delta = _format_delta_section(
                metrics, prev_inner_metrics,
                title="Changes from Previous Submission")
            if delta:
                parts.append(delta)
        else:
            # First submission — show full profiler snapshot
            parts.append(_format_profile_section(metrics, 0))

        # Data-driven suggestions for remaining bottlenecks
        suggestions = _format_suggestions_section(
            metrics, kernel_type=self.env.kernel_type)
        if suggestions:
            parts.append(suggestions)

        return "\n".join(parts)

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
        profile_fn=None,
    ) -> list:
        loop = self._get_or_create_loop()
        return loop.run_until_complete(
            self.generate_beams(strategies, kernel_slice, current_metrics,
                                round_num, profile_fn)
        )

    def run_refine_beams(self, survivors: list, round_num: int,
                         profile_fn=None) -> list:
        loop = self._get_or_create_loop()
        return loop.run_until_complete(
            self.refine_beams(survivors, round_num, profile_fn))
