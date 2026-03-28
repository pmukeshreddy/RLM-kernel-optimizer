"""
engine.py — RLM core engine.
Orchestrates root LLM decomposition, parallel sub-LLM beam generation, and refinement.
"""

from __future__ import annotations
import asyncio
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

import anthropic
from anthropic import AsyncAnthropic

from .environment import RLMEnvironment, KernelCandidate
from .root_prompts import SYSTEM_PROMPT, combine_prompt
from .reflector import (
    _get_launch_signature, _format_profile_section,
    _format_suggestions_section, _format_react_trace,
    _format_last_error_section, _format_delta_section,
    _compute_proven_ineffective,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


# ── Refinement: tool-use agent loop ────────────────────────────────────────

MAX_INNER_TURNS = 5

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

INSPECT_SASS_TOOL = {
    "name": "inspect_sass",
    "description": (
        "Compile CUDA code and return the full SASS assembly (cuobjdump -sass output). "
        "Use this to see exactly what instructions the compiler generated — load widths, "
        "branch counts, spills, vectorization, etc. Does NOT run or benchmark the kernel. "
        "Costs no submit_kernel turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "cuda_code": {
                "type": "string",
                "description": "Complete .cu file to compile and disassemble.",
            }
        },
        "required": ["cuda_code"],
    },
}

READ_FILE_TOOL = {
    "name": "read_file",
    "description": (
        "Read a source file from the project. Available files:\n"
        "- kernels/common/nvfp4_utils.cuh — FP4/FP8 quantization helpers, pack/unpack\n"
        "- kernels/common/b200_intrinsics.cuh — Blackwell TMA, TMEM, pipeline wrappers\n"
        "- kernels/reference/add_rmsnorm.cu — Naive Add+RMSNorm+FP4 reference kernel\n"
        "- kernels/reference/silu_mul.cu — Naive SiLU*Mul+FP4 reference kernel\n"
        "- kernels/reference/nvfp4_quantize.cu — Naive BF16→FP4 reference kernel\n"
        "Costs no submit_kernel turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path from project root (e.g. 'kernels/common/nvfp4_utils.cuh')",
            }
        },
        "required": ["path"],
    },
}

SEARCH_DOCS_TOOL = {
    "name": "search_docs",
    "description": (
        "Search CUDA intrinsic documentation. Query by keyword to find correct "
        "function signatures, headers, and usage examples. Covers: FP4/FP8 conversion "
        "(cuda_fp4.h, cuda_fp8.h), warp intrinsics (shuffle, reduction), fast math "
        "(SFU), memory intrinsics (ldg, stcg, async copy), bfloat16/half operations.\n"
        "Example queries: 'fp4 convert float', 'fp8 e4m3 to float', 'warp reduction', "
        "'fast reciprocal sqrt', 'bfloat16 pair load'\n"
        "Costs no submit_kernel turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords (e.g. 'fp4 quantize float', 'e4m3 convert', 'warp reduce')",
            }
        },
        "required": ["query"],
    },
}

ALL_TOOLS = [SUBMIT_KERNEL_TOOL, INSPECT_SASS_TOOL, READ_FILE_TOOL, SEARCH_DOCS_TOOL]

REFINE_SYSTEM_PROMPT = f"""\
You are a CUDA kernel optimization agent. You have {{turns}} submit_kernel calls.

Your speedup is measured against FlashInfer, a production GPU library.
You target a single GPU (B200, sm_100a) and a single problem shape — use this to your advantage.

Available tools (only submit_kernel counts toward your turn limit):
- submit_kernel: compile, test correctness, and benchmark your kernel
- inspect_sass: compile code and see the raw SASS assembly (instruction-level view)
- read_file: read project header files (nvfp4_utils.cuh, b200_intrinsics.cuh) or reference kernels
- search_docs: look up CUDA intrinsic signatures and usage (fp4, fp8, warp, fast math, memory)

Target hardware — NVIDIA B200 (sm_100a, Blackwell):
- HBM3e: 8 TB/s bandwidth, 192 GB
- L2 cache: 126 MB — benchmark uses L2 cache cycling (data is COLD every iteration)
- 142 SMs, 228 KB shared memory per SM, 255 registers per thread
- 128-bit load/store = uint4 = 8 bf16 values per transaction
- Use read_file to check available hardware intrinsics in the project headers

Before EVERY submit_kernel call, explain in 2-3 sentences:
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

def _build_refine_system_prompt(speedup: float) -> str:
    """Build REFINE_SYSTEM_PROMPT with dynamic constraint for current speedup."""
    # Provide a unified constraint regardless of speedup
    constraint = "- Structural changes, algorithmic rewrites, and surgical optimizations are all allowed."
    return REFINE_SYSTEM_PROMPT.replace("{{turns}}", str(MAX_INNER_TURNS)).replace("{{constraint}}", constraint)


class RLMEngine:
    """
    Main orchestrator for the RLM beam search loop.
    Handles: decomposition → beam generation → profiler-guided refinement → combination.
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

        # Build baseline profiler context for strategy selection
        baseline_context = ""
        if env.baseline_naive_us and env.baseline_us_reported:
            rows = env.problem_shapes[0][0]
            sm_count = env.hw_spec.get("sm", {}).get("count", 148)
            cm = env.baseline_compiler_metrics
            cm_str = cm.summary_str() if cm else "unavailable"
            baseline_context = (
                f"BASELINE PROFILER DATA (reference kernel):\n"
                f"  Naive kernel timing: {env.baseline_naive_us:.3f} us\n"
                f"  FlashInfer timing:   {env.baseline_us_reported:.3f} us\n"
                f"  Compiler: {cm_str}\n"
                f"  Grid: {rows} blocks launched on {sm_count} SMs"
                f"{' — some SMs get zero work' if rows < sm_count else ''}\n"
            )
        if not baseline_context:
            baseline_context = "BASELINE PROFILER DATA: unavailable — analyze kernel source to infer bottleneck type.\n"

        num_strategies = self.beam_width * 2  # extra strategies held in reserve
        example_lines = "\n".join(
            f'  {{"name": "short_name", "what": "one line description of the concrete change"}},'
            for _ in range(num_strategies)
        )
        prompt = f"""\
You are a CUDA optimization expert. Analyze this kernel and propose exactly {num_strategies}
DIVERSE optimization techniques that would give the biggest speedup.

Speedup is measured against FlashInfer, a production GPU library.
You target ONE GPU (B200, sm_100a) and ONE shape ({env.problem_shapes[0]}).

{baseline_context}

Kernel type: {env.kernel_type}
Problem shape: {env.problem_shapes[0]}
Target: NVIDIA B200 (Blackwell, sm_100a, 8 TB/s HBM3e, 126 MB L2, 142 SMs, 228KB smem/SM)

```cuda
{kernel_src}
```

Analyze the profiler data and kernel source. Propose {num_strategies} diverse optimizations,
most impactful first. Each should be a concrete, actionable change.

Return as a JSON array of objects:
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

        # Fallback: generic strategies if LLM response couldn't be parsed
        logger.warning("Could not parse LLM strategy response, using generic fallback")
        generic = [
            {"name": "optimization_a", "what": "Analyze kernel and apply the most impactful optimization"},
            {"name": "optimization_b", "what": "Try a different optimization approach than beam A"},
            {"name": "optimization_c", "what": "Try a third independent optimization approach"},
            {"name": "optimization_d", "what": "Try a fourth independent optimization approach"},
        ]
        return generic[:num_strategies]

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
            system_prompt = _build_refine_system_prompt(0.0)

            shape_str = str(self.env.problem_shapes[0])
            
            prompt_parts = [
                f"You are beam \"{strat_name}\" — generate an optimized CUDA kernel.",
                f"Direction: {strat_desc}",
                f"\nProblem shape: {shape_str} (FIXED — you may hard-code dimensions as compile-time constants).",
                "\nSpeedup is measured against FlashInfer, a production GPU library. "
                "You target ONE GPU (B200, sm_100a) and ONE shape.",
            ]
            
            prompt_parts.extend([
                f"\n## Naive reference kernel (starting point):\n"
                f"```cuda\n{kernel_slice}\n```",
                f"\n{launch_sig}",
                "\nBefore calling submit_kernel, explain in 2-3 sentences:\n"
                "1. What your planned change is\n"
                "2. Why it addresses the specific performance metrics without breaking correctness\n\n"
                "Then call submit_kernel with your complete .cu file.",
            ])
            initial_prompt = "\n\n".join(prompt_parts)

            messages = [{"role": "user", "content": initial_prompt}]
            best = None
            prev_inner_metrics = None
            submit_count = 0
            max_api_turns = MAX_INNER_TURNS + 4  # extra turns for non-submit tools

            for turn in range(max_api_turns):
                if submit_count >= MAX_INNER_TURNS:
                    break
                try:
                    response = await self._call_llm_with_tools_async(
                        messages=messages,
                        tools=ALL_TOOLS,
                        model=self.sub_model,
                        system=system_prompt,
                        temperature=0.5,
                    )
                except RuntimeError as e:
                    logger.error("Budget exceeded gen %s turn %d: %s",
                                 strat_name, turn, e)
                    break

                messages.append({"role": "assistant", "content": response.content})

                # Handle all tool calls (inspect_sass, read_file handled inline)
                submit_code, submit_block_id, aux_results = self._handle_tool_calls(
                    response, messages, profile_fn, strat_name, round_num,
                    1.0, prev_inner_metrics)

                # If only auxiliary tools were called, continue the conversation
                if submit_code is None and not submit_block_id:
                    if not aux_results:
                        # No tool calls at all
                        has_any_tool = any(b.type == "tool_use" for b in response.content)
                        if not has_any_tool:
                            logger.info("GEN [%s] turn %d: no tool call", strat_name, turn)
                            break
                    continue

                if submit_code is None:
                    # submit_kernel with empty code — error already appended
                    continue

                submit_count += 1
                result = await asyncio.get_event_loop().run_in_executor(
                    None, profile_fn, submit_code, strat_name, round_num)

                tool_result_text = self._format_tool_result(
                    result, 1.0, prev_inner_metrics)
                if result["compile_ok"] and result["correct"] and result.get("metrics"):
                    prev_inner_metrics = result["metrics"]
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": submit_block_id,
                     "content": tool_result_text}
                ]})

                logger.info("GEN [%s] submit %d: compile=%s correct=%s speedup=%.3fx",
                            strat_name, submit_count,
                            result["compile_ok"], result["correct"],
                            result.get("speedup", 0))

                if result["compile_ok"] and result["correct"]:
                    if best is None or result["speedup"] > best.speedup:
                        best = KernelCandidate(
                            code=submit_code,
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
            shape_str = str(self.env.problem_shapes[0])
            
            prompt = f"""\
You are an expert CUDA kernel optimizer targeting NVIDIA B200 (sm_100a, Blackwell).

Apply this optimization to the kernel below:

## Optimization: {strat_name}
{strat_desc}

## Context
Speedup is measured against FlashInfer, a production GPU library.
You target ONE GPU (B200, sm_100a) and ONE shape ({shape_str}).

## Naive reference kernel (starting point):
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
            # No strategy description — use minimal prompt
            prompt = f"""\
You are an expert CUDA kernel optimizer targeting NVIDIA B200 (sm_100a, Blackwell).

Apply the "{strat_name}" optimization to this kernel:

```cuda
{kernel_slice}
```

{launch_sig}

Return the COMPLETE .cu file in a single ```cuda code block. No explanations.
"""

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
            "Make ONE targeted change per submission so you can measure its impact. "
            "Do not bundle multiple unrelated optimizations.\n\n"
            "Before calling submit_kernel, explain:\n"
            "1. What single change you are making and why you expect it to help.\n\n"
            "Then call submit_kernel with your complete .cu file.")

        initial_prompt = "\n\n".join(prompt_parts)
        logger.info("REFINE PROMPT [%s]: %d chars", parent.strategy, len(initial_prompt))

        # ── Multi-turn tool-use conversation ──────────────────────────────
        messages = [{"role": "user", "content": initial_prompt}]
        best = None              # best KernelCandidate from inner loop
        last_error = ""
        prev_inner_metrics = metrics  # initialize to parent metrics for turn 0 deltas
        submit_count = 0
        max_api_turns = MAX_INNER_TURNS + 4  # extra turns for non-submit tools

        for turn in range(max_api_turns):
            if submit_count >= MAX_INNER_TURNS:
                break
            try:
                system_prompt = _build_refine_system_prompt(parent.speedup)
                response = await self._call_llm_with_tools_async(
                    messages=messages,
                    tools=ALL_TOOLS,
                    model=self.sub_model,
                    system=system_prompt,
                    temperature=0.4,
                )
            except RuntimeError as e:
                logger.error("Budget exceeded refine %s turn %d: %s",
                             parent.strategy, turn, e)
                break

            # Extract and log model's reasoning/thoughts
            text_blocks = [b.text for b in response.content if hasattr(b, 'text') and b.text.strip()]
            if text_blocks:
                reasoning = "\n".join(text_blocks)
                logger.info("\n🧠 MODEL THOUGHTS [%s turn %d]:\n%s\n", parent.strategy, turn, reasoning)

            # Append assistant message to conversation history
            messages.append({"role": "assistant", "content": response.content})

            # Handle all tool calls (inspect_sass, read_file handled inline)
            submit_code, submit_block_id, aux_results = self._handle_tool_calls(
                response, messages, profile_fn, parent.strategy, round_num,
                parent.speedup, prev_inner_metrics)

            # If only auxiliary tools were called, continue the conversation
            if submit_code is None and not submit_block_id:
                if not aux_results:
                    has_any_tool = any(b.type == "tool_use" for b in response.content)
                    if not has_any_tool:
                        text_parts = [b.text for b in response.content
                                      if hasattr(b, 'text')]
                        logger.info("REFINE [%s] turn %d: no tool call: %s",
                                    parent.strategy, turn,
                                    " ".join(text_parts)[:200])
                        break
                continue

            if submit_code is None:
                # submit_kernel with empty code — error already appended
                continue

            submit_count += 1

            # Profile the submission via callback
            if profile_fn:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, profile_fn, submit_code, parent.strategy, round_num)
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
                {"type": "tool_result", "tool_use_id": submit_block_id,
                 "content": tool_result_text}
            ]})

            logger.info("\n📊 FEEDBACK QUALITY DELIVERED [%s submit %d]:\n%s\n", parent.strategy, submit_count, tool_result_text)

            new_spd = result.get("speedup", 0)
            delta = new_spd - parent.speedup
            logger.info("📈 IMPROVEMENT TRACKER [%s submit %d]: compile=%s correct=%s | Base=%.3fx -> New=%.3fx (Delta: %+.3fx)",
                         parent.strategy, submit_count,
                         result["compile_ok"], result["correct"],
                         parent.speedup, new_spd, delta)

            # Track best viable result
            if result["compile_ok"] and result["correct"]:
                if best is None or result["speedup"] > best.speedup:
                    best = KernelCandidate(
                        code=submit_code,
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

    # ── Auxiliary tool handlers ────────────────────────────────────────────────

    def _handle_inspect_sass(self, cuda_code: str) -> str:
        """Compile code and return raw SASS disassembly via cuobjdump."""
        build_dir = Path(self.env.search_config.get("output", {}).get("output_dir", "outputs")) / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        kernel_file = build_dir / "_sass_inspect.cu"
        binary_file = build_dir / "_sass_inspect"

        # Write source — need harness for compilation, but we only care about SASS
        kernel_file.write_text(cuda_code)

        nvcc_flags = [
            "-O3", "-arch=sm_100a", "--use_fast_math", "-std=c++17",
            f"-I{PROJECT_ROOT / 'kernels' / 'common'}",
            "-c",  # compile only, no link — faster and avoids missing main()
            "-o", str(binary_file) + ".o",
        ]
        cmd = ["nvcc"] + nvcc_flags + [str(kernel_file)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                error_lines = [l for l in result.stderr.splitlines()
                               if 'error' in l.lower() and ('(' in l or ':' in l)]
                first_err = error_lines[0].strip() if error_lines else result.stderr[:300]
                return f"COMPILE ERROR (cannot inspect SASS):\n{first_err}"
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return f"Compilation failed: {e}"

        # Disassemble
        try:
            sass_result = subprocess.run(
                ["cuobjdump", "-sass", str(binary_file) + ".o"],
                capture_output=True, text=True, timeout=30,
            )
            if sass_result.returncode != 0:
                return f"cuobjdump failed: {sass_result.stderr[:300]}"
            sass = sass_result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return f"cuobjdump not available: {e}"

        # Truncate if very long (keep first 400 lines — enough for one kernel)
        lines = sass.splitlines()
        if len(lines) > 400:
            sass = "\n".join(lines[:400]) + f"\n... ({len(lines) - 400} more lines truncated)"

        return sass if sass.strip() else "No SASS output (empty binary?)"

    def _handle_read_file(self, path: str) -> str:
        """Read an allowed project file."""
        ALLOWED_PREFIXES = [
            "kernels/common/",
            "kernels/reference/",
        ]
        # Normalize and validate
        clean = path.strip().lstrip("/")
        if not any(clean.startswith(p) for p in ALLOWED_PREFIXES):
            return (f"Access denied: '{path}'. Allowed paths:\n"
                    "- kernels/common/nvfp4_utils.cuh\n"
                    "- kernels/common/b200_intrinsics.cuh\n"
                    "- kernels/reference/add_rmsnorm.cu\n"
                    "- kernels/reference/silu_mul.cu\n"
                    "- kernels/reference/nvfp4_quantize.cu")

        full_path = PROJECT_ROOT / clean
        if not full_path.exists():
            return f"File not found: {clean}"
        try:
            content = full_path.read_text()
            # Truncate very large files
            if len(content) > 12000:
                content = content[:12000] + "\n... (truncated)"
            return content
        except Exception as e:
            return f"Error reading {clean}: {e}"

    def _handle_tool_calls(self, response, messages, profile_fn, strategy_name,
                           round_num, parent_speedup, prev_inner_metrics):
        """Process all tool calls in a response. Returns (submit_result, updated_prev_metrics, had_submit).

        Non-submit tools (inspect_sass, read_file) are handled inline and their
        results appended to messages. Only submit_kernel triggers profiling.
        Returns the submit_kernel result dict if one was found, else None.
        """
        tool_results = []
        submit_result = None
        submit_code = None
        submit_block_id = None

        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "inspect_sass":
                code = block.input.get("cuda_code", "")
                if not code:
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": "Error: empty cuda_code.", "is_error": True,
                    })
                else:
                    logger.info("🔍 INSPECT_SASS [%s]: compiling for SASS dump", strategy_name)
                    sass_output = self._handle_inspect_sass(code)
                    logger.info("🔍 SASS [%s]: %d lines returned", strategy_name,
                                len(sass_output.splitlines()))
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": sass_output,
                    })

            elif block.name == "read_file":
                path = block.input.get("path", "")
                if not path:
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": "Error: empty path.", "is_error": True,
                    })
                else:
                    logger.info("📖 READ_FILE [%s]: %s", strategy_name, path)
                    content = self._handle_read_file(path)
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": content,
                    })

            elif block.name == "search_docs":
                query = block.input.get("query", "")
                if not query:
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": "Error: empty query.", "is_error": True,
                    })
                else:
                    from .cuda_docs import search_intrinsics
                    logger.info("📚 SEARCH_DOCS [%s]: %s", strategy_name, query)
                    doc_result = search_intrinsics(query)
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": doc_result,
                    })

            elif block.name == "submit_kernel":
                code = block.input.get("cuda_code", "")
                if not code:
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": "Error: empty cuda_code. Submit the complete .cu file.",
                        "is_error": True,
                    })
                else:
                    submit_code = code
                    submit_block_id = block.id

        # Handle submit_kernel separately (needs async profiling)
        # Return info for caller to handle
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return submit_code, submit_block_id, tool_results

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
