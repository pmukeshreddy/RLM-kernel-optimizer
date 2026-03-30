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

import anthropic
from anthropic import AsyncAnthropic

from .coder import build_coder_prompt
from .environment import RLMEnvironment, KernelCandidate
from .feedback import build_sandbox_feedback
from .feedback import _kernel_aliases, _kernel_operation_phrase
from .fixer import build_fixer_prompt
from .planner import (
    build_initial_plan_prompt,
    build_tree_plan_prompt,
    fallback_branches,
    parse_plan_response,
)
from .rag_retriever import init_knowledge_base
from .root_prompts import SYSTEM_PROMPT, combine_prompt
from .reflector import (
    _get_launch_signature,
    _format_profile_section,
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

SEARCH_PINECONE_TOOL = {
    "name": "search_pinecone",
    "description": (
        "Search the Pinecone knowledge index for CUDA optimization notes, prior "
        "experiments, compiler pitfalls, and kernel-specific guidance that the user "
        "already stored there. Use this when the local docs are not enough. "
        "Costs no submit_kernel turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Semantic search query to run against Pinecone.",
            },
            "top_k": {
                "type": "integer",
                "description": "Optional number of matches to return.",
            },
        },
        "required": ["query"],
    },
}

ALL_TOOLS = [
    SUBMIT_KERNEL_TOOL,
    INSPECT_SASS_TOOL,
    READ_FILE_TOOL,
    SEARCH_DOCS_TOOL,
    SEARCH_PINECONE_TOOL,
]

REFINE_SYSTEM_PROMPT = f"""\
You are a CUDA kernel optimization agent. You have {{turns}} submit_kernel calls.

Your speedup is measured against FlashInfer, a production GPU library.
You target a single GPU (B200, sm_100a) and a single problem shape — use this to your advantage.

Available tools (only submit_kernel counts toward your turn limit):
- submit_kernel: compile, test correctness, and benchmark your kernel
- inspect_sass: compile code and see the raw SASS assembly (instruction-level view)
- read_file: read project header files (nvfp4_utils.cuh, b200_intrinsics.cuh) or reference kernels
- search_docs: look up CUDA intrinsic signatures and usage (fp4, fp8, warp, fast math, memory)
- search_pinecone: query the user's Pinecone knowledge index for project-specific guidance

Target hardware — NVIDIA B200 (sm_100a, Blackwell):
- HBM3e: 8 TB/s bandwidth, 192 GB
- L2 cache: 126 MB — benchmark uses L2 cache cycling (data is COLD every iteration)
- 148 SMs, 228 KB shared memory per SM, 255 registers per thread
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

def _build_refine_system_prompt(speedup: float, prev_metrics: dict = None) -> str:
    """Build REFINE_SYSTEM_PROMPT with data-driven constraint based on SASS analysis."""
    if speedup >= 1.0 and prev_metrics:
        # Already beating baseline — structural changes regress at this point.
        # Guide model toward the specific instruction-level bottlenecks SASS identified.
        cm = prev_metrics.get("_compiler", {})
        sass_total = cm.get("sass_total_instructions", 0)
        bra = cm.get("sass_bra", 0)
        fadd = cm.get("sass_fadd", 0)
        fmul = cm.get("sass_fmul", 0)
        ffma = cm.get("sass_ffma", 0)
        stg32 = cm.get("sass_stg_32", 0)

        bottlenecks = []
        if fadd > 0 and fmul > 0 and (fadd + fmul) > ffma:
            bottlenecks.append(f"FADD={fadd}+FMUL={fmul} not fused into FFMA — use fmaf()")
        if bra > 3:
            bottlenecks.append(f"BRA={bra} branch instructions — use branchless alternatives")
        if stg32 > 1:
            bottlenecks.append(f"STG.32={stg32} narrow stores — pack into wider writes")

        fsetp = cm.get("sass_fsetp", 0)
        sel = cm.get("sass_sel", 0)
        if fsetp + sel > 10:
            bottlenecks.append(f"FSETP={fsetp}+SEL={sel} predicate chains — replace software FP4 quantization with hardware intrinsics")

        f2f = cm.get("sass_f2f", 0)
        if f2f > 8:
            bottlenecks.append(f"F2F={f2f} type conversions — keep math in native bf16 pairs")

        prmt = cm.get("sass_prmt", 0)
        lop3 = cm.get("sass_lop3", 0)
        shf_cnt = cm.get("sass_shf", 0)
        # Only penalize PRMT/LOP3/SHF if we are still doing software FP4 quantization
        # (indicated by high FSETP + SEL). If FSETP is low, PRMT is being used correctly
        # to pack uint8_t values into uint32_t/uint64_t for vectorized stores!
        if (prmt + lop3 + shf_cnt > 8) and (fsetp + sel >= 4):
            bottlenecks.append(f"PRMT={prmt}+LOP3={lop3}+SHF={shf_cnt} bit manipulation — use __nv_cvt_bfloat16raw2_to_fp4x2 instead of manual packing")

        if bottlenecks:
            hint = "; ".join(bottlenecks)
            constraint = (
                f"- You are ABOVE baseline ({speedup:.2f}x). Your remaining bottleneck is instruction-level, "
                f"not structural. The SASS shows: {hint}\n"
                f"- Each change should target reducing specific SASS instruction counts."
            )
        else:
            constraint = (
                f"- You are ABOVE baseline ({speedup:.2f}x). Prefer surgical instruction-level "
                f"changes over structural rewrites."
            )
    else:
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

        self.root_model    = cfg["models"].get("planner_model", cfg["models"]["root_model"])
        self.sub_model     = cfg["models"].get("coder_model", cfg["models"]["sub_model"])
        self.fixer_model   = cfg["models"].get("fixer_model", cfg["models"]["sub_model"])
        self.combine_model = cfg["models"]["combine_model"]
        self.beam_width    = cfg["beam"]["width"]
        self.refine_rounds = cfg["beam"]["refine_rounds"]
        self.combine_top_k = cfg["beam"]["combine_top_k"]
        self.tree_speedup_threshold = float(cfg["beam"].get("tree_speedup_threshold", 1.0))
        self.tree_branching_factor = int(cfg["beam"].get("tree_branching_factor", 2))
        self.max_tokens    = cfg["cost_control"]["max_tokens_per_sub_call"]
        self.rag = init_knowledge_base(cfg.get("rag", {}))

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

    def _planner_baseline_context(self) -> str:
        env = self.env
        if env.baseline_naive_us and env.baseline_us_reported:
            rows = env.problem_shapes[0][0]
            sm_count = env.hw_spec.get("sm", {}).get("count", 148)
            cm = env.baseline_compiler_metrics
            cm_str = cm.summary_str() if cm else "unavailable"
            return (
                f"BASELINE PROFILER DATA (reference kernel):\n"
                f"  Naive kernel timing: {env.baseline_naive_us:.3f} us\n"
                f"  FlashInfer timing:   {env.baseline_us_reported:.3f} us\n"
                f"  Compiler: {cm_str}\n"
                f"  Grid: {rows} blocks launched on {sm_count} SMs"
                f"{' — some SMs get zero work' if rows < sm_count else ''}\n"
            )
        return "BASELINE PROFILER DATA: unavailable — analyze kernel source to infer bottleneck type.\n"

    def _search_pinecone_context(self, queries: list[str], top_k: int = 3) -> str:
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries:
            return "No Pinecone query provided."
        matches = self.rag.search_many(clean_queries[:4], top_k=top_k)
        return self.rag.format_matches(matches)

    def _initial_plan_queries(self) -> list[str]:
        env = self.env
        shape = "x".join(str(dim) for dim in env.problem_shapes[0])
        operation = _kernel_operation_phrase(env.kernel_type)
        aliases = ", ".join(_kernel_aliases(env.kernel_type)[:4])
        return [
            (
                f"Operation: {operation}. "
                f"Shape: {shape}. Aliases: {aliases}. Need: production CUDA kernel source_code."
            ),
            f"{operation} FlashInfer bottleneck CUDA source code",
            f"{operation} vectorized loads stores bf16 fp4 CUDA source code",
        ]

    def _expand_tree_plans(self, parent: KernelCandidate) -> list[dict]:
        rag_context = self._search_pinecone_context(
            parent.plan_branch.get("rag_queries")
            or [f"{self.env.kernel_type} {parent.strategy} next optimization"]
        )
        feedback = build_sandbox_feedback(
            {
                "compile_ok": parent.compile_ok,
                "correct": parent.correct,
                "speedup": parent.speedup,
                "metrics": parent.metrics,
                "error": parent.compile_error,
            },
            parent_speedup=parent.speedup,
            prev_inner_metrics=parent.prev_metrics,
            kernel_type=self.env.kernel_type,
        )
        prompt = build_tree_plan_prompt(
            kernel_type=self.env.kernel_type,
            operation=_kernel_operation_phrase(self.env.kernel_type),
            aliases=_kernel_aliases(self.env.kernel_type),
            problem_shape=self.env.problem_shapes[0],
            parent_strategy=parent.strategy,
            parent_speedup=parent.speedup,
            kernel_src=parent.best_code or parent.code,
            feedback_summary=feedback.planner_summary(),
            rag_context=rag_context,
            branch_count=self.tree_branching_factor,
        )
        response, _, _ = self._call_llm(prompt, model=self.root_model, temperature=0.2)
        return parse_plan_response(
            response,
            count=self.tree_branching_factor,
            prefix=f"{parent.strategy}_child",
            parent_strategy=parent.strategy,
        )

    # ── Round 0: Decomposition ────────────────────────────────────────────────

    def decompose(self) -> list:
        env = self.env
        num_strategies = self.beam_width * 2
        rag_context = self._search_pinecone_context(self._initial_plan_queries())
        prompt = build_initial_plan_prompt(
            kernel_type=env.kernel_type,
            operation=_kernel_operation_phrase(env.kernel_type),
            aliases=_kernel_aliases(env.kernel_type),
            problem_shape=env.problem_shapes[0],
            kernel_src=env.kernel_src,
            baseline_context=self._planner_baseline_context(),
            rag_context=rag_context,
            branch_count=num_strategies,
        )

        logger.info("Planner: generating %d root branches for %s", num_strategies, env.kernel_type)
        response, _, _ = self._call_llm(prompt, model=self.root_model, temperature=0.2)
        strategies = parse_plan_response(
            response,
            count=num_strategies,
            prefix="root_plan",
        )
        if strategies:
            logger.info("Planner produced %d branches", len(strategies))
            return strategies

        logger.warning("Planner returned no usable branches, using fallback plans")
        return fallback_branches(num_strategies, prefix="root_plan")

    # ── Sub-LLM beam generation (parallel) ───────────────────────────────────

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        strategy_name: str,
        round_num: int,
        profile_fn,
        model_id: str,
        comparison_speedup: float = 0.0,
        prev_inner_metrics: dict | None = None,
        strategy_context: str = "",
        plan_branch: dict | None = None,
        parent_candidate: KernelCandidate | None = None,
    ) -> KernelCandidate:
        messages = [{"role": "user", "content": initial_prompt}]
        best = None
        last_error = ""
        feedback_route = ""
        submit_count = 0
        max_api_turns = MAX_INNER_TURNS + 4
        best_speedup = comparison_speedup
        plan_branch = dict(plan_branch or {})

        for turn in range(max_api_turns):
            if submit_count >= MAX_INNER_TURNS:
                break

            system_prompt = _build_refine_system_prompt(best_speedup, prev_inner_metrics)
            try:
                response = await self._call_llm_with_tools_async(
                    messages=messages,
                    tools=ALL_TOOLS,
                    model=model_id,
                    system=system_prompt,
                    temperature=0.4,
                )
            except RuntimeError as exc:
                logger.error("Budget exceeded for %s turn %d: %s", strategy_name, turn, exc)
                break

            text_blocks = [
                block.text for block in response.content
                if hasattr(block, "text") and block.text.strip()
            ]
            if text_blocks:
                logger.info(
                    "\nAGENT [%s turn %d]:\n%s\n",
                    strategy_name,
                    turn,
                    "\n".join(text_blocks),
                )

            messages.append({"role": "assistant", "content": response.content})

            submit_code, submit_block_id, aux_results = self._handle_tool_calls(
                response, messages, profile_fn, strategy_name, round_num,
                max(best_speedup, comparison_speedup), prev_inner_metrics,
            )

            if submit_code is None and not submit_block_id:
                has_any_tool = any(block.type == "tool_use" for block in response.content)
                if aux_results:
                    messages.append({"role": "user", "content": aux_results})
                    continue
                if not has_any_tool:
                    break
                continue

            if submit_code is None:
                if aux_results:
                    messages.append({"role": "user", "content": aux_results})
                continue

            submit_count += 1
            if profile_fn:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, profile_fn, submit_code, strategy_name, round_num
                )
            else:
                result = {
                    "compile_ok": False,
                    "correct": False,
                    "speedup": 0.0,
                    "metrics": {},
                    "error": "No profiler available",
                    "bottleneck": "unknown",
                }

            feedback = build_sandbox_feedback(
                result=result,
                parent_speedup=max(best_speedup, comparison_speedup),
                prev_inner_metrics=prev_inner_metrics,
                kernel_type=self.env.kernel_type,
            )
            feedback_route = feedback.route

            if result["compile_ok"] and result["correct"] and result.get("metrics"):
                prev_inner_metrics = result["metrics"]

            all_results = list(aux_results) + [
                {
                    "type": "tool_result",
                    "tool_use_id": submit_block_id,
                    "content": feedback.to_tool_result_json(),
                }
            ]
            messages.append({"role": "user", "content": all_results})

            logger.info(
                "SANDBOX [%s submit %d]: route=%s speedup=%.3fx compile=%s correct=%s",
                strategy_name,
                submit_count,
                feedback.route,
                result.get("speedup", 0.0),
                result.get("compile_ok"),
                result.get("correct"),
            )

            if result["compile_ok"] and result["correct"]:
                if best is None or result["speedup"] > best.speedup:
                    best = KernelCandidate(
                        code=submit_code,
                        strategy=strategy_name,
                        round_num=round_num,
                        compile_ok=True,
                        correct=True,
                        speedup=result["speedup"],
                        metrics=result.get("metrics", {}),
                        bottleneck=result.get("bottleneck", "unknown"),
                        prev_metrics=parent_candidate.metrics if parent_candidate else None,
                        parent_strategy=(
                            parent_candidate.strategy if parent_candidate else plan_branch.get("parent_strategy", "")
                        ),
                        plan_branch=dict(plan_branch),
                        feedback_route=feedback.route,
                    )
                    best.strategy_context = strategy_context
                    best.best_code = submit_code
                    best.best_speedup = result["speedup"]
                    best_speedup = result["speedup"]
            else:
                last_error = result.get("error", "") or feedback.root_cause

        if best:
            return best

        failed = KernelCandidate(
            code="",
            strategy=strategy_name,
            round_num=round_num,
            compile_ok=False,
            prev_metrics=parent_candidate.metrics if parent_candidate else None,
            parent_strategy=(
                parent_candidate.strategy if parent_candidate else plan_branch.get("parent_strategy", "")
            ),
            plan_branch=dict(plan_branch),
            feedback_route=feedback_route,
        )
        failed.compile_error = last_error or "All inner refinement attempts failed"
        failed.strategy_context = strategy_context
        return failed

    async def _generate_single_beam(
        self,
        strategy,
        kernel_slice: str,
        current_metrics: dict = None,
        round_num: int = 0,
        profile_fn=None,
    ) -> KernelCandidate:
        if isinstance(strategy, dict):
            plan_branch = dict(strategy)
            strat_name = plan_branch.get("name", "unknown")
            strat_desc = plan_branch.get("change_summary") or plan_branch.get("what", "")
        else:
            strat_name = str(strategy)
            strat_desc = ""
            plan_branch = {
                "name": strat_name,
                "goal": strat_name,
                "what": strat_desc,
                "change_summary": strat_desc,
                "expected_signal": "Sandbox output improves.",
                "rag_queries": [],
            }

        launch_sig = _get_launch_signature(self.env.kernel_type)

        if profile_fn and strat_desc:
            rag_context = self._search_pinecone_context(
                plan_branch.get("rag_queries")
                or [f"{self.env.kernel_type} {strat_name} CUDA optimization"]
            )
            current_profile = (
                _format_profile_section(current_metrics, round_num)
                if current_metrics else ""
            )
            initial_prompt = build_coder_prompt(
                plan_branch=plan_branch,
                kernel_code=kernel_slice,
                launch_signature=launch_sig,
                rag_context=rag_context,
                current_profile=current_profile,
            )
            return await self._run_agent_loop(
                initial_prompt=initial_prompt,
                strategy_name=strat_name,
                round_num=round_num,
                profile_fn=profile_fn,
                model_id=self.sub_model,
                comparison_speedup=0.0,
                prev_inner_metrics=current_metrics,
                strategy_context=strat_desc,
                plan_branch=plan_branch,
            )

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
            plan_branch=plan_branch,
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
        tasks = []
        for candidate in survivors:
            if candidate.speedup >= self.tree_speedup_threshold:
                child_plans = self._expand_tree_plans(candidate)
                for child_plan in child_plans:
                    tasks.append(
                        self._refine_single_beam(
                            candidate,
                            round_num,
                            profile_fn=profile_fn,
                            plan_branch=child_plan,
                            fixer_mode=False,
                        )
                    )
            else:
                feedback = build_sandbox_feedback(
                    {
                        "compile_ok": candidate.compile_ok,
                        "correct": candidate.correct,
                        "speedup": candidate.speedup,
                        "metrics": candidate.metrics,
                        "error": candidate.compile_error,
                    },
                    parent_speedup=candidate.speedup,
                    prev_inner_metrics=candidate.prev_metrics,
                    kernel_type=self.env.kernel_type,
                )
                repair_plan = {
                    "name": f"{candidate.strategy}_repair",
                    "goal": "Repair the failing or below-baseline branch.",
                    "what": feedback.next_action,
                    "change_summary": feedback.next_action,
                    "expected_signal": "Compilation succeeds, correctness holds, and speed improves.",
                    "rag_queries": feedback.rag_queries,
                    "planner_notes": feedback.planner_summary(),
                    "parent_strategy": candidate.strategy,
                    "tree_ready": False,
                }
                tasks.append(
                    self._refine_single_beam(
                        candidate,
                        round_num,
                        profile_fn=profile_fn,
                        plan_branch=repair_plan,
                        fixer_mode=True,
                        feedback=feedback,
                    )
                )
        return list(await asyncio.gather(*tasks))

    async def _refine_single_beam(
        self,
        parent: 'KernelCandidate',
        round_num: int,
        profile_fn=None,
        plan_branch: dict | None = None,
        fixer_mode: bool = False,
        feedback=None,
    ) -> 'KernelCandidate':
        metrics = parent.metrics or {}
        launch_sig = _get_launch_signature(self.env.kernel_type)
        base_code = parent.best_code or parent.code
        plan_branch = dict(plan_branch or parent.plan_branch or {})
        plan_branch.setdefault("parent_strategy", parent.strategy)
        branch_name = plan_branch.get("name", "repair")
        strategy_name = f"{parent.strategy}__{branch_name}_r{round_num}"

        if feedback is None:
            feedback = build_sandbox_feedback(
                {
                    "compile_ok": parent.compile_ok,
                    "correct": parent.correct,
                    "speedup": parent.speedup,
                    "metrics": parent.metrics,
                    "error": parent.compile_error,
                },
                parent_speedup=parent.speedup,
                prev_inner_metrics=parent.prev_metrics,
                kernel_type=self.env.kernel_type,
            )

        rag_queries = plan_branch.get("rag_queries") or feedback.rag_queries
        rag_context = self._search_pinecone_context(
            rag_queries or [f"{self.env.kernel_type} {branch_name} CUDA optimization"]
        )

        if fixer_mode:
            initial_prompt = build_fixer_prompt(
                plan_branch=plan_branch,
                kernel_code=base_code,
                launch_signature=launch_sig,
                rag_context=rag_context,
                feedback_json=feedback.to_tool_result_json(),
            )
            model_id = self.fixer_model
        else:
            current_profile = _format_profile_section(metrics, round_num) if metrics else ""
            initial_prompt = build_coder_prompt(
                plan_branch=plan_branch,
                kernel_code=base_code,
                launch_signature=launch_sig,
                rag_context=rag_context,
                current_profile=current_profile,
            )
            model_id = self.sub_model

        return await self._run_agent_loop(
            initial_prompt=initial_prompt,
            strategy_name=strategy_name,
            round_num=round_num,
            profile_fn=profile_fn,
            model_id=model_id,
            comparison_speedup=parent.speedup,
            prev_inner_metrics=metrics,
            strategy_context=plan_branch.get("change_summary") or plan_branch.get("what", ""),
            plan_branch=plan_branch,
            parent_candidate=parent,
        )

    # ── Auxiliary tool handlers ────────────────────────────────────────────────

    def _handle_inspect_sass(self, cuda_code: str) -> str:
        """Compile code and return raw SASS disassembly via cuobjdump."""
        import hashlib
        build_dir = Path(self.env.search_config.get("output", {}).get("output_dir", "outputs")) / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        # Unique filename to avoid clobber when beams run concurrently
        uid = hashlib.md5(cuda_code.encode()).hexdigest()[:8]
        kernel_file = build_dir / f"_sass_inspect_{uid}.cu"
        binary_file = build_dir / f"_sass_inspect_{uid}"

        # Write source — need harness for compilation, but we only care about SASS
        kernel_file.write_text(cuda_code)

        nvcc_flags = [
            "-O3", "-arch=sm_100a", "--use_fast_math", "-std=c++17",
            f"-I{PROJECT_ROOT / 'kernels' / 'common'}",
            f"-I{PROJECT_ROOT}",
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
        """Process all tool calls in a response.

        Returns (submit_code, submit_block_id, aux_results).
        aux_results is a list of tool_result dicts for non-submit tools.

        IMPORTANT: Does NOT append to messages — caller is responsible for
        combining aux_results with submit_kernel result into ONE user message
        to avoid consecutive user messages (Anthropic API requirement).
        """
        aux_results = []
        submit_code = None
        submit_block_id = None

        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "inspect_sass":
                code = block.input.get("cuda_code", "")
                if not code:
                    aux_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": "Error: empty cuda_code.", "is_error": True,
                    })
                else:
                    logger.info("🔍 INSPECT_SASS [%s]: compiling for SASS dump", strategy_name)
                    sass_output = self._handle_inspect_sass(code)
                    logger.info("🔍 SASS [%s]: %d lines returned", strategy_name,
                                len(sass_output.splitlines()))
                    aux_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": sass_output,
                    })

            elif block.name == "read_file":
                path = block.input.get("path", "")
                if not path:
                    aux_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": "Error: empty path.", "is_error": True,
                    })
                else:
                    logger.info("📖 READ_FILE [%s]: %s", strategy_name, path)
                    content = self._handle_read_file(path)
                    aux_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": content,
                    })

            elif block.name == "search_docs":
                query = block.input.get("query", "")
                if not query:
                    aux_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": "Error: empty query.", "is_error": True,
                    })
                else:
                    from .cuda_docs import search_intrinsics
                    logger.info("📚 SEARCH_DOCS [%s]: %s", strategy_name, query)
                    doc_result = search_intrinsics(query)
                    aux_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": doc_result,
                    })

            elif block.name == "search_pinecone":
                query = block.input.get("query", "")
                top_k = block.input.get("top_k")
                if not query:
                    aux_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": "Error: empty query.", "is_error": True,
                    })
                else:
                    logger.info("🧠 SEARCH_PINECONE [%s]: %s", strategy_name, query)
                    matches = self.rag.search_many([query], top_k=top_k or 3)
                    aux_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": self.rag.format_matches(matches),
                    })

            elif block.name == "submit_kernel":
                code = block.input.get("cuda_code", "")
                if not code:
                    aux_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": "Error: empty cuda_code. Submit the complete .cu file.",
                        "is_error": True,
                    })
                else:
                    submit_code = code
                    submit_block_id = block.id

        return submit_code, submit_block_id, aux_results

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
