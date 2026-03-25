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
        """Select strategies using kernel-type-aware detection + LLM validation.

        1. Get ideal strategies for this kernel type (from strategy_bank)
        2. Ask the LLM to validate/reorder based on actual kernel analysis
        3. Parse LLM's JSON response, fall back to kernel-aware defaults
        """
        env = self.env
        hot_loop = env.get_hot_loop_src()
        # detect_missing_optimizations now returns kernel-aware ideal strategies
        ideal_strategies = env.detect_missing_optimizations()
        enabled = env.search_config["strategies"]["enabled"]

        if not ideal_strategies:
            ideal_strategies = enabled[:self.beam_width]

        prompt = decompose_prompt(
            env_summary=env.state_summary(),
            kernel_slice=hot_loop,
            missing_opts=ideal_strategies,
        )

        logger.info("Round 0: kernel-type=%s ideal_strategies=%s",
                     env.kernel_type, ideal_strategies)
        response, _, _ = self._call_llm(prompt, model=self.root_model, temperature=0.2)
        logger.debug("Decomposition response: %s", response[:300])

        # Try to parse LLM's strategy recommendations as JSON list
        import json
        parsed = []
        # Look for a JSON array in the response
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                raw = json.loads(json_match.group())
                parsed = [s for s in raw if isinstance(s, str) and s in enabled]
            except (json.JSONDecodeError, TypeError):
                pass

        # If LLM returned valid strategies, use them (but validate against enabled list)
        if len(parsed) >= self.beam_width:
            logger.info("Using LLM-selected strategies: %s", parsed[:self.beam_width])
            return parsed[:self.beam_width]

        # Otherwise use the kernel-aware ideal strategies
        logger.info("Using kernel-aware ideal strategies: %s",
                     ideal_strategies[:self.beam_width])
        return ideal_strategies[:self.beam_width]

    # ── Sub-LLM beam generation (parallel) ───────────────────────────────────

    async def _generate_single_beam(
        self,
        strategy: str,
        kernel_slice: str,
        current_metrics: dict = None,
        round_num: int = 0,
    ) -> KernelCandidate:
        prompt = get_prompt_for_strategy(
            strategy=strategy,
            kernel_slice=kernel_slice,
            hw_spec=self.env.hw_spec,
            current_metrics=current_metrics,
        )

        try:
            response, _, _ = await self._call_llm_async(
                prompt, model=self.sub_model, temperature=0.4
            )
        except RuntimeError as e:
            logger.error("Budget exceeded during beam %s: %s", strategy, e)
            return KernelCandidate(code="", strategy=strategy, round_num=round_num)

        code = self._extract_cuda_code(response)
        if not code:
            logger.warning("No CUDA code extracted for strategy=%s (response starts: %s)",
                           strategy, response[:100])
        return KernelCandidate(
            code=code,
            strategy=strategy,
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

    # ── Refinement rounds ─────────────────────────────────────────────────────

    async def refine_beams(self, survivors: list, round_num: int) -> list:
        tasks = []
        for candidate in survivors:
            strategy     = self._bottleneck_to_strategy(candidate.bottleneck, candidate.strategy)
            # Pass the full candidate code so the LLM returns a complete compilable file
            kernel_slice = candidate.code if candidate.code else self.env.kernel_src
            tasks.append(
                self._generate_single_beam(
                    strategy=strategy,
                    kernel_slice=kernel_slice,
                    current_metrics=candidate.metrics,
                    round_num=round_num,
                )
            )
        return list(await asyncio.gather(*tasks))

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

    # ── Compile-error fix loop ─────────────────────────────────────────────────

    def fix_compile_error(self, broken_code: str, error_msg: str) -> str:
        """Send compile error back to LLM and ask it to fix the code."""
        prompt = f"""\
The following CUDA code failed to compile. Fix the error and return the corrected COMPLETE .cu file.

Compilation error:
```
{error_msg}
```

Broken code:
```cuda
{broken_code}
```

CRITICAL: Return ONLY the fixed COMPLETE .cu file in a single ```cuda code block. Keep all #includes, kernel functions, and launch_* wrapper. No explanations.
"""
        try:
            response, _, _ = self._call_llm(prompt, model=self.sub_model, temperature=0.2)
        except RuntimeError:
            return ""
        return self._extract_cuda_code(response)

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

    def run_decompose(self) -> list:
        return self.decompose()

    def run_generate_beams(
        self, strategies: list, kernel_slice: str,
        current_metrics: dict = None, round_num: int = 0,
    ) -> list:
        return asyncio.run(
            self.generate_beams(strategies, kernel_slice, current_metrics, round_num)
        )

    def run_refine_beams(self, survivors: list, round_num: int) -> list:
        return asyncio.run(self.refine_beams(survivors, round_num))
