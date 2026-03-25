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
