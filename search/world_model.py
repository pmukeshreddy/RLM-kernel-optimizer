"""
world_model.py — K-Search world model for scoring and evolving optimization strategies.

The world model:
1. INIT: Proposes scored optimization strategies given kernel + hardware context
2. EVOLVE: After each action cycle, updates scores, inserts new strategies, prunes dead ones
3. FAILURE: When a strategy fails completely, downgrades it and proposes alternatives

Based on K-Search (arxiv 2602.19128): "LLM Kernel Generation via Co-Evolving
Intrinsic World Model."
"""

from __future__ import annotations
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Retry WM calls up to 3 times if JSON parsing fails
WM_MAX_RETRIES = 3

WM_SYSTEM = """\
You are the WORLD MODEL for a CUDA kernel optimization search on NVIDIA B200 (Blackwell).
You score optimization strategies, propose new ones based on profiler results, and prune dead ends.
Always respond with ONLY valid JSON — no markdown, no explanation, no code blocks."""


def _parse_json(text: str) -> Optional[dict | list]:
    """Extract JSON from LLM response (tolerant of markdown fences)."""
    # Strip markdown code fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Try parsing the whole thing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON array or object
    for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return None


class WorldModel:
    """K-Search world model: scores strategies, evolves the search tree."""

    def __init__(self, engine, kernel_src: str, kernel_type: str,
                 problem_shape: tuple, hw_summary: str):
        self.engine = engine
        self.kernel_src = kernel_src
        self.kernel_type = kernel_type
        self.problem_shape = problem_shape
        self.hw_summary = hw_summary

    # ── INIT: Propose scored strategies ────────────────────────────────────

    def init_strategies(self, n: int = 8) -> list[dict]:
        """Ask LLM to propose n scored optimization strategies.

        Returns list of dicts: {name, what, score, difficulty, rationale}
        """
        prompt = f"""\
Analyze this CUDA kernel and propose exactly {n} distinct optimization strategies.
Rank them by expected impact against FlashInfer (a production GPU library).

CONTEXT:
- Kernel type: {self.kernel_type}
- Problem shape: {self.problem_shape} (FIXED — can be hard-coded)
- Target: {self.hw_summary}
- Competition: FlashInfer — already well-optimized for general use.
  Your advantage: FlashInfer targets many GPUs/shapes. You target ONE GPU + ONE shape.
- These kernels are memory-bound (L2 cache cycled, data from HBM every time).
  Compute-only optimizations will NOT help.

DO NOT propose these generic strategies (FlashInfer already does them, score=0):
- "vectorized loads" / "use uint4/float4" — already standard in FlashInfer
- "warp shuffle reduction" / "__shfl_xor_sync" — already standard
- "single pass fusion" — already standard
- "fast math intrinsics" / "__expf" — already standard
- "__ldg read-only cache" — already standard
These will NOT beat FlashInfer. Score them 0.0 if you include them.

INSTEAD propose techniques that EXPLOIT your unique advantages:
- Shape specialization: hard-code dimensions {self.problem_shape} as compile-time
  constants, fully unroll all loops to exact trip counts, eliminate every branch
- __launch_bounds__ with low register budget (32-48 regs): maximize occupancy for
  memory-bound kernels — higher occupancy hides HBM latency better
- Hardware FP4/FP8 intrinsics: __nv_cvt_float_to_fp8, hardware FP4 packing —
  single-cycle vs ~20 instructions of manual bit manipulation
- Optimal launch config: tune block size and items-per-thread for this exact problem size
- Multi-row per block: share loaded data (weights, scales) across rows
- Vectorized packed stores: accumulate FP4 output into uint2/uint4, store in one transaction
- Register-only datapath: all computation in registers, no shared memory intermediates

```cuda
{self.kernel_src}
```

For each strategy:
- name: short identifier (snake_case)
- what: one-line description of the concrete code change
- score: 0.0-1.0 (probability of significant speedup over FlashInfer)
- difficulty: 1-5 (1=straightforward, 5=very hard to implement correctly)
- rationale: why this beats FlashInfer specifically (not just general CUDA advice)

Return as JSON array, highest score first:
[
  {{"name": "...", "what": "...", "score": 0.9, "difficulty": 2, "rationale": "..."}},
  ...
]"""

        for attempt in range(WM_MAX_RETRIES):
            response = self.engine.call_llm_sync(
                prompt, system=WM_SYSTEM, temperature=0.4)
            parsed = _parse_json(response)
            if isinstance(parsed, list) and len(parsed) > 0:
                # Validate and normalize
                strategies = []
                for s in parsed:
                    if not isinstance(s, dict) or "name" not in s:
                        continue
                    strategies.append({
                        "name": s.get("name", "unknown"),
                        "what": s.get("what", s.get("description", "")),
                        "score": float(s.get("score", 0.5)),
                        "difficulty": int(s.get("difficulty", 3)),
                        "rationale": s.get("rationale", ""),
                    })
                if strategies:
                    strategies.sort(key=lambda x: -x["score"])
                    logger.info("WM init: %d strategies proposed", len(strategies))
                    for s in strategies:
                        logger.info("  [%.2f d=%d] %s: %s",
                                    s["score"], s["difficulty"],
                                    s["name"], s["what"][:80])
                    return strategies

            logger.warning("WM init attempt %d: failed to parse JSON", attempt + 1)

        logger.error("WM init failed after %d retries — falling back to decompose",
                     WM_MAX_RETRIES)
        return []

    # ── EVOLVE: Update tree after action cycle results ─────────────────────

    def evolve(self, tree_json: str, results_summary: str) -> dict:
        """After a round of action cycles, ask LLM to evolve the tree.

        Returns dict with: updates, inserts, deletes
        """
        prompt = f"""\
You are the world model for a CUDA kernel optimization search.

## Current search tree (JSON):
{tree_json}

## Last round results:
{results_summary}

## Kernel context:
- Type: {self.kernel_type}, Shape: {self.problem_shape}
- Target: {self.hw_summary}
- Competition: FlashInfer (production GPU library)

## Instructions:
Based on the profiler results, update the search tree:

IMPORTANT: Do NOT propose generic strategies that FlashInfer already does (vectorized loads,
warp shuffles, fast math, __ldg). These score 0. Instead propose strategies that exploit
shape specialization ({self.problem_shape}), hardware intrinsics, launch config tuning,
__launch_bounds__, or multi-row data reuse.

1. UPDATES: Re-score existing OPEN strategies based on what we learned.
   - If a sibling strategy succeeded, related approaches should score higher.
   - If a strategy failed with specific errors, lower its score.
   - Generic strategies (vectorized loads, warp shuffles) should be scored 0.0.
   - Shape-specialized and hardware-intrinsic strategies should score highest.

2. INSERTS: Propose 1-3 NEW strategies as children of successful (CLOSED) nodes.
   - Each new strategy should build ON TOP of the parent's optimization.
   - Focus on what the profiler data reveals as the remaining bottleneck.
   - New strategies must be DIFFERENT from existing tree nodes.
   - Do NOT insert generic strategies — only shape-specific or hardware-specific ones.

3. DELETES: Remove OPEN strategies that are now clearly wrong or dominated.
   - Only delete leaf nodes with no children.

Return as JSON:
{{
  "updates": [
    {{"node_id": "...", "score": 0.7, "reason": "..."}}
  ],
  "inserts": [
    {{"parent_id": "...", "name": "...", "what": "...", "score": 0.8, "difficulty": 3}}
  ],
  "deletes": [
    {{"node_id": "...", "reason": "..."}}
  ],
  "analysis": "1-2 sentence summary of what we learned this round"
}}"""

        for attempt in range(WM_MAX_RETRIES):
            response = self.engine.call_llm_sync(
                prompt, system=WM_SYSTEM, temperature=0.3)
            parsed = _parse_json(response)
            if isinstance(parsed, dict):
                edits = {
                    "updates": parsed.get("updates", []),
                    "inserts": parsed.get("inserts", []),
                    "deletes": parsed.get("deletes", []),
                }
                analysis = parsed.get("analysis", "")
                n_ops = (len(edits["updates"]) + len(edits["inserts"])
                         + len(edits["deletes"]))
                logger.info("WM evolve: %d ops (upd=%d ins=%d del=%d) — %s",
                            n_ops, len(edits["updates"]),
                            len(edits["inserts"]), len(edits["deletes"]),
                            analysis[:120])
                return edits

            logger.warning("WM evolve attempt %d: failed to parse", attempt + 1)

        logger.warning("WM evolve failed — no tree updates this round")
        return {"updates": [], "inserts": [], "deletes": []}

    # ── FAILURE: Handle failed action cycles ───────────────────────────────

    def note_failure(self, tree_json: str, failed_nodes_summary: str) -> dict:
        """When strategies fail completely, ask LLM to adapt.

        Returns same format as evolve().
        """
        prompt = f"""\
You are the world model for a CUDA kernel optimization search.

## Current search tree:
{tree_json}

## Failed strategies this round:
{failed_nodes_summary}

## Instructions:
These strategies failed after multiple attempts. Update the tree:

1. UPDATES: Lower the scores of failed strategies (they're harder than expected).
2. INSERTS: Propose 1-2 alternative strategies that avoid the same pitfalls.
   - If failure was a compile error, propose simpler approaches.
   - If failure was correctness, propose more conservative changes.
3. DELETES: Remove strategies that are clearly impossible.

Return as JSON:
{{
  "updates": [{{"node_id": "...", "score": 0.2, "reason": "..."}}],
  "inserts": [{{"parent_id": "...", "name": "...", "what": "...", "score": 0.6, "difficulty": 2}}],
  "deletes": [{{"node_id": "...", "reason": "..."}}]
}}"""

        for attempt in range(WM_MAX_RETRIES):
            response = self.engine.call_llm_sync(
                prompt, system=WM_SYSTEM, temperature=0.3)
            parsed = _parse_json(response)
            if isinstance(parsed, dict):
                return {
                    "updates": parsed.get("updates", []),
                    "inserts": parsed.get("inserts", []),
                    "deletes": parsed.get("deletes", []),
                }
            logger.warning("WM failure attempt %d: failed to parse", attempt + 1)

        return {"updates": [], "inserts": [], "deletes": []}
