from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class PlanBranch:
    name: str
    goal: str
    change_summary: str
    expected_signal: str
    rag_queries: list[str] = field(default_factory=list)
    planner_notes: str = ""
    parent_strategy: str = ""
    tree_ready: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "goal": self.goal,
            "what": self.change_summary,
            "change_summary": self.change_summary,
            "expected_signal": self.expected_signal,
            "rag_queries": list(self.rag_queries),
            "planner_notes": self.planner_notes,
            "parent_strategy": self.parent_strategy,
            "tree_ready": self.tree_ready,
        }


def _coerce_string_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def fallback_branches(
    count: int,
    prefix: str,
    parent_strategy: str = "",
) -> list[dict]:
    branches = []
    for idx in range(count):
        branches.append(
            PlanBranch(
                name=f"{prefix}_{idx + 1}",
                goal="Make one measurable CUDA optimization change.",
                change_summary="Implement one targeted optimization and preserve correctness.",
                expected_signal="Compiler succeeds and sandbox metrics improve.",
                rag_queries=[],
                planner_notes="Fallback plan because the planner output could not be parsed.",
                parent_strategy=parent_strategy,
                tree_ready=bool(parent_strategy),
            ).to_dict()
        )
    return branches


def parse_plan_response(
    text: str,
    count: int,
    prefix: str,
    parent_strategy: str = "",
) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return fallback_branches(count, prefix, parent_strategy=parent_strategy)

    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return fallback_branches(count, prefix, parent_strategy=parent_strategy)

    branches = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or f"{prefix}_{idx + 1}").strip()
        change_summary = str(
            item.get("change_summary")
            or item.get("what")
            or item.get("goal")
            or "Implement one targeted optimization."
        ).strip()
        branches.append(
            PlanBranch(
                name=name,
                goal=str(item.get("goal") or change_summary).strip(),
                change_summary=change_summary,
                expected_signal=str(
                    item.get("expected_signal")
                    or "Sandbox metrics should show a clear change."
                ).strip(),
                rag_queries=_coerce_string_list(item.get("rag_queries")),
                planner_notes=str(
                    item.get("planner_notes") or item.get("notes") or ""
                ).strip(),
                parent_strategy=str(
                    item.get("parent_strategy") or parent_strategy
                ).strip(),
                tree_ready=bool(item.get("tree_ready", bool(parent_strategy))),
            ).to_dict()
        )

    if not branches:
        return fallback_branches(count, prefix, parent_strategy=parent_strategy)

    return branches[:count]


def build_initial_plan_prompt(
    kernel_type: str,
    problem_shape: tuple,
    kernel_src: str,
    baseline_context: str,
    rag_context: str,
    branch_count: int,
) -> str:
    return f"""\
You are the planner agent for a CUDA kernel optimizer.
Do not write CUDA code. Produce the execution plan only.

Task:
- Kernel type: {kernel_type}
- Fixed shape: {problem_shape}
- Produce exactly {branch_count} root branches.

Baseline and profiler context:
{baseline_context}

Pinecone RAG context:
{rag_context or "No Pinecone context returned."}

Reference kernel:
```cuda
{kernel_src}
```

Return ONLY a JSON array with exactly {branch_count} objects:
[
  {{
    "name": "short_branch_name",
    "goal": "what this branch is trying to prove",
    "change_summary": "the concrete change for the coder agent",
    "expected_signal": "which sandbox result would validate the branch",
    "rag_queries": ["query 1", "query 2"],
    "planner_notes": "short warning or constraint",
    "tree_ready": false
  }}
]

Rules:
- Be concrete and branch-diverse.
- Prefer changes the sandbox can validate in one iteration.
- No prose outside the JSON array.
"""


def build_tree_plan_prompt(
    kernel_type: str,
    problem_shape: tuple,
    parent_strategy: str,
    parent_speedup: float,
    kernel_src: str,
    feedback_summary: str,
    rag_context: str,
    branch_count: int,
) -> str:
    return f"""\
You are the planner agent expanding a successful CUDA branch into a small search tree.
Do not write CUDA code. Produce child branches only.

Task:
- Kernel type: {kernel_type}
- Fixed shape: {problem_shape}
- Parent branch: {parent_strategy}
- Current sandbox speedup: {parent_speedup:.3f}x
- Produce exactly {branch_count} child branches.

Current sandbox assessment JSON:
{feedback_summary}

Pinecone RAG context:
{rag_context or "No Pinecone context returned."}

Current best kernel:
```cuda
{kernel_src}
```

Return ONLY a JSON array with exactly {branch_count} objects:
[
  {{
    "name": "child_branch_name",
    "goal": "the next hypothesis to test",
    "change_summary": "a surgical change that keeps the working structure",
    "expected_signal": "which profiler or speedup change should happen",
    "rag_queries": ["query 1", "query 2"],
    "planner_notes": "what to preserve while editing",
    "parent_strategy": "{parent_strategy}",
    "tree_ready": true
  }}
]

Rules:
- Do not propose full rewrites.
- Each child branch must attack a different remaining bottleneck.
- Keep the parent's working structure intact.
- No prose outside the JSON array.
"""
