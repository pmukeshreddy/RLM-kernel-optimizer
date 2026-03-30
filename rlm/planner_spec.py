from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class PlannerSpec:
    mode: str
    kernel_type: str
    operation: str
    aliases: list[str]
    problem_shape: tuple
    branch_count: int
    kernel_src: str
    rag_context: str
    objective: str
    baseline_context: str = ""
    feedback_summary: str = ""
    parent_strategy: str = ""
    parent_speedup: float | None = None
    success_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict:
        payload = {
            "mode": self.mode,
            "kernel_type": self.kernel_type,
            "operation": self.operation,
            "aliases": list(self.aliases),
            "problem_shape": list(self.problem_shape),
            "branch_count": self.branch_count,
            "objective": self.objective,
            "success_criteria": list(self.success_criteria),
            "constraints": list(self.constraints),
        }
        if self.baseline_context:
            payload["baseline_context"] = self.baseline_context
        if self.feedback_summary:
            payload["feedback_summary"] = self.feedback_summary
        if self.parent_strategy:
            payload["parent_strategy"] = self.parent_strategy
        if self.parent_speedup is not None:
            payload["parent_speedup"] = round(float(self.parent_speedup), 6)
        return payload

    def to_prompt_json(self) -> str:
        return json.dumps(self.to_prompt_dict(), indent=2, sort_keys=True)


def build_root_planner_spec(
    *,
    kernel_type: str,
    operation: str,
    aliases: list[str],
    problem_shape: tuple,
    kernel_src: str,
    baseline_context: str,
    rag_context: str,
    branch_count: int,
) -> PlannerSpec:
    return PlannerSpec(
        mode="root",
        kernel_type=kernel_type,
        operation=operation,
        aliases=aliases,
        problem_shape=problem_shape,
        branch_count=branch_count,
        kernel_src=kernel_src,
        rag_context=rag_context,
        objective="Generate root branches that each test one distinct optimization hypothesis.",
        baseline_context=baseline_context,
        success_criteria=[
            "Each branch should make one focused adaptation only.",
            "Each branch is testable in one sandbox iteration.",
            "The set of branches is diverse, not repeated variations.",
        ],
        constraints=[
            "Use Pinecone RAG context when naming the concrete technique to test.",
            "Prefer production-style CUDA implementation patterns over vague advice.",
            "Prefer exact operation matches over kernels that only share datatype or hardware family.",
            "Do not propose full rewrites in every branch; vary the plan surface.",
        ],
    )


def build_tree_planner_spec(
    *,
    kernel_type: str,
    operation: str,
    aliases: list[str],
    problem_shape: tuple,
    kernel_src: str,
    feedback_summary: str,
    rag_context: str,
    branch_count: int,
    parent_strategy: str,
    parent_speedup: float,
) -> PlannerSpec:
    return PlannerSpec(
        mode="tree",
        kernel_type=kernel_type,
        operation=operation,
        aliases=aliases,
        problem_shape=problem_shape,
        branch_count=branch_count,
        kernel_src=kernel_src,
        rag_context=rag_context,
        objective="Expand the working parent into surgical child branches without breaking the current structure.",
        feedback_summary=feedback_summary,
        parent_strategy=parent_strategy,
        parent_speedup=parent_speedup,
        success_criteria=[
            "Each child branch should try a different minimal follow-up adaptation.",
            "Children preserve the parent's working structure.",
            "Children are small enough to validate in one sandbox turn.",
        ],
        constraints=[
            "Use the sandbox feedback and Pinecone RAG context directly.",
            "Prefer child branches that preserve the closest retrieved source pattern from the working family.",
            "Do not repeat the parent plan with different wording.",
            "Avoid full rewrites and avoid stacking multiple risky changes into one child.",
        ],
    )
