from __future__ import annotations

import json
from dataclasses import dataclass, field

KERNEL_QUERY_CONTEXT = {
    "add_rmsnorm": {
        "operation": "fused add rmsnorm fp4 quantization",
        "aliases": ["rmsnorm", "residual add", "fp4 quantization", "layernorm"],
    },
    "silu_mul": {
        "operation": "fused silu mul fp4 quantization",
        "aliases": ["silu", "swiglu", "gated silu", "fp4 quantization"],
    },
    "nvfp4_quantize": {
        "operation": "nvfp4 block quantization",
        "aliases": ["fp4 quantization", "nvfp4", "bf16 to fp4", "packing"],
    },
}


@dataclass
class SandboxFeedback:
    status: str
    stage: str
    route: str
    confidence: float
    speedup: float
    parent_speedup: float
    root_cause: str
    next_action: str
    action_type: str
    evidence: list[dict] = field(default_factory=list)
    rag_queries: list[str] = field(default_factory=list)
    rag_filters: dict = field(default_factory=dict)
    preserve: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    focus: list[str] = field(default_factory=list)
    error: str = ""

    def to_payload(self) -> dict:
        payload = {
            "verdict": self.status,
            "stage": self.stage,
            "route": self.route,
            "confidence": self.confidence,
            "speedup": round(self.speedup, 6),
            "parent_speedup": round(self.parent_speedup, 6),
            "root_cause": self.root_cause,
            "evidence": self.evidence,
            "next_action": {
                "type": self.action_type,
                "instruction": self.next_action,
                "preserve": self.preserve,
                "avoid": self.avoid,
                "focus": self.focus,
            },
            "rag": {
                "provider": "pinecone",
                "queries": self.rag_queries,
                "filters": self.rag_filters,
            },
        }
        if self.error:
            payload["error"] = self.error[:600]
        return payload

    def to_tool_result_json(self) -> str:
        return json.dumps(self.to_payload(), indent=2, sort_keys=True)

    def planner_summary(self) -> str:
        summary = {
            "verdict": self.status,
            "stage": self.stage,
            "route": self.route,
            "speedup": round(self.speedup, 6),
            "parent_speedup": round(self.parent_speedup, 6),
            "root_cause": self.root_cause,
            "next_action": {
                "type": self.action_type,
                "instruction": self.next_action,
            },
            "rag": {
                "queries": self.rag_queries[:3],
                "filters": self.rag_filters,
            },
        }
        if self.evidence:
            summary["evidence"] = self.evidence[:6]
        return json.dumps(summary, indent=2, sort_keys=True)


def _first_actionable_error(error: str) -> str:
    if not error:
        return "Unknown compiler error."
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    actionable = [
        line for line in lines
        if "error" in line.lower() and ("(" in line or ":" in line)
    ]
    return actionable[0] if actionable else lines[0]


def _metric_evidence(name: str, value, kind: str = "metric", unit: str | None = None) -> dict:
    item = {"kind": kind, "name": name, "value": value}
    if unit:
        item["unit"] = unit
    return item


def _delta_evidence(name: str, before, after, unit: str | None = None) -> dict:
    item = {"kind": "delta", "name": name, "before": before, "after": after}
    if unit:
        item["unit"] = unit
    return item


def _compiler_queries(kernel_type: str, error: str) -> list[str]:
    first = _first_actionable_error(error)
    operation = _kernel_operation_phrase(kernel_type)
    queries = [
        f"Operation: {operation}. Problem: CUDA compile error. Signature: {first}. Need: production CUDA source_code.",
        f"{operation} launch signature compile error CUDA",
    ]
    if "__syncthreads" in error:
        queries.append(f"{operation} __syncthreads divergent branch fix CUDA")
    if "undefined reference" in error.lower():
        queries.append(f"{operation} launch wrapper signature linker error CUDA")
    return _unique_queries(queries)


def _kernel_operation_phrase(kernel_type: str) -> str:
    context = KERNEL_QUERY_CONTEXT.get(kernel_type, {})
    return context.get("operation", kernel_type.replace("_", " "))


def _kernel_aliases(kernel_type: str) -> list[str]:
    context = KERNEL_QUERY_CONTEXT.get(kernel_type, {})
    aliases = list(context.get("aliases", []))
    aliases.append(kernel_type.replace("_", " "))
    return [alias for alias in aliases if alias]


def _unique_queries(queries: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for query in queries:
        cleaned = " ".join(str(query).split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def _bottleneck_focus_terms(metrics: dict) -> list[str]:
    compiler = metrics.get("_compiler", {}) if metrics else {}
    focus = []

    if compiler.get("registers_per_thread", 0) > 96:
        focus.extend(["register pressure", "occupancy tuning", "launch bounds"])
    if compiler.get("spill_stores_bytes", 0) or compiler.get("spill_loads_bytes", 0):
        focus.append("spill elimination")
    if compiler.get("sass_stg_32", 0) > 1:
        focus.extend(["vectorized stores", "uint4 store packing", "store alignment"])
    if compiler.get("sass_ldg_32", 0) > 2:
        focus.extend(["vectorized loads", "uint4 load alignment", "coalesced loads"])
    if compiler.get("sass_bra", 0) > 3:
        focus.extend(["warp shuffle reduction", "branchless reduction", "predicate reduction"])
    if compiler.get("sass_fadd", 0) + compiler.get("sass_fmul", 0) > compiler.get("sass_ffma", 0):
        focus.extend(["ffma fusion", "bf16 pair math", "fmaf"])

    mem_tput = metrics.get("mem_throughput_pct", 0)
    compute_tput = metrics.get("compute_throughput_pct", 0)
    if mem_tput > compute_tput and mem_tput > 0:
        focus.extend(["memory bandwidth", "coalesced memory path"])
    elif compute_tput > 0:
        focus.extend(["instruction mix", "compute efficiency"])

    return _unique_queries(focus)


def _build_targeted_query(kernel_type: str, metrics: dict, focus_terms: list[str]) -> str:
    operation = _kernel_operation_phrase(kernel_type)
    aliases = ", ".join(_kernel_aliases(kernel_type)[:4])
    optimizations = ", ".join(focus_terms[:4]) or "instruction mix"
    return (
        f"Operation: {operation}. "
        f"Aliases: {aliases}. Optimizations: {optimizations}. "
        f"Need: production CUDA kernel source_code."
    )


def _performance_queries(kernel_type: str, metrics: dict) -> list[str]:
    operation = _kernel_operation_phrase(kernel_type)
    focus_terms = _bottleneck_focus_terms(metrics)
    queries = [_build_targeted_query(kernel_type, metrics, focus_terms)]

    if focus_terms:
        primary = " ".join(focus_terms[:2])
        queries.append(f"{operation} {primary} CUDA source code")

    compiler = metrics.get("_compiler", {}) if metrics else {}
    if compiler.get("sass_bra", 0) > 3:
        queries.append(f"{operation} warp shuffle reduction branchless CUDA")
    if compiler.get("sass_stg_32", 0) > 1:
        queries.append(f"{operation} vectorized stores uint4 store packing CUDA")
    if compiler.get("sass_ldg_32", 0) > 2:
        queries.append(f"{operation} vectorized loads uint4 alignment CUDA")
    if compiler.get("registers_per_thread", 0) > 96:
        queries.append(f"{operation} register pressure occupancy reduction CUDA")
    if compiler.get("sass_fadd", 0) + compiler.get("sass_fmul", 0) > compiler.get("sass_ffma", 0):
        queries.append(f"{operation} bf16 ffma fusion CUDA")

    if len(queries) == 1:
        queries.append(f"{operation} production CUDA optimization source code")

    return _unique_queries(queries)


def _performance_root_cause(metrics: dict) -> str:
    compiler = metrics.get("_compiler", {}) if metrics else {}
    if compiler.get("registers_per_thread", 0) > 96:
        return "Register pressure is likely reducing occupancy."
    if compiler.get("sass_stg_32", 0) > 1:
        return "Narrow global stores are dominating the write path."
    if compiler.get("sass_ldg_32", 0) > 2:
        return "Narrow global loads are limiting memory efficiency."
    if compiler.get("sass_bra", 0) > 3:
        return "Branch-heavy control flow is adding instruction overhead."
    if compiler.get("sass_fadd", 0) + compiler.get("sass_fmul", 0) > compiler.get("sass_ffma", 0):
        return "Arithmetic is not fusing cleanly into FFMA."

    mem_tput = metrics.get("mem_throughput_pct", 0)
    compute_tput = metrics.get("compute_throughput_pct", 0)
    if mem_tput > compute_tput:
        return "The kernel still looks memory-bound."
    if compute_tput > 0:
        return "The kernel still has compute-side inefficiency."
    return "No single dominant bottleneck was isolated from the sandbox metrics."


def _collect_performance_evidence(
    speedup: float,
    parent_speedup: float,
    metrics: dict,
    prev_inner_metrics: dict | None,
) -> list[dict]:
    evidence = [
        _metric_evidence("speedup", round(speedup, 6), unit="x"),
        _delta_evidence("speedup", round(parent_speedup, 6), round(speedup, 6), unit="x"),
    ]

    for key, unit in (
        ("sm_occupancy", "%"),
        ("mem_throughput_pct", "%"),
        ("compute_throughput_pct", "%"),
    ):
        value = metrics.get(key)
        if value:
            evidence.append(_metric_evidence(key, round(value, 3), unit=unit))

    compiler = metrics.get("_compiler", {}) if metrics else {}
    prev_compiler = (prev_inner_metrics or {}).get("_compiler", {})
    for key in (
        "registers_per_thread",
        "spill_stores_bytes",
        "spill_loads_bytes",
        "sass_ldg_32",
        "sass_ldg_64",
        "sass_ldg_128",
        "sass_stg_32",
        "sass_stg_64",
        "sass_stg_128",
        "sass_ffma",
        "sass_fadd",
        "sass_fmul",
        "sass_bra",
    ):
        value = compiler.get(key)
        if value is None:
            continue
        evidence.append(_metric_evidence(key, value, kind="compiler"))
        if key in prev_compiler and prev_compiler.get(key) != value:
            evidence.append(_delta_evidence(key, prev_compiler.get(key), value))

    return evidence[:14]


def build_sandbox_feedback(
    result: dict,
    parent_speedup: float,
    prev_inner_metrics: dict | None,
    kernel_type: str,
) -> SandboxFeedback:
    compile_ok = result.get("compile_ok", False)
    correct = result.get("correct", False)
    speedup = float(result.get("speedup", 0.0) or 0.0)
    metrics = result.get("metrics", {}) or {}

    common_preserve = ["launch_signature", "correctness", "working_kernel_structure"]
    common_avoid = ["full_rewrite"]

    if not compile_ok:
        error = result.get("error", "Unknown compilation error")
        root_cause = _first_actionable_error(error)
        return SandboxFeedback(
            status="compile_error",
            stage="compile",
            route="fixer_with_rag",
            confidence=0.98,
            speedup=0.0,
            parent_speedup=parent_speedup,
            root_cause=root_cause,
            next_action="Fix the compiler error and keep the launch contract unchanged.",
            action_type="repair_compile",
            evidence=[
                _metric_evidence("speedup", 0.0, unit="x"),
                {"kind": "compiler_error", "name": "first_error", "value": root_cause},
            ],
            rag_queries=_compiler_queries(kernel_type, error),
            rag_filters={"kernel_type": kernel_type, "failure_mode": "compile_error"},
            preserve=["launch_signature", "kernel_interface"],
            avoid=["signature_changes", "unrelated_optimizations"],
            focus=["compiler_error", "include_paths", "wrapper_signature"],
            error=error,
        )

    if not correct:
        error = result.get("error", "Output mismatch (atol=1e-2)")
        return SandboxFeedback(
            status="correctness_failure",
            stage="correctness",
            route="fixer_with_rag",
            confidence=0.95,
            speedup=speedup,
            parent_speedup=parent_speedup,
            root_cause=error[:240],
            next_action="Repair correctness before making any new optimization change.",
            action_type="repair_correctness",
            evidence=_collect_performance_evidence(speedup, parent_speedup, metrics, prev_inner_metrics),
            rag_queries=[
                f"{kernel_type} CUDA correctness fix",
                f"{kernel_type} kernel numerics debugging",
            ],
            rag_filters={"kernel_type": kernel_type, "failure_mode": "correctness_failure"},
            preserve=["launch_signature"],
            avoid=["new_optimizations", "algorithm_changes_before_fix"],
            focus=["numerics", "bounds", "synchronization"],
            error=error,
        )

    performance_queries = _performance_queries(kernel_type, metrics)
    root_cause = _performance_root_cause(metrics)
    evidence = _collect_performance_evidence(speedup, parent_speedup, metrics, prev_inner_metrics)

    if speedup < 1.0:
        return SandboxFeedback(
            status="below_baseline",
            stage="benchmark",
            route="fixer_with_rag",
            confidence=0.88,
            speedup=speedup,
            parent_speedup=parent_speedup,
            root_cause=root_cause,
            next_action="Revise the bottleneck assumption and make one smaller repair-oriented change.",
            action_type="revise_bottleneck",
            evidence=evidence,
            rag_queries=performance_queries,
            rag_filters={"kernel_type": kernel_type, "failure_mode": "below_baseline"},
            preserve=common_preserve,
            avoid=["full_rewrite", "stacking_multiple_changes"],
            focus=["top_compiler_evidence", "memory_path", "instruction_mix"],
        )

    relation = "improved" if speedup > parent_speedup else "held_above_baseline"
    return SandboxFeedback(
        status=relation,
        stage="benchmark",
        route="planner_tree",
        confidence=0.84,
        speedup=speedup,
        parent_speedup=parent_speedup,
        root_cause=root_cause,
        next_action="Preserve the working structure and branch into surgical follow-up hypotheses.",
        action_type="branch_tree",
        evidence=evidence,
        rag_queries=performance_queries,
        rag_filters={"kernel_type": kernel_type, "failure_mode": "above_baseline"},
        preserve=common_preserve,
        avoid=common_avoid,
        focus=["remaining_instruction_bottlenecks", "surgical_changes_only"],
    )
