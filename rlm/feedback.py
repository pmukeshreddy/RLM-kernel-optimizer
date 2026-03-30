from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

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

OBSERVATION_COMPILER_KEYS = (
    "registers_per_thread",
    "spill_stores_bytes",
    "spill_loads_bytes",
    "static_smem_bytes",
    "cmem_bytes",
    "stack_frame_bytes",
)


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
    observations: dict = field(default_factory=dict)
    hypothesis_test: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    rag_queries: list[str] = field(default_factory=list)
    rag_filters: dict = field(default_factory=dict)
    preserve: list[str] = field(default_factory=list)
    revert: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    focus: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    abort_if: list[str] = field(default_factory=list)
    error: str = ""

    def to_payload(self) -> dict:
        preserve = _unique_queries(self.preserve)
        revert = _unique_queries(self.revert)
        avoid = _unique_queries(self.avoid)
        focus = _unique_queries(self.focus)
        success_criteria = _unique_queries(self.success_criteria)
        abort_if = _unique_queries(self.abort_if)
        payload = {
            "verdict": self.status,
            "stage": self.stage,
            "route": self.route,
            "confidence": self.confidence,
            "speedup": round(self.speedup, 6),
            "parent_speedup": round(self.parent_speedup, 6),
            "root_cause": self.root_cause,
            "observations": self.observations,
            "hypothesis_test": self.hypothesis_test,
            "evidence": self.evidence,
            "next_action": {
                "type": self.action_type,
                "instruction": self.next_action,
                "preserve": preserve,
                "revert": revert,
                "avoid": avoid,
                "focus": focus,
                "success_criteria": success_criteria,
                "abort_if": abort_if,
            },
            "memory": self.memory,
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
            "route": self.route,
            "speedup": round(self.speedup, 6),
            "parent_speedup": round(self.parent_speedup, 6),
            "root_cause": self.root_cause,
            "hypothesis_test": {
                "previous_hypothesis": self.hypothesis_test.get("previous_hypothesis", ""),
                "status": self.hypothesis_test.get("status", ""),
            },
            "next_action": {
                "type": self.action_type,
                "instruction": self.next_action,
                "focus": self.focus[:4],
            },
            "memory": {
                "branch_family": self.memory.get("branch_family", ""),
                "plateau_count": self.memory.get("plateau_count", 0),
                "tried_and_failed": self.memory.get("tried_and_failed", [])[:3],
                "tried_and_helped": self.memory.get("tried_and_helped", [])[:3],
            },
            "rag": {
                "queries": self.rag_queries[:3],
                "filters": self.rag_filters,
            },
        }
        if self.observations:
            summary["observations"] = {
                key: self.observations[key]
                for key in ("timing_us", "speedup", "delta_vs_parent")
                if key in self.observations
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


def _get_candidate_attr(candidate: Any, name: str, default):
    if candidate is None:
        return default
    return getattr(candidate, name, default)


def _candidate_plan(candidate: Any) -> dict:
    plan = _get_candidate_attr(candidate, "plan_branch", {}) or {}
    return plan if isinstance(plan, dict) else {}


def _branch_family(candidate: Any) -> str:
    family = _get_candidate_attr(candidate, "branch_family", "") or ""
    if family:
        return family
    parent_strategy = _get_candidate_attr(candidate, "parent_strategy", "") or ""
    if parent_strategy:
        return parent_strategy.split("__", 1)[0]
    strategy = _get_candidate_attr(candidate, "strategy", "") or ""
    return strategy.split("__", 1)[0]


def _latest_experiment_label(candidate: Any) -> str:
    plan = _candidate_plan(candidate)
    for key in ("change_summary", "what", "goal", "bottleneck", "name"):
        value = plan.get(key)
        if value:
            return str(value)
    strategy = _get_candidate_attr(candidate, "strategy", "")
    return strategy or "latest optimization attempt"


def _candidate_memory(candidate: Any, root_cause: str) -> dict:
    history = list(_get_candidate_attr(candidate, "refinement_history", []) or [])
    failed = []
    helped = []
    plateau_count = 0

    for entry in history[-8:]:
        label = (
            entry.get("branch")
            or entry.get("strategy_desc")
            or entry.get("strategy")
            or "unnamed_change"
        )
        outcome = entry.get("outcome", "")
        if outcome == "improved":
            helped.append(label)
        elif outcome in {"compile_fail", "correctness_fail", "regression", "stagnant"}:
            failed.append(label)
        if outcome == "stagnant":
            plateau_count += 1

    return {
        "branch_family": _branch_family(candidate),
        "best_branch_family": _branch_family(candidate),
        "latest_experiment": _latest_experiment_label(candidate),
        "plateau_count": plateau_count,
        "refine_attempts": int(_get_candidate_attr(candidate, "refine_attempts", 0) or 0),
        "tried_and_failed": failed[-4:],
        "tried_and_helped": helped[-4:],
        "current_root_cause": root_cause,
    }


def _bottleneck_focus_terms(metrics: dict, root_cause: str | None = None) -> list[str]:
    compiler = metrics.get("_compiler", {}) if metrics else {}
    focus = []
    cause = (root_cause or "").lower()

    if compiler.get("spill_stores_bytes", 0) or compiler.get("spill_loads_bytes", 0) or "spill" in cause:
        focus.append("spill elimination")
    if compiler.get("registers_per_thread", 0) > 96 or "register pressure" in cause:
        focus.extend(["register pressure", "occupancy tuning", "launch bounds"])

    mem_tput = metrics.get("mem_throughput_pct", 0)
    compute_tput = metrics.get("compute_throughput_pct", 0)
    occupancy = metrics.get("sm_occupancy", 0)
    if "memory traffic" in cause or (mem_tput > compute_tput and mem_tput > 0):
        focus.extend(["pass elimination", "coalesced memory path", "vectorized access", "cache reuse"])
    elif "arithmetic throughput" in cause or compute_tput > 0:
        focus.extend(["hot-path simplification", "hardware intrinsics", "branchless arithmetic"])
    elif "latency" in cause or occupancy < 75.0:
        focus.extend(["latency hiding", "occupancy tuning", "independent work per thread"])
    else:
        focus.extend(["single hot path", "localized experiment"])

    return _unique_queries(focus)


def _build_targeted_query(kernel_type: str, focus_terms: list[str], memory: dict) -> str:
    operation = _kernel_operation_phrase(kernel_type)
    aliases = ", ".join(_kernel_aliases(kernel_type)[:4])
    optimizations = ", ".join(focus_terms[:4]) or "instruction mix"
    latest = memory.get("latest_experiment", "")
    return (
        f"Operation: {operation}. "
        f"Aliases: {aliases}. "
        f"Current experiment: {latest}. "
        f"Optimizations: {optimizations}. "
        f"Need: production CUDA kernel source_code."
    )


def _performance_queries(kernel_type: str, metrics: dict, root_cause: str, memory: dict) -> list[str]:
    focus_terms = _bottleneck_focus_terms(metrics, root_cause)
    operation = _kernel_operation_phrase(kernel_type)
    queries = [_build_targeted_query(kernel_type, focus_terms, memory)]

    latest = memory.get("latest_experiment", "")
    if latest:
        queries.append(f"{operation} {latest} CUDA source code")
    if focus_terms:
        queries.append(f"{operation} {' '.join(focus_terms[:2])} CUDA source code")

    cause = root_cause.lower()
    if "memory traffic" in cause:
        queries.append(f"{operation} eliminate extra memory pass register reuse CUDA")
        queries.append(f"{operation} vectorized access coalesced memory CUDA")
    if "register pressure" in cause:
        queries.append(f"{operation} register pressure occupancy reduction CUDA")
    if "arithmetic throughput" in cause:
        queries.append(f"{operation} hardware intrinsics hot path simplification CUDA")
    if "latency" in cause:
        queries.append(f"{operation} latency hiding occupancy ilp CUDA")

    return _unique_queries(queries)


def _performance_root_cause(metrics: dict) -> str:
    compiler = metrics.get("_compiler", {}) if metrics else {}
    spill_total = compiler.get("spill_stores_bytes", 0) + compiler.get("spill_loads_bytes", 0)
    regs = compiler.get("registers_per_thread", 0)
    occupancy = float(metrics.get("sm_occupancy", 0) or 0.0)
    mem_tput = float(metrics.get("mem_throughput_pct", 0) or 0.0)
    compute_tput = float(metrics.get("compute_throughput_pct", 0) or 0.0)

    if spill_total > 0:
        return "Hypothesis: spills and local-memory traffic may be limiting performance."
    if regs > 96 or (regs > 0 and occupancy < 75.0):
        return "Hypothesis: register pressure may be limiting occupancy."
    if mem_tput >= max(compute_tput, 25.0):
        return "Hypothesis: memory traffic is the main limiter."
    if compute_tput >= max(mem_tput, 25.0):
        return "Hypothesis: arithmetic throughput is the current limiter."
    if occupancy > 0 and occupancy < 60.0:
        return "Hypothesis: the kernel is latency-limited due to low occupancy."
    return "Hypothesis: no single dominant bottleneck is confirmed yet."


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
        ("duration_us", "us"),
        ("sm_occupancy", "%"),
        ("mem_throughput_pct", "%"),
        ("compute_throughput_pct", "%"),
    ):
        value = metrics.get(key)
        if value is not None and value != 0:
            evidence.append(_metric_evidence(key, round(float(value), 3), unit=unit))

    compiler = metrics.get("_compiler", {}) if metrics else {}
    prev_compiler = (prev_inner_metrics or {}).get("_compiler", {})
    for key in OBSERVATION_COMPILER_KEYS:
        value = compiler.get(key)
        if value is None:
            continue
        evidence.append(_metric_evidence(key, value, kind="compiler"))
        if key in prev_compiler and prev_compiler.get(key) != value:
            evidence.append(_delta_evidence(key, prev_compiler.get(key), value))

    return evidence[:16]


def _build_observations(
    compile_ok: bool,
    correct: bool,
    speedup: float,
    parent_speedup: float,
    metrics: dict,
    prev_inner_metrics: dict | None,
    error: str = "",
) -> dict:
    observations = {
        "compile_ok": bool(compile_ok),
        "correct": bool(correct),
        "speedup": round(speedup, 6),
        "parent_speedup": round(parent_speedup, 6),
    }

    duration_us = metrics.get("duration_us")
    if duration_us:
        observations["timing_us"] = round(float(duration_us), 3)
    for key in ("sm_occupancy", "mem_throughput_pct", "compute_throughput_pct"):
        value = metrics.get(key)
        if value:
            observations[key] = round(float(value), 3)

    delta_vs_parent = {}
    if parent_speedup:
        delta_vs_parent["speedup"] = round(speedup - parent_speedup, 6)

    prev_metrics = prev_inner_metrics or {}
    for key in ("duration_us", "sm_occupancy", "mem_throughput_pct", "compute_throughput_pct"):
        before = prev_metrics.get(key)
        after = metrics.get(key)
        if before is not None and after is not None and before != after:
            delta_vs_parent[key] = round(float(after) - float(before), 3)

    compiler = metrics.get("_compiler", {}) if metrics else {}
    prev_compiler = prev_metrics.get("_compiler", {}) if prev_metrics else {}
    for key in OBSERVATION_COMPILER_KEYS:
        before = prev_compiler.get(key)
        after = compiler.get(key)
        if before is not None and after is not None and before != after:
            delta_vs_parent[key] = after - before

    if delta_vs_parent:
        observations["delta_vs_parent"] = delta_vs_parent
    if error:
        observations["error"] = error[:240]
    return observations


def _root_cause_metric_support(root_cause: str, metrics: dict, prev_inner_metrics: dict | None) -> tuple[list[str], list[str]]:
    cause = root_cause.lower()
    compiler = metrics.get("_compiler", {}) if metrics else {}
    prev_compiler = (prev_inner_metrics or {}).get("_compiler", {})
    evidence_for = []
    evidence_against = []

    def add_delta(name: str, better_when_smaller: bool = True):
        before = prev_compiler.get(name)
        after = compiler.get(name)
        if before is None or after is None or before == after:
            return
        improved = after < before if better_when_smaller else after > before
        msg = f"{name} {before} -> {after}"
        if improved:
            evidence_for.append(msg)
        else:
            evidence_against.append(msg)

    before_occ = (prev_inner_metrics or {}).get("sm_occupancy")
    after_occ = metrics.get("sm_occupancy")
    before_mem = (prev_inner_metrics or {}).get("mem_throughput_pct")
    after_mem = metrics.get("mem_throughput_pct")
    before_compute = (prev_inner_metrics or {}).get("compute_throughput_pct")
    after_compute = metrics.get("compute_throughput_pct")

    if "spill" in cause:
        before_spills = prev_compiler.get("spill_stores_bytes", 0) + prev_compiler.get("spill_loads_bytes", 0)
        after_spills = compiler.get("spill_stores_bytes", 0) + compiler.get("spill_loads_bytes", 0)
        if before_spills != after_spills:
            msg = f"spill_bytes {before_spills} -> {after_spills}"
            if after_spills < before_spills:
                evidence_for.append(msg)
            else:
                evidence_against.append(msg)
    elif "register pressure" in cause:
        add_delta("registers_per_thread", better_when_smaller=True)
        if before_occ is not None and after_occ is not None and before_occ != after_occ:
            msg = f"sm_occupancy {before_occ} -> {after_occ}"
            if after_occ > before_occ:
                evidence_for.append(msg)
            else:
                evidence_against.append(msg)
    elif "memory traffic" in cause:
        if before_mem is not None and after_mem is not None and before_mem != after_mem:
            msg = f"mem_throughput_pct {before_mem} -> {after_mem}"
            if after_mem > before_mem:
                evidence_for.append(msg)
            else:
                evidence_against.append(msg)
    elif "arithmetic throughput" in cause:
        if before_compute is not None and after_compute is not None and before_compute != after_compute:
            msg = f"compute_throughput_pct {before_compute} -> {after_compute}"
            if after_compute > before_compute:
                evidence_for.append(msg)
            else:
                evidence_against.append(msg)
    elif "latency" in cause and before_occ is not None and after_occ is not None and before_occ != after_occ:
        msg = f"sm_occupancy {before_occ} -> {after_occ}"
        if after_occ > before_occ:
            evidence_for.append(msg)
        else:
            evidence_against.append(msg)

    return evidence_for, evidence_against


def _hypothesis_test(
    *,
    root_cause: str,
    speedup: float,
    parent_speedup: float,
    metrics: dict,
    prev_inner_metrics: dict | None,
    candidate: Any,
    compile_ok: bool,
    correct: bool,
    error: str = "",
) -> dict:
    previous_hypothesis = _latest_experiment_label(candidate)
    evidence_for, evidence_against = _root_cause_metric_support(root_cause, metrics, prev_inner_metrics)

    if not compile_ok:
        status = "inconclusive"
        evidence_against.append(_first_actionable_error(error))
    elif not correct:
        status = "falsified"
        evidence_against.append(error[:160] or "Correctness failed.")
    elif speedup > parent_speedup + 0.02:
        status = "confirmed" if evidence_for else "partially_confirmed"
        evidence_for.append(f"speedup improved {parent_speedup:.3f}x -> {speedup:.3f}x")
    elif speedup < max(parent_speedup - 0.001, 1.0):
        status = "falsified"
        evidence_against.append(f"speedup regressed {parent_speedup:.3f}x -> {speedup:.3f}x")
    elif evidence_for:
        status = "partially_confirmed"
        evidence_against.append("Proxy metrics moved, but runtime did not materially improve.")
    else:
        status = "inconclusive"
        evidence_against.append("Runtime held flat, so the previous hypothesis is not yet validated.")

    return {
        "previous_hypothesis": previous_hypothesis,
        "status": status,
        "evidence_for": evidence_for[:4],
        "evidence_against": evidence_against[:4],
    }


def _next_experiment_fields(
    *,
    status: str,
    root_cause: str,
    candidate: Any,
    speedup: float,
    parent_speedup: float,
) -> tuple[str, list[str], list[str], list[str], list[str], list[str], list[str]]:
    cause = root_cause.lower()
    preserve = ["launch_signature", "correctness", "working_kernel_structure"]
    revert = []
    avoid = ["full_rewrite"]
    focus = []
    success_criteria = ["Correctness must hold."]
    abort_if = []

    history = list(_get_candidate_attr(candidate, "refinement_history", []) or [])
    if status == "falsified":
        latest = history[-1] if history else {}
        label = latest.get("branch") or latest.get("strategy_desc") or latest.get("strategy")
        if label:
            revert.append(str(label))

    if "memory traffic" in cause:
        instruction = "Keep the working math path. Change only one memory path or one extra pass through global memory."
        focus = ["pass elimination", "coalesced access", "vectorized access", "cache reuse"]
        success_criteria.extend([
            "timing_us improves against the parent.",
            "register pressure and spills stay controlled.",
        ])
        abort_if = [
            "registers_per_thread increases by more than 8 with <=1% speedup gain",
            "new spills appear without a compensating runtime win",
        ]
    elif "spill" in cause:
        instruction = "Keep the algorithm unchanged. Reduce spills by trimming live state or simplifying per-thread work."
        focus = ["spill elimination", "live-range trimming", "smaller per-thread state"]
        success_criteria.extend([
            "spill bytes go down.",
            "timing_us improves against the parent."
        ])
        abort_if = [
            "register count rises while spills remain",
            "the fix requires a launch-contract change",
        ]
    elif "register pressure" in cause:
        instruction = "Keep the fastest path intact. Reduce live values or split work without changing the algorithm."
        focus = ["live-range trimming", "launch bounds", "smaller per-thread state"]
        success_criteria.extend([
            "registers_per_thread drops or occupancy rises.",
            "runtime does not regress."
        ])
        abort_if = [
            "spills appear",
            "occupancy falls without a meaningful runtime win",
        ]
    elif "arithmetic throughput" in cause:
        instruction = "Keep the current memory path. Simplify one arithmetic hot path or swap in one hardware intrinsic."
        focus = ["hot-path simplification", "hardware intrinsics", "branchless arithmetic"]
        success_criteria.extend([
            "timing_us improves against the parent.",
            "registers_per_thread and spills stay controlled.",
        ])
        abort_if = [
            "instruction changes require a full rewrite",
            "register pressure rises without a runtime gain",
        ]
    elif "latency" in cause:
        instruction = "Preserve the working kernel structure. Add one localized change that improves occupancy or hides latency."
        focus = ["latency hiding", "occupancy tuning", "independent work per thread"]
        success_criteria.extend([
            "occupancy or measured throughput improves with runtime gain.",
            "new spills do not appear."
        ])
        abort_if = [
            "occupancy drops below 75% without a compensating runtime win",
        ]
    else:
        instruction = "Revise the bottleneck assumption and make one smaller localized change."
        focus = ["single hot path", "measured bottleneck", "one change only"]
        success_criteria.append("timing_us improves against the parent.")
        abort_if = ["multiple unrelated changes are required to explain the result"]

    if speedup < 1.0:
        avoid.append("stacking_multiple_changes")
    elif speedup <= parent_speedup + 0.02:
        avoid.append("new_branch_family")
    else:
        avoid.append("breaking_working_structure")

    return instruction, preserve, revert, avoid, focus, success_criteria[:4], abort_if[:4]


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


def build_sandbox_feedback(
    result: dict,
    parent_speedup: float,
    prev_inner_metrics: dict | None,
    kernel_type: str,
    candidate: Any = None,
) -> SandboxFeedback:
    compile_ok = result.get("compile_ok", False)
    correct = result.get("correct", False)
    speedup = float(result.get("speedup", 0.0) or 0.0)
    metrics = result.get("metrics", {}) or {}
    error = result.get("error", "") or ""

    if not compile_ok:
        root_cause = _first_actionable_error(error)
        memory = _candidate_memory(candidate, root_cause)
        observations = _build_observations(
            compile_ok=False,
            correct=False,
            speedup=0.0,
            parent_speedup=parent_speedup,
            metrics=metrics,
            prev_inner_metrics=prev_inner_metrics,
            error=error,
        )
        hypothesis_test = _hypothesis_test(
            root_cause=root_cause,
            speedup=0.0,
            parent_speedup=parent_speedup,
            metrics=metrics,
            prev_inner_metrics=prev_inner_metrics,
            candidate=candidate,
            compile_ok=False,
            correct=False,
            error=error,
        )
        instruction = "Fix the compiler error and keep the launch contract unchanged."
        return SandboxFeedback(
            status="compile_error",
            stage="compile",
            route="fixer_with_rag",
            confidence=0.98,
            speedup=0.0,
            parent_speedup=parent_speedup,
            root_cause=root_cause,
            next_action=instruction,
            action_type="repair_compile",
            evidence=[
                _metric_evidence("speedup", 0.0, unit="x"),
                {"kind": "compiler_error", "name": "first_error", "value": root_cause},
            ],
            observations=observations,
            hypothesis_test=hypothesis_test,
            memory=memory,
            rag_queries=_compiler_queries(kernel_type, error),
            rag_filters={"kernel_type": kernel_type, "failure_mode": "compile_error"},
            preserve=["launch_signature", "kernel_interface"],
            revert=[],
            avoid=["signature_changes", "unrelated_optimizations"],
            focus=["compiler_error", "include_paths", "wrapper_signature"],
            success_criteria=["Compilation succeeds without changing the launch contract."],
            abort_if=["The fix requires changing the kernel interface or wrapper signature."],
            error=error,
        )

    if not correct:
        root_cause = error[:240] or "Output mismatch (atol=1e-2)"
        memory = _candidate_memory(candidate, root_cause)
        observations = _build_observations(
            compile_ok=True,
            correct=False,
            speedup=speedup,
            parent_speedup=parent_speedup,
            metrics=metrics,
            prev_inner_metrics=prev_inner_metrics,
            error=error,
        )
        hypothesis_test = _hypothesis_test(
            root_cause=root_cause,
            speedup=speedup,
            parent_speedup=parent_speedup,
            metrics=metrics,
            prev_inner_metrics=prev_inner_metrics,
            candidate=candidate,
            compile_ok=True,
            correct=False,
            error=error,
        )
        return SandboxFeedback(
            status="correctness_failure",
            stage="correctness",
            route="fixer_with_rag",
            confidence=0.95,
            speedup=speedup,
            parent_speedup=parent_speedup,
            root_cause=root_cause,
            next_action="Repair correctness before making any new optimization change.",
            action_type="repair_correctness",
            evidence=_collect_performance_evidence(speedup, parent_speedup, metrics, prev_inner_metrics),
            observations=observations,
            hypothesis_test=hypothesis_test,
            memory=memory,
            rag_queries=[
                f"{kernel_type} CUDA correctness fix",
                f"{kernel_type} kernel numerics debugging",
            ],
            rag_filters={"kernel_type": kernel_type, "failure_mode": "correctness_failure"},
            preserve=["launch_signature"],
            revert=[],
            avoid=["new_optimizations", "algorithm_changes_before_fix"],
            focus=["numerics", "bounds", "synchronization"],
            success_criteria=["Correctness passes before any new optimization is attempted."],
            abort_if=["A new performance optimization is introduced before the mismatch is fixed."],
            error=error,
        )

    root_cause = _performance_root_cause(metrics)
    memory = _candidate_memory(candidate, root_cause)
    observations = _build_observations(
        compile_ok=True,
        correct=True,
        speedup=speedup,
        parent_speedup=parent_speedup,
        metrics=metrics,
        prev_inner_metrics=prev_inner_metrics,
    )
    hypothesis_test = _hypothesis_test(
        root_cause=root_cause,
        speedup=speedup,
        parent_speedup=parent_speedup,
        metrics=metrics,
        prev_inner_metrics=prev_inner_metrics,
        candidate=candidate,
        compile_ok=True,
        correct=True,
    )
    evidence = _collect_performance_evidence(speedup, parent_speedup, metrics, prev_inner_metrics)
    queries = _performance_queries(kernel_type, metrics, root_cause, memory)
    instruction, preserve, revert, avoid, focus, success_criteria, abort_if = _next_experiment_fields(
        status=hypothesis_test["status"],
        root_cause=root_cause,
        candidate=candidate,
        speedup=speedup,
        parent_speedup=parent_speedup,
    )

    if speedup < 1.0:
        return SandboxFeedback(
            status="below_baseline",
            stage="benchmark",
            route="fixer_with_rag",
            confidence=0.88,
            speedup=speedup,
            parent_speedup=parent_speedup,
            root_cause=root_cause,
            next_action=instruction,
            action_type="revise_bottleneck",
            evidence=evidence,
            observations=observations,
            hypothesis_test=hypothesis_test,
            memory=memory,
            rag_queries=queries,
            rag_filters={"kernel_type": kernel_type, "failure_mode": "below_baseline"},
            preserve=preserve,
            revert=revert,
            avoid=avoid + ["new_branch_family"],
            focus=focus,
            success_criteria=success_criteria,
            abort_if=abort_if,
        )

    if speedup > parent_speedup + 0.02:
        return SandboxFeedback(
            status="improved",
            stage="benchmark",
            route="planner_tree",
            confidence=0.86,
            speedup=speedup,
            parent_speedup=parent_speedup,
            root_cause=root_cause,
            next_action="Preserve the working structure and branch into one surgical follow-up experiment.",
            action_type="branch_tree",
            evidence=evidence,
            observations=observations,
            hypothesis_test=hypothesis_test,
            memory=memory,
            rag_queries=queries,
            rag_filters={"kernel_type": kernel_type, "failure_mode": "improved"},
            preserve=preserve,
            revert=revert,
            avoid=avoid + ["full_rewrite"],
            focus=focus,
            success_criteria=success_criteria,
            abort_if=abort_if,
        )

    return SandboxFeedback(
        status="plateaued_above_baseline",
        stage="benchmark",
        route="fixer_with_rag",
        confidence=0.82,
        speedup=speedup,
        parent_speedup=parent_speedup,
        root_cause=root_cause,
        next_action=instruction,
        action_type="targeted_refine",
        evidence=evidence,
        observations=observations,
        hypothesis_test=hypothesis_test,
        memory=memory,
        rag_queries=queries,
        rag_filters={"kernel_type": kernel_type, "failure_mode": "plateaued_above_baseline"},
        preserve=preserve,
        revert=revert,
        avoid=avoid + ["new_branch_family"],
        focus=focus,
        success_criteria=success_criteria,
        abort_if=abort_if,
    )
