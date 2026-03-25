"""
environment.py — RLM environment state management.
Loads kernel source, hardware spec, manages optimization state across rounds.
"""

from __future__ import annotations
import re
import yaml
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class KernelCandidate:
    """Represents a single kernel variant at any point in the search."""
    code: str
    strategy: str
    round_num: int = 0
    metrics: dict = field(default_factory=dict)
    speedup: float = 1.0
    bottleneck: str = "unknown"
    compile_ok: bool = False
    correct: bool = False
    ncu_report_path: Optional[str] = None

    def is_viable(self) -> bool:
        return self.compile_ok and self.correct

    def summary(self) -> str:
        return (
            f"[{self.strategy}] round={self.round_num} "
            f"speedup={self.speedup:.3f}x "
            f"bottleneck={self.bottleneck} "
            f"compile={'ok' if self.compile_ok else 'FAIL'} "
            f"correct={'yes' if self.correct else 'NO'}"
        )


@dataclass
class OptimizationHistory:
    """Tracks everything tried across all rounds and strategies."""
    entries: list = field(default_factory=list)

    def record(self, candidate: KernelCandidate, notes: str = "") -> None:
        self.entries.append({
            "timestamp": time.time(),
            "strategy": candidate.strategy,
            "round": candidate.round_num,
            "speedup": candidate.speedup,
            "bottleneck": candidate.bottleneck,
            "compile_ok": candidate.compile_ok,
            "correct": candidate.correct,
            "notes": notes,
        })

    def best_speedup(self) -> float:
        viable = [e["speedup"] for e in self.entries
                  if e["compile_ok"] and e["correct"]]
        return max(viable) if viable else 1.0

    def strategies_tried(self) -> list:
        return list({e["strategy"] for e in self.entries})

    def to_summary_str(self) -> str:
        lines = []
        for e in self.entries:
            lines.append(
                f"  round={e['round']} strategy={e['strategy']} "
                f"speedup={e['speedup']:.3f}x "
                f"ok={e['compile_ok']} correct={e['correct']}"
            )
        return "\n".join(lines) if lines else "  (none)"


class RLMEnvironment:
    """
    Shared state object for the RLM REPL.
    Root LLM reads/writes this; sub-LLMs get slices of it.
    """

    def __init__(
        self,
        kernel_name: str,
        kernel_src_path: str,
        config_path: str = None,
        kernel_type: str = "add_rmsnorm",
        problem_shape: tuple = None,
    ):
        self.kernel_name = kernel_name
        self.kernel_src_path = Path(kernel_src_path)
        self.kernel_src_raw: str = self.kernel_src_path.read_text()
        self.kernel_src: str = self._expand_local_includes(self.kernel_src_raw)
        self.kernel_type: str = kernel_type

        hw_spec_path = PROJECT_ROOT / "config" / "b200_spec.yaml"
        with open(hw_spec_path) as f:
            self.hw_spec: dict = yaml.safe_load(f)

        search_cfg_path = config_path or PROJECT_ROOT / "config" / "search_config.yaml"
        with open(search_cfg_path) as f:
            self.search_config: dict = yaml.safe_load(f)

        self.ncu_report: Optional[dict] = None
        self.baseline_us: Optional[float] = None
        self.baseline_us_reported: float = 12.4

        # Task-specific shape takes priority over config shapes
        if problem_shape is not None:
            self.problem_shapes: list = [problem_shape]
        else:
            self.problem_shapes: list = [
                tuple(s) for s in self.search_config["eval"]["problem_shapes"]
            ]

        self.optimization_history = OptimizationHistory()
        self.current_round: int = 0
        self.total_api_cost_usd: float = 0.0
        self.candidates: list = []
        self.hack_rejections: list = []

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _expand_local_includes(self, src: str) -> str:
        """Expand local #include directives so the LLM sees helper function signatures."""
        lines = src.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#include "') and stripped.endswith('"'):
                rel_path = stripped[len('#include "'):-1]
                header = self.kernel_src_path.parent / rel_path
                if header.exists():
                    result.append(f"// === expanded from {rel_path} ===")
                    result.append(header.read_text())
                    result.append(f"// === end {rel_path} ===")
                    continue
            result.append(line)
        return "\n".join(result)

    # ── Kernel source navigation ──────────────────────────────────────────────

    def find_kernel_function(self, pattern: str = r"__global__\s+void\s+(\w+)") -> list:
        return re.findall(pattern, self.kernel_src)

    def find_hot_loop(self) -> tuple:
        lines = self.kernel_src.split("\n")
        hot_start = 0
        depth_max = 0
        depth = 0
        for i, line in enumerate(lines):
            depth += line.count("{") - line.count("}")
            if "for" in line and ("[" in line or "load" in line.lower()):
                if depth > depth_max:
                    depth_max = depth
                    hot_start = i
        hot_end = min(hot_start + 30, len(lines))
        return hot_start, hot_end

    def get_hot_loop_src(self) -> str:
        s, e = self.find_hot_loop()
        lines = self.kernel_src.split("\n")
        return "\n".join(lines[s:e])

    def get_kernel_slice(self, start: int, end: int) -> str:
        return self.kernel_src[start:end]

    def count_memory_ops(self) -> dict:
        src = self.kernel_src
        return {
            "loads":       len(re.findall(r"\b(?:__ldg|ld\.global|tex1Dfetch)\b", src)),
            "stores":      len(re.findall(r"\b(?:__stg|st\.global|atomicAdd)\b", src)),
            "float4":      src.count("float4"),
            "cp_async":    src.count("cp.async"),
            "tma":         src.count("tma_load") + src.count("tcgen05"),
            "syncthreads": src.count("__syncthreads"),
            "shfl":        src.count("__shfl"),
        }

    def detect_missing_optimizations(self) -> list:
        ops = self.count_memory_ops()
        enabled = self.search_config["strategies"]["enabled"]
        missing = []
        if ops["float4"] == 0 and "vectorize_loads" in enabled:
            missing.append("vectorize_loads")
        if ops["tma"] == 0 and "tma_prefetch" in enabled:
            missing.append("tma_prefetch")
        if ops["syncthreads"] > 2 and ops["shfl"] == 0 and "warp_reduction" in enabled:
            missing.append("warp_reduction")
        if "fuse_passes" in enabled and self.kernel_src.count("__global__") > 1:
            missing.append("fuse_passes")
        return missing

    # ── Cost tracking ─────────────────────────────────────────────────────────

    def record_api_cost(self, tokens_in: int, tokens_out: int, model: str) -> float:
        pricing = {
            "claude-sonnet-4-6":         {"in": 3.0,  "out": 15.0},
            "claude-opus-4-6":           {"in": 15.0, "out": 75.0},
            "claude-haiku-4-5-20251001": {"in": 0.25, "out": 1.25},
        }
        p = pricing.get(model, {"in": 3.0, "out": 15.0})
        cost = (tokens_in * p["in"] + tokens_out * p["out"]) / 1_000_000
        self.total_api_cost_usd += cost
        return cost

    def budget_remaining(self) -> float:
        return self.search_config["cost_control"]["max_total_api_cost_usd"] - self.total_api_cost_usd

    def over_budget(self) -> bool:
        return self.total_api_cost_usd >= self.search_config["cost_control"]["max_total_api_cost_usd"]

    # ── State summary ─────────────────────────────────────────────────────────

    def state_summary(self) -> str:
        ops = self.count_memory_ops()
        missing = self.detect_missing_optimizations()
        return (
            f"=== RLM Environment State ===\n"
            f"Kernel:        {self.kernel_name}\n"
            f"Source:        {self.kernel_src_path} ({len(self.kernel_src)} chars)\n"
            f"Round:         {self.current_round}\n"
            f"Candidates:    {len(self.candidates)}\n"
            f"Best speedup:  {self.optimization_history.best_speedup():.3f}x\n"
            f"API cost:      ${self.total_api_cost_usd:.4f} / "
            f"${self.search_config['cost_control']['max_total_api_cost_usd']:.2f}\n"
            f"Memory ops:    loads={ops['loads']} stores={ops['stores']} "
            f"float4={ops['float4']} tma={ops['tma']}\n"
            f"Sync ops:      syncthreads={ops['syncthreads']} shfl={ops['shfl']}\n"
            f"Missing opts:  {', '.join(missing) or 'none detected'}\n"
            f"Strategies tried: {', '.join(self.optimization_history.strategies_tried()) or 'none'}\n"
            f"History:\n{self.optimization_history.to_summary_str()}\n"
        )
