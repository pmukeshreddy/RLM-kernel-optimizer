"""
ncu_runner.py — Run NSight Compute (ncu) and parse output.
Handles compile, profile, and metric extraction for CUDA kernels.
Falls back to HybridProfiler when NCU is unavailable (permissions, missing importer).
"""

from __future__ import annotations
import csv
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .metrics import KernelMetrics, NCU_METRICS_QUERY, parse_ncu_csv_line, metrics_from_dict
from .hybrid_profiler import HybridProfiler

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


class NCURunner:
    """
    Manages CUDA kernel compilation and NCU profiling.
    Workflow: write .cu → nvcc compile → ncu profile → parse CSV → KernelMetrics
    Falls back to hybrid profiling (analytical + occupancy API) when NCU fails.
    """

    def __init__(self, config: dict, hw_spec: dict = None):
        prof_cfg = config.get("profiler", {})
        self.ncu_path     = prof_cfg.get("ncu_path", "ncu")
        self.warmup       = prof_cfg.get("warmup_iters", 3)
        self.profile_iter = prof_cfg.get("profile_iters", 10)
        self.output_dir   = Path(config.get("output", {}).get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Profiler mode: "ncu" (ncu only), "hybrid" (skip ncu), "auto" (try ncu, fall back)
        self.profiler_mode = prof_cfg.get("tool", "auto")

        self.nvcc      = "nvcc"
        self.cuda_arch = "sm_100a"
        self.nvcc_flags = [
            "-O3", f"-arch={self.cuda_arch}", "--use_fast_math", "-std=c++17",
            f"-I{PROJECT_ROOT / 'kernels' / 'common'}",
        ]

        # Initialize hybrid profiler as fallback
        if hw_spec is None:
            import yaml
            hw_spec_path = PROJECT_ROOT / "config" / "b200_spec.yaml"
            if hw_spec_path.exists():
                with open(hw_spec_path) as f:
                    hw_spec = yaml.safe_load(f)
            else:
                hw_spec = {}
        self.hybrid = HybridProfiler(config, hw_spec)
        self._ncu_failed_permanently = False  # sticky flag after permission error

    # ── Compilation ───────────────────────────────────────────────────────────

    def compile_kernel(
        self,
        kernel_src: str,
        harness_src: str,
        output_name: str = "kernel_bench",
    ) -> tuple:
        """Compile kernel + harness. Returns (success, error_msg, binary_path)."""
        build_dir = self.output_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        kernel_file = build_dir / f"{output_name}.cu"
        binary_file = build_dir / output_name
        kernel_file.write_text(kernel_src + "\n\n" + harness_src)

        cmd = [self.nvcc] + self.nvcc_flags + [str(kernel_file), "-o", str(binary_file)]
        logger.info("Compiling: %s", " ".join(cmd[:4]) + " ...")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning("Compilation failed:\n%s", result.stderr[:400])
            return False, result.stderr, binary_file

        return True, "", binary_file

    # ── Profiling ─────────────────────────────────────────────────────────────

    def profile(
        self,
        binary_path: Path,
        report_name: str = "profile",
        kernel_regex: str = ".*",
        kernel_src: str = "",
        kernel_type: str = "",
        problem_shape: tuple = (),
        baseline_us: float = 0.0,
        timing_us: float = 0.0,
    ) -> Optional[KernelMetrics]:
        """
        Profile a compiled CUDA binary.
        Tries NCU first (unless mode is "hybrid" or NCU previously failed permanently).
        Falls back to hybrid profiler automatically.
        """
        # Skip NCU entirely if mode is hybrid or NCU has permanently failed
        if self.profiler_mode == "hybrid" or self._ncu_failed_permanently:
            return self._hybrid_fallback(
                kernel_src, timing_us, kernel_type, problem_shape, baseline_us
            )

        # Try NCU
        metrics = self._run_ncu(binary_path, report_name, kernel_regex)

        # If NCU failed, fall back to hybrid
        if metrics is None and kernel_src and kernel_type and problem_shape:
            logger.info("NCU returned no metrics — falling back to hybrid profiler")
            return self._hybrid_fallback(
                kernel_src, timing_us, kernel_type, problem_shape, baseline_us
            )

        return metrics

    def _run_ncu(
        self,
        binary_path: Path,
        report_name: str = "profile",
        kernel_regex: str = ".*",
    ) -> Optional[KernelMetrics]:
        """Run NCU profiling. Returns None on failure."""
        report_dir = self.output_dir / "ncu_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{report_name}.ncu-rep"

        ncu_cmd = [
            self.ncu_path,
            "--target-processes", "all",
            "--kernel-name", kernel_regex,
            "--launch-skip", str(self.warmup),
            "--launch-count", str(self.profile_iter),
            "--metrics", NCU_METRICS_QUERY,
            "-o", str(report_path),
            "-f",
            str(binary_path),
        ]

        logger.info("NCU cmd: %s", " ".join(ncu_cmd))
        try:
            result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=300)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("NCU execution failed: %s", e)
            return None

        logger.info("NCU finished: rc=%d report_exists=%s stdout=%d bytes stderr=%d bytes",
                    result.returncode, report_path.exists(),
                    len(result.stdout), len(result.stderr))

        # Detect permanent failures (permissions, missing importer)
        stderr_lower = result.stderr.lower()
        if "err_nvgpuctrperm" in stderr_lower or "permission" in stderr_lower:
            logger.warning("NCU permission error detected — switching to hybrid profiler permanently")
            self._ncu_failed_permanently = True
            return None
        if "importer" in stderr_lower and "not found" in stderr_lower:
            logger.warning("NCU importer missing — switching to hybrid profiler permanently")
            self._ncu_failed_permanently = True
            return None

        if result.returncode not in (0, 1):
            logger.error("NCU failed (rc=%d): %s", result.returncode, result.stderr[:300])
            return None

        if report_path.exists():
            logger.info("NCU report file: %s (%.1f KB)", report_path, report_path.stat().st_size / 1024)
            return self._export_and_parse(report_path)
        if result.stdout and "Metric Name" in result.stdout:
            return self._parse_ncu_csv(result.stdout)
        logger.warning("NCU: no report file and no CSV in stdout.\n  stdout: %s\n  stderr: %s",
                       result.stdout[:500], result.stderr[:300])
        return None

    def _hybrid_fallback(
        self,
        kernel_src: str,
        timing_us: float,
        kernel_type: str,
        problem_shape: tuple,
        baseline_us: float,
    ) -> Optional[KernelMetrics]:
        """Use hybrid profiler as fallback."""
        if not kernel_src or not kernel_type or not problem_shape or timing_us <= 0:
            logger.warning("Hybrid fallback: insufficient info (type=%s shape=%s timing=%.1f)",
                          kernel_type, problem_shape, timing_us)
            return None
        return self.hybrid.profile(
            kernel_src=kernel_src,
            timing_us=timing_us,
            kernel_type=kernel_type,
            problem_shape=problem_shape,
            baseline_us=baseline_us,
        )

    def _export_and_parse(self, report_path: Path) -> Optional[KernelMetrics]:
        export_cmd = [self.ncu_path, "--import", str(report_path), "--csv", "--page", "raw"]
        logger.info("NCU export cmd: %s", " ".join(export_cmd[:4]))
        result = subprocess.run(export_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error("NCU export failed (rc=%d): %s", result.returncode, result.stderr[:300])
            return None
        logger.info("NCU CSV export: %d bytes, first 200 chars: %s",
                    len(result.stdout), result.stdout[:200])
        return self._parse_ncu_csv(result.stdout)

    def _parse_ncu_csv(self, csv_text: str) -> Optional[KernelMetrics]:
        from .metrics import NCU_METRIC_IDS
        id_to_field = {v: k for k, v in NCU_METRIC_IDS.items()}
        raw = {}
        # NCU CSV output has comment lines starting with "==" — strip them before parsing
        lines = [l for l in csv_text.splitlines() if not l.startswith("==")]
        reader = csv.DictReader(lines)
        for row in reader:
            metric_id = row.get("Metric Name", "").strip()
            value_str = row.get("Metric Value", "0").strip()
            if metric_id in id_to_field:
                raw[id_to_field[metric_id]] = parse_ncu_csv_line(metric_id, value_str)
        if not raw:
            # Log what metric names we actually got vs what we expected
            seen_metrics = set()
            for row in csv.DictReader(lines):
                seen_metrics.add(row.get("Metric Name", "").strip())
            logger.warning("No recognizable metrics. Got %d unique metric names. Sample: %s",
                          len(seen_metrics), list(seen_metrics)[:5])
            logger.warning("Expected metrics (sample): %s", list(NCU_METRIC_IDS.values())[:3])
            return None
        logger.info("Parsed %d NCU metrics: %s", len(raw), list(raw.keys()))
        return metrics_from_dict(raw)

    # ── Timing ────────────────────────────────────────────────────────────────

    def benchmark_timing(
        self, binary_path: Path, warmup: int = 500, iters: int = 100
    ) -> Optional[float]:
        """Run binary, parse 'timing_us: <float>' from stdout. Returns microseconds."""
        if not binary_path.exists():
            return None
        result = subprocess.run(
            [str(binary_path), f"--warmup={warmup}", f"--iters={iters}"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error("Benchmark failed: %s", result.stderr[:200])
            return None

        match = re.search(r"timing_us:\s*([\d.]+)", result.stdout)
        if match:
            return float(match.group(1))
        matches = re.findall(r"(\d+\.\d+)\s*us", result.stdout)
        return float(matches[-1]) if matches else None

    # ── Convenience: compile + profile in one step ────────────────────────────

    def compile_and_profile(
        self,
        kernel_src: str,
        harness_src: str,
        name: str,
        baseline_us: float,
        kernel_type: str = "",
        problem_shape: tuple = (),
    ) -> tuple:
        """Returns (compile_ok, metrics_or_None, speedup)."""
        ok, err, binary = self.compile_kernel(kernel_src, harness_src, name)
        if not ok:
            return False, None, 0.0

        timing_us = self.benchmark_timing(binary)
        if timing_us is None:
            return True, None, 0.0

        speedup = baseline_us / timing_us if timing_us > 0 else 0.0
        metrics = self.profile(
            binary, report_name=name,
            kernel_src=kernel_src, kernel_type=kernel_type,
            problem_shape=problem_shape, baseline_us=baseline_us,
            timing_us=timing_us,
        )
        if metrics:
            metrics.duration_us = timing_us
            metrics.speedup = speedup

        return True, metrics, speedup
