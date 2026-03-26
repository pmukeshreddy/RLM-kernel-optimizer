"""
ncu_runner.py — Run NSight Compute (ncu) and parse output.
Handles compile, profile, and metric extraction for CUDA kernels.
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

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


class NCURunner:
    """
    Manages CUDA kernel compilation and NCU profiling.
    Workflow: write .cu → nvcc compile → ncu profile → parse CSV → KernelMetrics
    """

    def __init__(self, config: dict):
        prof_cfg = config.get("profiler", {})
        self.ncu_path     = prof_cfg.get("ncu_path", "ncu")
        self.warmup       = prof_cfg.get("warmup_iters", 3)
        self.profile_iter = prof_cfg.get("profile_iters", 10)
        self.output_dir   = Path(config.get("output", {}).get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.nvcc      = "nvcc"
        self.cuda_arch = "sm_100a"
        self.nvcc_flags = [
            "-O3", f"-arch={self.cuda_arch}", "--use_fast_math", "-std=c++17",
            f"-I{PROJECT_ROOT / 'kernels' / 'common'}",
        ]

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
    ) -> Optional[KernelMetrics]:
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
            "--csv",
            "--output", str(report_path),
            "--force-overwrite",
            str(binary_path),
        ]

        logger.info("Profiling: %s ...", " ".join(ncu_cmd[:5]))
        result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=300)

        if result.returncode not in (0, 1):
            logger.error("NCU failed (rc=%d): %s", result.returncode, result.stderr[:300])
            return None

        if report_path.exists():
            logger.debug("NCU report saved to %s", report_path)
            return self._export_and_parse(report_path)
        if result.stdout and "Metric Name" in result.stdout:
            return self._parse_ncu_csv(result.stdout)
        # Log why we got nothing
        logger.warning("NCU produced no report file and no CSV stdout. rc=%d stderr=%s stdout=%s",
                       result.returncode, result.stderr[:200], result.stdout[:200])
        return None

    def _export_and_parse(self, report_path: Path) -> Optional[KernelMetrics]:
        export_cmd = [self.ncu_path, "--import", str(report_path), "--csv", "--page", "raw"]
        result = subprocess.run(export_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.warning("NCU export failed (rc=%d): %s", result.returncode, result.stderr[:300])
            return None
        logger.debug("NCU CSV export: %d bytes", len(result.stdout))
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
            logger.warning("No recognizable metrics in NCU CSV. Columns: %s, rows: %d",
                          list(reader.fieldnames) if hasattr(reader, 'fieldnames') else 'unknown',
                          sum(1 for _ in csv.DictReader(lines)))
            return None
        logger.debug("Parsed %d NCU metrics: %s", len(raw), list(raw.keys()))
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
    ) -> tuple:
        """Returns (compile_ok, metrics_or_None, speedup)."""
        ok, err, binary = self.compile_kernel(kernel_src, harness_src, name)
        if not ok:
            return False, None, 0.0

        timing_us = self.benchmark_timing(binary)
        if timing_us is None:
            return True, None, 0.0

        speedup = baseline_us / timing_us if timing_us > 0 else 0.0
        metrics = self.profile(binary, report_name=name)
        if metrics:
            metrics.duration_us = timing_us
            metrics.speedup = speedup

        return True, metrics, speedup
