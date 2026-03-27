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

from .metrics import KernelMetrics, CompilerMetrics, NCU_METRICS_QUERY, parse_ncu_csv_line, metrics_from_dict
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
        """Compile kernel + harness. Returns (success, error_msg, binary_path).
        Also extracts compiler metrics (registers, spills) via -Xptxas,-v."""
        build_dir = self.output_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        kernel_file = build_dir / f"{output_name}.cu"
        binary_file = build_dir / output_name
        kernel_file.write_text(kernel_src + "\n\n" + harness_src)

        # Add -Xptxas,-v to get register/spill info from compiler
        cmd = [self.nvcc] + self.nvcc_flags + ["-Xptxas", "-v", str(kernel_file), "-o", str(binary_file)]
        logger.info("Compiling: %s", " ".join(cmd[:4]) + " ...")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            # Filter ptxas info lines so actual errors are visible in logs
            error_lines = [l for l in result.stderr.splitlines()
                           if not l.strip().startswith("ptxas info")]
            error_msg = "\n".join(error_lines).strip() or result.stderr.strip()
            logger.warning("Compilation failed:\n%s", error_msg[:800])
            return False, error_msg, binary_file, None

        # Parse compiler metrics from ptxas verbose output
        compiler_metrics = self._parse_ptxas_verbose(result.stderr)

        # Run SASS disassembly for instruction mix
        sass_metrics = self._parse_sass_disassembly(binary_file)
        if sass_metrics:
            compiler_metrics.sass_total_instructions = sass_metrics.sass_total_instructions
            compiler_metrics.sass_ldg_32 = sass_metrics.sass_ldg_32
            compiler_metrics.sass_ldg_64 = sass_metrics.sass_ldg_64
            compiler_metrics.sass_ldg_128 = sass_metrics.sass_ldg_128
            compiler_metrics.sass_stg_32 = sass_metrics.sass_stg_32
            compiler_metrics.sass_stg_64 = sass_metrics.sass_stg_64
            compiler_metrics.sass_stg_128 = sass_metrics.sass_stg_128
            compiler_metrics.sass_lds = sass_metrics.sass_lds
            compiler_metrics.sass_sts = sass_metrics.sass_sts
            compiler_metrics.sass_ldl = sass_metrics.sass_ldl
            compiler_metrics.sass_stl = sass_metrics.sass_stl
            compiler_metrics.sass_ffma = sass_metrics.sass_ffma
            compiler_metrics.sass_hfma2 = sass_metrics.sass_hfma2
            compiler_metrics.sass_mufu = sass_metrics.sass_mufu
            compiler_metrics.sass_fadd = sass_metrics.sass_fadd
            compiler_metrics.sass_fmul = sass_metrics.sass_fmul
            compiler_metrics.sass_bar = sass_metrics.sass_bar
            compiler_metrics.sass_shfl = sass_metrics.sass_shfl
            compiler_metrics.sass_bra = sass_metrics.sass_bra

        logger.info("Compiler metrics: %s", compiler_metrics.summary_str())

        return True, "", binary_file, compiler_metrics

    def _parse_ptxas_verbose(self, stderr: str) -> CompilerMetrics:
        """Parse nvcc -Xptxas,-v output for register count, spills, shared memory."""
        cm = CompilerMetrics()

        # Match: Used N registers, M bytes smem, K bytes cmem[0]
        reg_match = re.search(r'Used\s+(\d+)\s+registers', stderr)
        if reg_match:
            cm.registers_per_thread = int(reg_match.group(1))

        smem_match = re.search(r'(\d+)\s+bytes\s+smem', stderr)
        if smem_match:
            cm.static_smem_bytes = int(smem_match.group(1))

        cmem_match = re.search(r'(\d+)\s+bytes\s+cmem\[0\]', stderr)
        if cmem_match:
            cm.cmem_bytes = int(cmem_match.group(1))

        # Match: N bytes stack frame, M bytes spill stores, K bytes spill loads
        stack_match = re.search(r'(\d+)\s+bytes\s+stack\s+frame', stderr)
        if stack_match:
            cm.stack_frame_bytes = int(stack_match.group(1))

        spill_st = re.search(r'(\d+)\s+bytes\s+spill\s+stores', stderr)
        if spill_st:
            cm.spill_stores_bytes = int(spill_st.group(1))

        spill_ld = re.search(r'(\d+)\s+bytes\s+spill\s+loads', stderr)
        if spill_ld:
            cm.spill_loads_bytes = int(spill_ld.group(1))

        return cm

    def _parse_sass_disassembly(self, binary_path: Path) -> Optional[CompilerMetrics]:
        """Disassemble binary with cuobjdump -sass and count instruction types."""
        try:
            result = subprocess.run(
                ["cuobjdump", "-sass", str(binary_path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.debug("cuobjdump failed: %s", result.stderr[:200])
                return None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("cuobjdump not available or timed out")
            return None

        sass = result.stdout
        cm = CompilerMetrics()

        # Count instruction opcodes from SASS output
        # SASS lines look like: /*0080*/  LDG.E.128.SYS R4, [R2] ;
        instructions = re.findall(r'/\*[0-9a-f]+\*/\s+([A-Z][A-Z0-9_.]+)', sass)
        cm.sass_total_instructions = len(instructions)

        for inst in instructions:
            # Global loads by width
            if inst.startswith("LDG"):
                if ".128" in inst:
                    cm.sass_ldg_128 += 1
                elif ".64" in inst:
                    cm.sass_ldg_64 += 1
                else:
                    cm.sass_ldg_32 += 1
            elif inst.startswith("STG"):
                if ".128" in inst:
                    cm.sass_stg_128 += 1
                elif ".64" in inst:
                    cm.sass_stg_64 += 1
                else:
                    cm.sass_stg_32 += 1
            elif inst.startswith("LDS"):
                cm.sass_lds += 1
            elif inst.startswith("STS"):
                cm.sass_sts += 1
            elif inst.startswith("LDL"):
                cm.sass_ldl += 1
            elif inst.startswith("STL"):
                cm.sass_stl += 1
            elif inst.startswith("FFMA"):
                cm.sass_ffma += 1
            elif inst.startswith("HFMA2"):
                cm.sass_hfma2 += 1
            elif inst.startswith("MUFU"):
                cm.sass_mufu += 1
            elif inst.startswith("FADD"):
                cm.sass_fadd += 1
            elif inst.startswith("FMUL"):
                cm.sass_fmul += 1
            elif inst == "BAR" or inst.startswith("BAR."):
                cm.sass_bar += 1
            elif inst.startswith("SHFL"):
                cm.sass_shfl += 1
            elif inst.startswith("BRA"):
                cm.sass_bra += 1

        return cm if cm.sass_total_instructions > 0 else None

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
        compiler_metrics: 'CompilerMetrics' = None,
    ) -> Optional[KernelMetrics]:
        """
        Profile a compiled CUDA binary.
        Tries NCU first (unless mode is "hybrid" or NCU previously failed permanently).
        Falls back to hybrid profiler automatically.
        """
        # Skip NCU entirely if mode is hybrid or NCU has permanently failed
        if self.profiler_mode == "hybrid" or self._ncu_failed_permanently:
            return self._hybrid_fallback(
                kernel_src, timing_us, kernel_type, problem_shape, baseline_us,
                compiler_metrics,
            )

        # Try NCU
        metrics = self._run_ncu(binary_path, report_name, kernel_regex)

        # If NCU failed, fall back to hybrid
        if metrics is None and kernel_src and kernel_type and problem_shape:
            logger.info("NCU returned no metrics — falling back to hybrid profiler")
            return self._hybrid_fallback(
                kernel_src, timing_us, kernel_type, problem_shape, baseline_us,
                compiler_metrics,
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

        # Detect permanent failures (permissions, missing importer, version too old)
        combined_output = (result.stderr + result.stdout).lower()
        if "err_nvgpuctrperm" in combined_output or "permission" in combined_output:
            logger.warning("NCU permission error — switching to hybrid profiler permanently")
            self._ncu_failed_permanently = True
            return None
        if "importer" in combined_output and "not found" in combined_output:
            logger.warning("NCU importer missing — switching to hybrid profiler permanently")
            self._ncu_failed_permanently = True
            return None
        if "no kernels were profiled" in combined_output:
            logger.warning("NCU: no kernels profiled (NCU version may be too old for this GPU) "
                          "— switching to hybrid profiler permanently")
            self._ncu_failed_permanently = True
            return None
        if "argument" in combined_output and "invalid" in combined_output:
            logger.warning("NCU: unsupported argument — switching to hybrid profiler permanently")
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
        compiler_metrics: 'CompilerMetrics' = None,
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
            compiler_metrics=compiler_metrics,
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
        ok, err, binary, cm = self.compile_kernel(kernel_src, harness_src, name)
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
            timing_us=timing_us, compiler_metrics=cm,
        )
        if metrics:
            metrics.duration_us = timing_us
            metrics.speedup = speedup

        return True, metrics, speedup
