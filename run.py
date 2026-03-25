"""
run.py — Entry point: solve all WaferBench NVFP4 kernel problems.

Usage:
    python run.py                          # optimize all kernels
    python run.py --kernel add_rmsnorm     # single kernel
    python run.py --dry-run                # validate setup, no LLM calls
    python run.py --beam-width 2 --rounds 2
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from rlm.environment import RLMEnvironment
from search.beam_search import BeamSearch
from eval.correctness import CorrectnessChecker
from eval.benchmark import Benchmarker, geometric_mean
from eval.waferbench_format import format_submission, save_submission, print_submission_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rlm_optimizer.log"),
    ],
)
logger = logging.getLogger(__name__)

WAFERBENCH_KERNELS = [
    {
        "name":        "add_rmsnorm_fp4quant_b128xh2048",
        "src":         "kernels/reference/add_rmsnorm.cu",
        "kernel_type": "add_rmsnorm",
        "description": "Fused Add+RMSNorm+NVFP4 quant (128×2048)",
        "shape":       (128, 2048),
        "baseline_us": 13.5,
    },
    {
        "name":        "add_rmsnorm_fp4quant_b128xh4096",
        "src":         "kernels/reference/add_rmsnorm.cu",
        "kernel_type": "add_rmsnorm",
        "description": "Fused Add+RMSNorm+NVFP4 quant (128×4096)",
        "shape":       (128, 4096),
        "baseline_us": 13.9,
    },
    {
        "name":        "add_rmsnorm_fp4quant_b128xh8192",
        "src":         "kernels/reference/add_rmsnorm.cu",
        "kernel_type": "add_rmsnorm",
        "description": "Fused Add+RMSNorm+NVFP4 quant (128×8192)",
        "shape":       (128, 8192),
        "baseline_us": 13.0,
    },
    {
        "name":        "nvfp4_quantize_m128xk14336",
        "src":         "kernels/reference/nvfp4_quantize.cu",
        "kernel_type": "nvfp4_quantize",
        "description": "BF16→NVFP4 block quantization (128×14336)",
        "shape":       (128, 14336),
        "baseline_us": 7.6,
    },
    {
        "name":        "silu_mul_fp4quant_b8xm256xk7168",
        "src":         "kernels/reference/silu_mul.cu",
        "kernel_type": "silu_mul",
        "description": "Fused SiLU×Mul+NVFP4 quant (8×256×7168)",
        "shape":       (8, 256, 7168),
        "baseline_us": 22.7,
    },
    {
        "name":        "silu_mul_fp4quant_b8xm256xk14336",
        "src":         "kernels/reference/silu_mul.cu",
        "kernel_type": "silu_mul",
        "description": "Fused SiLU×Mul+NVFP4 quant (8×256×14336)",
        "shape":       (8, 256, 14336),
        "baseline_us": 39.1,
    },
]


def optimize_kernel(
    kernel_def: dict,
    config: dict,
    dry_run: bool = False,
    output_dir: Path = None,
) -> dict:
    name        = kernel_def["name"]
    src_path    = PROJECT_ROOT / kernel_def["src"]
    baseline    = kernel_def["baseline_us"]
    kernel_type = kernel_def["kernel_type"]
    shape       = kernel_def["shape"]

    logger.info("\n%s\nOptimizing: %s\n%s", "="*60, name, "="*60)

    if not src_path.exists():
        logger.error("Kernel source not found: %s", src_path)
        return {}

    env = RLMEnvironment(kernel_name=name, kernel_src_path=str(src_path),
                         kernel_type=kernel_type, problem_shape=shape)
    env.baseline_us_reported = baseline

    overrides = config.get("_overrides", {})
    if "beam_width" in overrides:
        env.search_config["beam"]["width"] = overrides["beam_width"]
    if "rounds" in overrides:
        env.search_config["beam"]["refine_rounds"] = overrides["rounds"]

    logger.info(env.state_summary())

    if dry_run:
        logger.info("[DRY RUN] Skipping LLM calls for %s", name)
        return {"kernel_name": name, "dry_run": True, "status": "skipped"}

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    start_time = time.time()
    beam_search = BeamSearch(env)
    best        = beam_search.run()
    elapsed     = time.time() - start_time
    logger.info("Beam search completed in %.1f seconds", elapsed)

    checker  = CorrectnessChecker(env.search_config)
    passed, max_err, msg = checker.check(best.code, env.problem_shapes[0])
    best.correct = passed
    logger.info("Correctness: %s (max_err=%.6f)", "PASS" if passed else "FAIL", max_err)

    if not passed:
        logger.error("Correctness FAILED — falling back to reference kernel")
        best.code = src_path.read_text()

    benchmarker = Benchmarker(env.search_config, kernel_type=kernel_type)

    # Measure actual baseline on current hardware instead of using hardcoded values
    ref_src = src_path.read_text()
    baseline_per_shape = {}
    for s in env.problem_shapes:
        measured = benchmarker._compile_and_time(ref_src, s)
        if measured is not None:
            baseline_per_shape[s] = measured
            logger.info("Measured baseline for %s: %.2f us (hardcoded: %.2f us)", s, measured, baseline)
        else:
            baseline_per_shape[s] = baseline
            logger.warning("Baseline measurement failed for %s, using hardcoded %.2f us", s, baseline)

    bench_results = benchmarker.benchmark(best.code, baseline_per_shape)

    submission = format_submission(
        kernel_name=name,
        kernel_src=best.code,
        benchmark_results=bench_results,
        metadata={
            "strategy":         best.strategy,
            "bottleneck":       best.bottleneck,
            "search_rounds":    env.current_round,
            "api_cost_usd":     env.total_api_cost_usd,
            "elapsed_seconds":  elapsed,
            "correct":          passed,
            "max_abs_err":      max_err,
            "hack_rejections":  env.hack_rejections,
            "total_attempts":       env.total_attempts,
            "compile_passes":       env.compile_passes,
            "correctness_passes":   env.correctness_passes,
        },
    )

    if output_dir:
        save_submission(submission, output_dir, name)
        logger.info("Saved to %s", output_dir)

    print_submission_summary(submission)
    return submission


def main():
    parser = argparse.ArgumentParser(
        description="RLM + NCU-guided beam search kernel optimizer for WaferBench NVFP4"
    )
    parser.add_argument("--kernel",     type=str, default=None)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--rounds",     type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/submissions")
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--log-level",  type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    config_path = args.config or PROJECT_ROOT / "config" / "search_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["_overrides"] = {}
    if args.beam_width: config["_overrides"]["beam_width"] = args.beam_width
    if args.rounds:     config["_overrides"]["rounds"]     = args.rounds

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kernels = WAFERBENCH_KERNELS
    if args.kernel:
        kernels = [k for k in WAFERBENCH_KERNELS if k["name"] == args.kernel]
        if not kernels:
            logger.error("Unknown kernel: %s. Available: %s",
                         args.kernel, [k["name"] for k in WAFERBENCH_KERNELS])
            sys.exit(1)

    all_results = []
    total_cost  = 0.0
    for kernel_def in kernels:
        result = optimize_kernel(
            kernel_def=kernel_def,
            config=config,
            dry_run=args.dry_run,
            output_dir=output_dir,
        )
        all_results.append(result)
        total_cost += result.get("metadata", {}).get("api_cost_usd", 0.0)

    print(f"\n{'='*60}")
    print(f"All kernels complete. Total API cost: ${total_cost:.4f}")
    if not args.dry_run:
        speedups = [
            r.get("performance", {}).get("geomean_speedup", 0)
            for r in all_results if "performance" in r
        ]
        if speedups:
            print(f"Geometric mean speedup (all kernels): {geometric_mean(speedups):.3f}x")

        # Pass rate summary
        total_att = sum(r.get("metadata", {}).get("total_attempts", 0) for r in all_results)
        total_comp = sum(r.get("metadata", {}).get("compile_passes", 0) for r in all_results)
        total_corr = sum(r.get("metadata", {}).get("correctness_passes", 0) for r in all_results)
        if total_att > 0:
            print(f"\nFirst-try pass rates ({total_att} total attempts):")
            print(f"  Compile:     {total_comp}/{total_att} ({100*total_comp/total_att:.0f}%)")
            print(f"  Correctness: {total_corr}/{total_att} ({100*total_corr/total_att:.0f}%)")

        # Hack detection summary
        all_rejections = [
            rej
            for r in all_results
            for rej in r.get("metadata", {}).get("hack_rejections", [])
        ]
        if all_rejections:
            print(f"\nHack rejections: {len(all_rejections)} candidate(s) disqualified")
            from collections import Counter
            for hack_type, count in Counter(r["hack_type"] for r in all_rejections).items():
                print(f"  {hack_type}: {count}")
        else:
            print("Hack rejections: 0")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
