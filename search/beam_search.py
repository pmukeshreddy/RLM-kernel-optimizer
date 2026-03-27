"""
beam_search.py — Main beam search orchestrator.
Ties together: RLM engine, NCU profiler, diversity selection, and combination.
"""

from __future__ import annotations
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from rlm.engine import RLMEngine
from rlm.environment import RLMEnvironment, KernelCandidate
from profiler.ncu_runner import NCURunner
from profiler.bottleneck_classifier import BottleneckClassifier
from profiler.metrics import KernelMetrics, metrics_from_dict
from search.diversity_selector import DiversitySelector
from eval.hack_detector import is_clean
from eval.runtime_checks import run_runtime_checks
from eval.correctness import CorrectnessChecker

logger = logging.getLogger(__name__)


class BeamSearch:
    """
    NCU-guided beam search for CUDA kernel optimization.

    Algorithm:
      0. Root LLM decomposes → selects strategies
      1. Generate N beams in parallel (sub-LLMs)
      2. Compile + profile each beam with NCU
      3. Classify bottleneck per beam
      4. Select diverse survivors (1 per bottleneck cluster)
      5. Refine each survivor with targeted sub-LLM
      6. Repeat rounds 2-5 until budget or round limit
      7. Combine top-2 survivors → final kernel
    """

    def __init__(self, env: RLMEnvironment):
        self.env      = env
        self.engine   = RLMEngine(env)
        self.profiler = NCURunner(env.search_config, hw_spec=env.hw_spec)
        self.selector = DiversitySelector(env.search_config)
        self.clf      = BottleneckClassifier(env.search_config)
        self.checker  = CorrectnessChecker(env.search_config)
        self.beam_w   = env.search_config["beam"]["width"]
        self.rounds   = env.search_config["beam"]["refine_rounds"]
        self._env_lock = threading.Lock()  # guards shared env counters

    def _build_harness(self, problem_shape: tuple) -> str:
        kt = self.env.kernel_type
        if kt == "add_rmsnorm":
            return self._harness_add_rmsnorm(problem_shape)
        elif kt == "silu_mul":
            return self._harness_silu_mul(problem_shape)
        elif kt == "nvfp4_quantize":
            return self._harness_nvfp4_quantize(problem_shape)
        else:
            raise ValueError(f"Unknown kernel_type: {kt}")

    def _harness_add_rmsnorm(self, shape: tuple) -> str:
        rows, hidden = shape
        n, nb = rows * hidden, rows * hidden // 16
        input_bytes = n * 2 * 2 + hidden * 2  # (input+residual)*bf16 + weight
        return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <stdlib.h>
void launch_fused_add_rmsnorm_nvfp4(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, unsigned char*, __nv_fp8_storage_t*, int, int, cudaStream_t);
int main(int argc, char** argv) {{
    int warmup=500, iters=100;
    for(int i=1;i<argc;++i){{ sscanf(argv[i],"--warmup=%d",&warmup); sscanf(argv[i],"--iters=%d",&iters); }}
    const int rows={rows}, hidden={hidden}, N={n}, nb={nb};
    // Dynamic L2 cache cycling (KernelArena/ThunderKittens 2.0 methodology)
    int l2_bytes=0;
    cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
    int nbufs = 1;
    if ({input_bytes} > 0 && l2_bytes > 0 && {input_bytes} < l2_bytes * 3)
        nbufs = l2_bytes * 3 / {input_bytes} + 1;
    if (nbufs > 256) nbufs = 256;
    if (nbufs < 1) nbufs = 1;
    __nv_bfloat16 **di = (__nv_bfloat16**)malloc(nbufs*sizeof(void*));
    __nv_bfloat16 **dr = (__nv_bfloat16**)malloc(nbufs*sizeof(void*));
    __nv_bfloat16 *dw, *dro; unsigned char *dq; __nv_fp8_storage_t *ds;
    for(int b=0;b<nbufs;++b) {{ cudaMalloc(&di[b],N*2); cudaMalloc(&dr[b],N*2); }}
    cudaMalloc(&dw,hidden*2);
    cudaMalloc(&dro,N*2); cudaMalloc(&dq,N/2); cudaMalloc(&ds,nb);
    cudaStream_t s; cudaStreamCreate(&s);
    // Warmup with L2 cycling
    for(int i=0;i<warmup;++i) launch_fused_add_rmsnorm_nvfp4(di[i%nbufs],dr[i%nbufs],dw,dro,dq,ds,rows,hidden,s);
    cudaStreamSynchronize(s);
    // Timed reps — 2 CUDA events wrapping all reps (ThunderKittens convention)
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0,s);
    for(int i=0;i<iters;++i) launch_fused_add_rmsnorm_nvfp4(di[i%nbufs],dr[i%nbufs],dw,dro,dq,ds,rows,hidden,s);
    cudaEventRecord(t1,s); cudaStreamSynchronize(s);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("timing_us: %.3f\\n", ms*1000.f/iters);
    printf("l2_cycle_bufs: %d\\n", nbufs);
    for(int b=0;b<nbufs;++b) {{ cudaFree(di[b]); cudaFree(dr[b]); }}
    free(di); free(dr);
    cudaFree(dw); cudaFree(dro); cudaFree(dq); cudaFree(ds);
    return 0;
}}
"""

    def _harness_silu_mul(self, shape: tuple) -> str:
        b, m, k = shape
        n = b * m * k
        nb = n // 16
        input_bytes = n * 2 * 2  # (gate + up) * bf16
        return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void launch_silu_mul_fp4quant(
    const __nv_bfloat16*, const __nv_bfloat16*,
    uint8_t*, __nv_fp8_storage_t*, int, cudaStream_t);
int main(int argc, char** argv) {{
    int warmup=500, iters=100;
    for(int i=1;i<argc;++i){{ sscanf(argv[i],"--warmup=%d",&warmup); sscanf(argv[i],"--iters=%d",&iters); }}
    const int N={n}, nb={nb};
    // Dynamic L2 cache cycling
    int l2_bytes=0;
    cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
    int nbufs = 1;
    if ({input_bytes} > 0 && l2_bytes > 0 && {input_bytes} < l2_bytes * 3)
        nbufs = l2_bytes * 3 / {input_bytes} + 1;
    if (nbufs > 256) nbufs = 256;
    if (nbufs < 1) nbufs = 1;
    __nv_bfloat16 **dg = (__nv_bfloat16**)malloc(nbufs*sizeof(void*));
    __nv_bfloat16 **du = (__nv_bfloat16**)malloc(nbufs*sizeof(void*));
    uint8_t *dq; __nv_fp8_storage_t *ds;
    for(int b=0;b<nbufs;++b) {{ cudaMalloc(&dg[b],N*2); cudaMalloc(&du[b],N*2); }}
    cudaMalloc(&dq,N/2); cudaMalloc(&ds,nb);
    cudaStream_t s; cudaStreamCreate(&s);
    for(int i=0;i<warmup;++i) launch_silu_mul_fp4quant(dg[i%nbufs],du[i%nbufs],dq,ds,N,s);
    cudaStreamSynchronize(s);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0,s);
    for(int i=0;i<iters;++i) launch_silu_mul_fp4quant(dg[i%nbufs],du[i%nbufs],dq,ds,N,s);
    cudaEventRecord(t1,s); cudaStreamSynchronize(s);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("timing_us: %.3f\\n", ms*1000.f/iters);
    printf("l2_cycle_bufs: %d\\n", nbufs);
    for(int b=0;b<nbufs;++b) {{ cudaFree(dg[b]); cudaFree(du[b]); }}
    free(dg); free(du);
    cudaFree(dq); cudaFree(ds);
    return 0;
}}
"""

    def _harness_nvfp4_quantize(self, shape: tuple) -> str:
        m, k = shape
        n = m * k
        nb = n // 16
        input_bytes = n * 2  # input * bf16
        return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void launch_nvfp4_quantize_bf16(
    const __nv_bfloat16*, uint8_t*, __nv_fp8_storage_t*, int, cudaStream_t);
int main(int argc, char** argv) {{
    int warmup=500, iters=100;
    for(int i=1;i<argc;++i){{ sscanf(argv[i],"--warmup=%d",&warmup); sscanf(argv[i],"--iters=%d",&iters); }}
    const int N={n}, nb={nb};
    // Dynamic L2 cache cycling
    int l2_bytes=0;
    cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
    int nbufs = 1;
    if ({input_bytes} > 0 && l2_bytes > 0 && {input_bytes} < l2_bytes * 3)
        nbufs = l2_bytes * 3 / {input_bytes} + 1;
    if (nbufs > 256) nbufs = 256;
    if (nbufs < 1) nbufs = 1;
    __nv_bfloat16 **din = (__nv_bfloat16**)malloc(nbufs*sizeof(void*));
    uint8_t *dpk; __nv_fp8_storage_t *dsc;
    for(int b=0;b<nbufs;++b) {{ cudaMalloc(&din[b],N*2); }}
    cudaMalloc(&dpk,N/2); cudaMalloc(&dsc,nb);
    cudaStream_t s; cudaStreamCreate(&s);
    for(int i=0;i<warmup;++i) launch_nvfp4_quantize_bf16(din[i%nbufs],dpk,dsc,N,s);
    cudaStreamSynchronize(s);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0,s);
    for(int i=0;i<iters;++i) launch_nvfp4_quantize_bf16(din[i%nbufs],dpk,dsc,N,s);
    cudaEventRecord(t1,s); cudaStreamSynchronize(s);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("timing_us: %.3f\\n", ms*1000.f/iters);
    printf("l2_cycle_bufs: %d\\n", nbufs);
    for(int b=0;b<nbufs;++b) {{ cudaFree(din[b]); }}
    free(din);
    cudaFree(dpk); cudaFree(dsc);
    return 0;
}}
"""

    def measure_search_baseline(self, problem_shape: tuple) -> Optional[float]:
        """Measure reference kernel timing with the SAME harness used for candidates.

        Ensures symmetric measurement: both baseline and candidate see the same
        L2 cache cycling, same CUDA event timing, same C++ dispatch overhead.
        """
        harness = self._build_harness(problem_shape)
        safe_name = f"baseline_{self.env.kernel_type}_{int(time.time())}"
        ok, err, binary, _ = self.profiler.compile_kernel(
            kernel_src=self.env.kernel_src_raw, harness_src=harness,
            output_name=safe_name,
        )
        if not ok:
            logger.warning("Baseline compilation failed: %s", err[:200])
            return None
        timing = self.profiler.benchmark_timing(binary)
        if timing:
            logger.info("Search baseline (with L2 cycling): %.3f us", timing)
        return timing

    def _profile_candidate(
        self,
        candidate: KernelCandidate,
        problem_shape: tuple,
        baseline_us: float,
    ) -> Optional[KernelMetrics]:
        clean, hack_type = is_clean(candidate.code)
        if not clean:
            logger.warning("Hack detected in candidate [%s]: %s — rejecting",
                           candidate.strategy, hack_type)
            candidate.compile_ok = False
            candidate.speedup    = 0.0
            candidate.bottleneck = "unknown"
            with self._env_lock:
                self.env.hack_rejections.append(
                    {"strategy": candidate.strategy, "hack_type": hack_type,
                     "round": candidate.round_num}
                )
            return None

        harness = self._build_harness(problem_shape)
        # Sanitize strategy name for filesystem (freeform names may have spaces/slashes)
        safe_strat = re.sub(r'[^a-zA-Z0-9_-]', '_', candidate.strategy)
        name    = f"{safe_strat}_r{candidate.round_num}_{int(time.time())}_{id(candidate)}"

        # Compile, check correctness, profile.
        # Note: counters reflect ALL attempts including inner refinement retries.
        ok = False
        metrics = None
        speedup = 0.0
        with self._env_lock:
            self.env.total_attempts += 1

        compile_ok, err_msg, binary, compiler_metrics = self.profiler.compile_kernel(
            kernel_src=candidate.code, harness_src=harness, output_name=name,
        )
        if not compile_ok:
            candidate.compile_ok = False
            candidate.compile_error = err_msg[:800]
            logger.error("  Compile FAIL [%s]: %s", candidate.strategy, err_msg[:400])
        if compile_ok:
            candidate.compile_ok = True  # nvcc succeeded
            with self._env_lock:
                self.env.compile_passes += 1
            passed, max_err, msg = self.checker.check(candidate.code, problem_shape,
                                                          kernel_type=self.env.kernel_type)
            if not passed:
                candidate.compile_error = f"Correctness: {msg[:600]}"
                logger.warning("  Correctness FAIL [%s] (err=%.4f): %s",
                               candidate.strategy, max_err, msg[:200])
            else:
                with self._env_lock:
                    self.env.correctness_passes += 1
                candidate.correct = True
                timing_us = self.profiler.benchmark_timing(binary)
                if timing_us is not None:
                    speedup = baseline_us / timing_us if timing_us > 0 else 0.0
                    metrics = self.profiler.profile(
                        binary, report_name=name,
                        kernel_src=candidate.code,
                        kernel_type=self.env.kernel_type,
                        problem_shape=problem_shape,
                        baseline_us=baseline_us,
                        timing_us=timing_us,
                        compiler_metrics=compiler_metrics,
                    )
                    if metrics:
                        metrics.duration_us = timing_us
                        metrics.speedup = speedup
                        logger.info("  Profiler: occ=%.1f%% timing=%.1fus speedup=%.3fx",
                                    metrics.sm_occupancy, metrics.duration_us, metrics.speedup)
                    else:
                        logger.warning("  Profiler returned no metrics for [%s]", candidate.strategy)
                ok = True

        # Runtime hack checks — run after compile confirms the kernel is valid CUDA
        if ok:
            rt_clean, rt_hack = run_runtime_checks(candidate.code, kernel_type=self.env.kernel_type)
            if not rt_clean:
                logger.warning("Runtime hack detected in candidate [%s]: %s — rejecting",
                               candidate.strategy, rt_hack)
                candidate.compile_ok = False
                candidate.speedup    = 0.0
                candidate.bottleneck = "unknown"
                with self._env_lock:
                    self.env.hack_rejections.append(
                        {"strategy": candidate.strategy, "hack_type": f"runtime:{rt_hack}",
                         "round": candidate.round_num}
                    )
                return None

        # Always set speedup from timing, even if NCU profiling failed
        candidate.speedup = speedup
        if metrics:
            candidate.metrics    = metrics.to_dict()
            candidate.bottleneck = self.clf.classify(metrics).value
        return metrics

    def _make_inner_profile_fn(self, problem_shape, baseline_us):
        """Create a profiling callback for the engine's inner refinement loop."""
        def fn(code, strategy, round_num):
            temp = KernelCandidate(
                code=code,
                strategy=f"{strategy}_inner",
                round_num=round_num,
                compile_ok=bool(code),
            )
            self._profile_candidate(temp, problem_shape, baseline_us)
            error = temp.compile_error
            if not error and not temp.compile_ok:
                error = "Rejected: code did not pass validation checks"
            if not error and not temp.correct:
                error = "Correctness failure: output mismatch"
            return {
                "compile_ok": temp.compile_ok,
                "correct": temp.correct,
                "speedup": temp.speedup,
                "metrics": temp.metrics,
                "bottleneck": temp.bottleneck,
                "error": error,
            }
        return fn

    def _profile_candidates_parallel(
        self,
        candidates: list,
        problem_shape: tuple,
        baseline_us: float,
    ) -> list:
        """Profile all candidates in parallel using a thread pool.
        Returns list of (candidate, KernelMetrics) tuples preserving input order."""
        if not candidates:
            return []

        results = [None] * len(candidates)

        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            future_to_idx = {
                pool.submit(self._profile_candidate, c, problem_shape, baseline_us): i
                for i, c in enumerate(candidates)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    m = future.result()
                except Exception as e:
                    logger.error("Profiling thread failed for [%s]: %s",
                                 candidates[idx].strategy, e)
                    m = None
                results[idx] = (candidates[idx], m or KernelMetrics())

        return results

    def run(self) -> KernelCandidate:
        """Execute the full beam search. Returns the best KernelCandidate."""
        env           = self.env
        problem_shape = env.problem_shapes[0]

        # Measure baseline with the SAME harness (L2 cycling, CUDA events) as candidates
        # so the speedup comparison is symmetric.  Fall back to FlashInfer measurement.
        search_baseline = self.measure_search_baseline(problem_shape)
        baseline_us = search_baseline if search_baseline else env.baseline_us_reported
        if search_baseline:
            logger.info("Using search-harness baseline: %.3f us (FlashInfer was %.3f us)",
                        baseline_us, env.baseline_us_reported)
        else:
            logger.info("Using FlashInfer baseline: %.3f us", baseline_us)

        logger.info("="*60)
        logger.info("Beam search: kernel=%s beam_width=%d rounds=%d",
                    env.kernel_name, self.beam_w, self.rounds)
        logger.info("="*60)

        # ── Round 0: Decompose + generate initial beams ────────────────────
        all_strategies = self.engine.run_decompose()
        strategies = all_strategies[:self.beam_w]
        self._reserve_strategies = all_strategies[self.beam_w:]
        logger.info("Active strategies: %s", [s.get("name", s) if isinstance(s, dict) else s for s in strategies])
        logger.info("Reserve strategies: %s", [s.get("name", s) if isinstance(s, dict) else s for s in self._reserve_strategies])
        env.current_round = 0

        kernel_slice = env.kernel_src
        profile_fn = self._make_inner_profile_fn(problem_shape, baseline_us)
        candidates = self.engine.run_generate_beams(
            strategies=strategies, kernel_slice=kernel_slice, round_num=0,
            profile_fn=profile_fn,
        )

        # Separate pre-profiled (from tool loop) vs unprofiled (from one-shot fallback)
        pre_profiled = [c for c in candidates if c.metrics]
        need_profiling = [c for c in candidates if not c.metrics]

        if need_profiling:
            logger.info("Profiling %d one-shot beams in parallel...", len(need_profiling))
            fresh_metrics = self._profile_candidates_parallel(
                need_profiling, problem_shape, baseline_us)
        else:
            fresh_metrics = []

        metrics_list = [
            (c, metrics_from_dict(c.metrics) if c.metrics else KernelMetrics())
            for c in pre_profiled
        ] + fresh_metrics

        for c, _ in metrics_list:
            env.optimization_history.record(c)
            logger.info("  %s", c.summary())

        # Collect failed round-0 strategies for retry — these were the LLM's
        # top picks but failed due to implementation bugs (not bad ideas)
        self._failed_strategies = []
        for (c, _), strat in zip(metrics_list, strategies):
            if not c.is_viable():
                error = c.compile_error or "Unknown failure"
                self._failed_strategies.append((strat, error))
        if self._failed_strategies:
            names = [s.get("name", s) if isinstance(s, dict) else s
                     for s, _ in self._failed_strategies]
            logger.info("Failed strategies saved for retry: %s", names)

        survivors = self.selector.select_survivors(metrics_list, max_survivors=self.beam_w)
        for s in survivors:
            if s.prev_metrics is None:
                s.prev_metrics = s.metrics
            # Initialize best-code tracking for refinement
            s.best_code = s.code
            s.best_speedup = s.speedup

        # ── Rounds 1-N: Refine ─────────────────────────────────────────────
        for round_num in range(1, self.rounds + 1):
            env.current_round = round_num

            if env.over_budget():
                logger.warning("Budget exhausted at round %d", round_num)
                break
            if not survivors:
                logger.warning("No viable survivors — stopping early")
                break

            logger.info("--- Round %d ---", round_num)

            # Split survivors into refineable vs stagnant
            to_refine = []
            stagnant = []
            for s in survivors:
                if s.refine_attempts >= 3:
                    stagnant.append(s)
                    logger.info("  Retiring stagnant [%s] after %d failed refine attempts",
                                s.strategy, s.refine_attempts)
                else:
                    to_refine.append(s)
                    # Don't increment here — increment based on outcome below

            # Refine non-stagnant survivors (inner loop profiles via callback)
            refined = []
            if to_refine:
                profile_fn = self._make_inner_profile_fn(problem_shape, baseline_us)
                refined = self.engine.run_refine_beams(
                    to_refine, round_num, profile_fn=profile_fn)

            # Fill stagnant slots: first retry failed strategies, then use reserves
            fresh = []
            slots_available = len(stagnant)
            best_code = max(survivors, key=lambda c: c.speedup).code if survivors else kernel_slice

            # Priority 1: retry failed strategies with error context
            if slots_available > 0 and self._failed_strategies:
                n_retry = min(slots_available, len(self._failed_strategies))
                to_retry = self._failed_strategies[:n_retry]
                self._failed_strategies = self._failed_strategies[n_retry:]

                retry_strats = []
                for strat, error in to_retry:
                    name = strat.get("name", strat) if isinstance(strat, dict) else strat
                    what = strat.get("what", "") if isinstance(strat, dict) else ""
                    retry_strats.append({
                        "name": name,
                        "what": (what + f"\n\nWARNING — Previous attempt FAILED:\n{error[:400]}\n"
                                 "Fix the bug. Common cause: __syncthreads() inside if/else "
                                 "causes deadlock — all threads in a block MUST hit every barrier."),
                    })

                logger.info("  Retrying %d failed strategies: %s",
                            n_retry, [s["name"] for s in retry_strats])
                fresh = self.engine.run_generate_beams(
                    strategies=retry_strats, kernel_slice=best_code,
                    round_num=round_num, profile_fn=profile_fn,
                )
                slots_available -= n_retry

            # Priority 2: fresh reserve strategies for remaining slots
            if slots_available > 0 and self._reserve_strategies:
                n_fresh = min(slots_available, len(self._reserve_strategies))
                fresh_strats = self._reserve_strategies[:n_fresh]
                self._reserve_strategies = self._reserve_strategies[n_fresh:]
                logger.info("  Replacing %d stagnant beams with fresh strategies: %s",
                            n_fresh, [s.get("name", s) if isinstance(s, dict) else s for s in fresh_strats])
                fresh += self.engine.run_generate_beams(
                    strategies=fresh_strats, kernel_slice=best_code,
                    round_num=round_num, profile_fn=profile_fn,
                )

            # Refined candidates are already profiled by the inner loop.
            # Fresh beams may be pre-profiled (tool loop) or need profiling (one-shot).
            fresh_pre_profiled = [c for c in fresh if c.metrics]
            fresh_need_profiling = [c for c in fresh if not c.metrics]
            if fresh_need_profiling:
                fresh_metrics = self._profile_candidates_parallel(
                    fresh_need_profiling, problem_shape, baseline_us)
            else:
                fresh_metrics = []

            # Combine: refined (already profiled) + fresh pre-profiled + fresh just-profiled
            new_metrics = [
                (c, metrics_from_dict(c.metrics) if c.metrics else KernelMetrics())
                for c in refined
            ] + [
                (c, metrics_from_dict(c.metrics) if c.metrics else KernelMetrics())
                for c in fresh_pre_profiled
            ] + fresh_metrics

            for c, _ in new_metrics:
                env.optimization_history.record(c)
                logger.info("  New: %s", c.summary())

            # Track refinement outcomes on parent survivors + build history
            for i, (refined_c, _) in enumerate(new_metrics):
                if i < len(to_refine):
                    parent = to_refine[i]
                    entry = {"round": round_num, "strategy": refined_c.strategy,
                             "speedup": refined_c.speedup,
                             "strategy_desc": getattr(refined_c, 'strategy_desc', '')}

                    if not refined_c.compile_ok:
                        entry["outcome"] = "compile_fail"
                        parent.last_refine_error = refined_c.compile_error or "Compile failure"
                        parent.refine_attempts += 1
                    elif not refined_c.correct:
                        entry["outcome"] = "correctness_fail"
                        parent.last_refine_error = refined_c.compile_error or "Correctness failure (output mismatch or kernel hung)"
                        parent.refine_attempts += 1
                    elif refined_c.speedup < parent.speedup - 0.001:
                        entry["outcome"] = "regression"
                        parent.refine_attempts += 1
                        msg = f"Your refinement was SLOWER: {refined_c.speedup:.3f}x vs {parent.speedup:.3f}x."
                        rm = refined_c.metrics or {}
                        rc = rm.get("_compiler", {})
                        pm = parent.metrics or {}
                        pc = pm.get("_compiler", {})
                        if rc and pc:
                            r_regs = rc.get("registers_per_thread", 0)
                            p_regs = pc.get("registers_per_thread", 0)
                            r_occ = rm.get("sm_occupancy", 0)
                            p_occ = pm.get("sm_occupancy", 0)
                            if r_regs != p_regs or r_occ != p_occ:
                                msg += f" Registers: {p_regs}->{r_regs}, Occupancy: {p_occ:.0f}%->{r_occ:.0f}%."
                        parent.last_refine_error = msg
                    elif refined_c.speedup > parent.speedup + 0.02:
                        entry["outcome"] = "improved"
                        parent.last_refine_error = ""
                        parent.refine_attempts = 0  # reset on real improvement
                        # Update best-known code for this beam lineage
                        if refined_c.speedup > (parent.best_speedup or 0):
                            refined_c.best_code = refined_c.code
                            refined_c.best_speedup = refined_c.speedup
                    else:
                        entry["outcome"] = "stagnant"
                        parent.refine_attempts += 1
                        parent.last_refine_error = (
                            f"Refinement produced same speedup ({refined_c.speedup:.3f}x vs "
                            f"{parent.speedup:.3f}x). Try a fundamentally different approach."
                        )

                    # Update prev_metrics so the delta section shows what the
                    # failed refinement changed — closes the feedback loop
                    if refined_c.metrics:
                        parent.prev_metrics = refined_c.metrics

                    parent.refinement_history.append(entry)
                    refined_c.refinement_history = list(parent.refinement_history)

            # Problem 2: only promote refinements that actually improved
            improved_new = []
            for i, (refined_c, m) in enumerate(new_metrics):
                if i < len(to_refine):
                    parent = to_refine[i]
                    if refined_c.speedup > parent.speedup:
                        improved_new.append((refined_c, m))
                    # else: regressed/stagnant — don't pollute the beam
                else:
                    # Fresh beams always enter the pool
                    improved_new.append((refined_c, m))

            all_candidates = [
                (s, metrics_from_dict(s.metrics) if s.metrics else KernelMetrics())
                for s in survivors
            ] + improved_new
            prev_survivor_ids = {id(s) for s in survivors}
            survivors = self.selector.select_survivors(all_candidates, max_survivors=self.beam_w)
            # Only set prev_metrics on NEW entrants; carried-over survivors keep theirs
            for s in survivors:
                if id(s) not in prev_survivor_ids:
                    s.prev_metrics = s.metrics

        # ── Final: Combine top-2 ───────────────────────────────────────────
        top_for_combine = self.selector.select_for_combination(survivors)
        logger.info("Combining %d top beams", len(top_for_combine))

        if len(top_for_combine) >= 2:
            final = self.engine.combine(top_for_combine)
            m     = self._profile_candidate(final, problem_shape, baseline_us)
            best_survivor = max(survivors, key=lambda c: c.speedup) if survivors else None
            if best_survivor and final.speedup < best_survivor.speedup * 0.95:
                logger.warning("Combination regressed (%.3fx < %.3fx) — reverting",
                               final.speedup, best_survivor.speedup)
                final = best_survivor
        else:
            final = (top_for_combine[0] if top_for_combine
                     else survivors[0] if survivors
                     else candidates[0])

        logger.info("="*60)
        logger.info("Search complete: %s", final.summary())
        logger.info("Total API cost: $%.4f", env.total_api_cost_usd)
        logger.info("="*60)

        # Clean up async client to prevent 'Event loop is closed' errors
        self.engine.close()

        return final
