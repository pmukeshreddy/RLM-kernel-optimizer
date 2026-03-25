"""
beam_search.py — Main beam search orchestrator.
Ties together: RLM engine, NCU profiler, diversity selection, and combination.
"""

from __future__ import annotations
import logging
import time
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
        self.profiler = NCURunner(env.search_config)
        self.selector = DiversitySelector(env.search_config)
        self.clf      = BottleneckClassifier(env.search_config)
        self.beam_w   = env.search_config["beam"]["width"]
        self.rounds   = env.search_config["beam"]["refine_rounds"]

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
        return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
void launch_fused_add_rmsnorm_nvfp4(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, unsigned char*, __nv_bfloat16*, int, int, cudaStream_t);
int main(int argc, char** argv) {{
    int warmup=5, iters=100;
    for(int i=1;i<argc;++i){{ sscanf(argv[i],"--warmup=%d",&warmup); sscanf(argv[i],"--iters=%d",&iters); }}
    const int rows={rows}, hidden={hidden}, N={n}, nb={nb};
    __nv_bfloat16 *di,*dr,*dw,*dro,*ds; unsigned char *dq;
    cudaMalloc(&di,N*2); cudaMalloc(&dr,N*2); cudaMalloc(&dw,hidden*2);
    cudaMalloc(&dro,N*2); cudaMalloc(&dq,N/2); cudaMalloc(&ds,nb*2);
    cudaStream_t s; cudaStreamCreate(&s);
    for(int i=0;i<warmup;++i) launch_fused_add_rmsnorm_nvfp4(di,dr,dw,dro,dq,ds,rows,hidden,s);
    cudaStreamSynchronize(s);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0,s);
    for(int i=0;i<iters;++i) launch_fused_add_rmsnorm_nvfp4(di,dr,dw,dro,dq,ds,rows,hidden,s);
    cudaEventRecord(t1,s); cudaStreamSynchronize(s);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("timing_us: %.3f\\n", ms*1000.f/iters);
    cudaFree(di); cudaFree(dr); cudaFree(dw); cudaFree(dro); cudaFree(dq); cudaFree(ds);
    return 0;
}}
"""

    def _harness_silu_mul(self, shape: tuple) -> str:
        b, m, k = shape
        n = b * m * k
        return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
void launch_silu_mul_bf16(
    const __nv_bfloat162*, const __nv_bfloat162*, __nv_bfloat162*, int, cudaStream_t);
int main(int argc, char** argv) {{
    int warmup=5, iters=100;
    for(int i=1;i<argc;++i){{ sscanf(argv[i],"--warmup=%d",&warmup); sscanf(argv[i],"--iters=%d",&iters); }}
    const int N={n};
    __nv_bfloat162 *dg,*du,*dout;
    cudaMalloc(&dg,N*2); cudaMalloc(&du,N*2); cudaMalloc(&dout,N*2);
    cudaStream_t s; cudaStreamCreate(&s);
    for(int i=0;i<warmup;++i) launch_silu_mul_bf16(dg,du,dout,N,s);
    cudaStreamSynchronize(s);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0,s);
    for(int i=0;i<iters;++i) launch_silu_mul_bf16(dg,du,dout,N,s);
    cudaEventRecord(t1,s); cudaStreamSynchronize(s);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("timing_us: %.3f\\n", ms*1000.f/iters);
    cudaFree(dg); cudaFree(du); cudaFree(dout);
    return 0;
}}
"""

    def _harness_nvfp4_quantize(self, shape: tuple) -> str:
        m, k = shape
        n = m * k
        nb = n // 16
        return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
void launch_nvfp4_quantize_bf16(
    const __nv_bfloat16*, uint8_t*, __nv_bfloat16*, int, cudaStream_t);
int main(int argc, char** argv) {{
    int warmup=5, iters=100;
    for(int i=1;i<argc;++i){{ sscanf(argv[i],"--warmup=%d",&warmup); sscanf(argv[i],"--iters=%d",&iters); }}
    const int N={n}, nb={nb};
    __nv_bfloat16 *din,*dsc; uint8_t *dpk;
    cudaMalloc(&din,N*2); cudaMalloc(&dpk,N/2); cudaMalloc(&dsc,nb*2);
    cudaStream_t s; cudaStreamCreate(&s);
    for(int i=0;i<warmup;++i) launch_nvfp4_quantize_bf16(din,dpk,dsc,N,s);
    cudaStreamSynchronize(s);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0,s);
    for(int i=0;i<iters;++i) launch_nvfp4_quantize_bf16(din,dpk,dsc,N,s);
    cudaEventRecord(t1,s); cudaStreamSynchronize(s);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("timing_us: %.3f\\n", ms*1000.f/iters);
    cudaFree(din); cudaFree(dpk); cudaFree(dsc);
    return 0;
}}
"""

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
            self.env.hack_rejections.append(
                {"strategy": candidate.strategy, "hack_type": hack_type,
                 "round": candidate.round_num}
            )
            return None

        harness = self._build_harness(problem_shape)
        name    = f"{candidate.strategy}_r{candidate.round_num}_{int(time.time())}"

        ok, metrics, speedup = self.profiler.compile_and_profile(
            kernel_src=candidate.code,
            harness_src=harness,
            name=name,
            baseline_us=baseline_us,
        )
        candidate.compile_ok = ok

        # Runtime hack checks — run after compile confirms the kernel is valid CUDA
        if ok:
            rt_clean, rt_hack = run_runtime_checks(candidate.code)
            if not rt_clean:
                logger.warning("Runtime hack detected in candidate [%s]: %s — rejecting",
                               candidate.strategy, rt_hack)
                candidate.compile_ok = False
                candidate.speedup    = 0.0
                candidate.bottleneck = "unknown"
                self.env.hack_rejections.append(
                    {"strategy": candidate.strategy, "hack_type": f"runtime:{rt_hack}",
                     "round": candidate.round_num}
                )
                return None

        if metrics:
            candidate.metrics    = metrics.to_dict()
            candidate.speedup    = speedup
            candidate.bottleneck = self.clf.classify(metrics).value
        return metrics

    def run(self) -> KernelCandidate:
        """Execute the full beam search. Returns the best KernelCandidate."""
        env           = self.env
        problem_shape = env.problem_shapes[0]
        baseline_us   = env.baseline_us_reported

        logger.info("="*60)
        logger.info("Beam search: kernel=%s beam_width=%d rounds=%d",
                    env.kernel_name, self.beam_w, self.rounds)
        logger.info("="*60)

        # ── Round 0: Decompose + generate initial beams ────────────────────
        strategies = self.engine.run_decompose()
        logger.info("Strategies: %s", strategies)
        env.current_round = 0

        kernel_slice = env.get_hot_loop_src()
        candidates = self.engine.run_generate_beams(
            strategies=strategies, kernel_slice=kernel_slice, round_num=0
        )

        logger.info("Profiling %d initial beams...", len(candidates))
        metrics_list = []
        for c in candidates:
            m = self._profile_candidate(c, problem_shape, baseline_us)
            metrics_list.append((c, m or KernelMetrics()))
            env.optimization_history.record(c)
            logger.info("  %s", c.summary())

        survivors = self.selector.select_survivors(metrics_list, max_survivors=self.beam_w)

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
            refined = self.engine.run_refine_beams(survivors, round_num)

            new_metrics = []
            for c in refined:
                m = self._profile_candidate(c, problem_shape, baseline_us)
                new_metrics.append((c, m or KernelMetrics()))
                env.optimization_history.record(c)
                logger.info("  Refined: %s", c.summary())

            all_candidates = [
                (s, metrics_from_dict(s.metrics) if s.metrics else KernelMetrics())
                for s in survivors
            ] + new_metrics
            survivors = self.selector.select_survivors(all_candidates, max_survivors=self.beam_w)

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
        return final
