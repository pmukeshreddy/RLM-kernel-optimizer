"""
beam_search.py — K-Search tree search with world model.

Implements the core K-Search algorithm (arxiv 2602.19128):
1. World model scores strategies BEFORE trying them
2. Deterministic selection: argmax by WM score
3. Action cycles with stagnation window K (multiple attempts per strategy)
4. WM evolution: Insert new strategies / Update scores / Prune dead ends
5. Tree search with backtracking to earlier checkpoints
"""

from __future__ import annotations
import json
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from rlm.engine import RLMEngine
from rlm.environment import RLMEnvironment, KernelCandidate
from profiler.ncu_runner import NCURunner
from profiler.bottleneck_classifier import BottleneckClassifier
from profiler.metrics import KernelMetrics, metrics_from_dict
from search.diversity_selector import DiversitySelector
from search.world_model import WorldModel
from eval.hack_detector import is_clean
from eval.runtime_checks import run_runtime_checks
from eval.correctness import CorrectnessChecker

logger = logging.getLogger(__name__)

# ── K-Search parameters ───────────────────────────────────────────────────
STAGNATION_K = 5           # consecutive non-improving attempts before closing a node
MAX_NODE_EXPANSIONS = 6    # max total attempts per node
MAX_DIFFICULTY = 4         # skip strategies with difficulty > this (until we have good results)
DIFFICULTY_UNLOCK_SPEEDUP = 1.3  # unlock difficulty 5 after this speedup


@dataclass
class SearchNode:
    """A node in the K-Search optimization tree.

    OPEN nodes have strategy metadata but no solution yet (frontier).
    CLOSED nodes have an attached best solution from their action cycle.
    """
    node_id: str
    strategy_name: str
    strategy_description: str = ""
    score: float = 0.5           # WM priority score (0-1)
    difficulty: int = 3          # estimated difficulty (1-5)

    # Solution state
    candidate: Optional[KernelCandidate] = None  # best solution (None = OPEN)
    is_open: bool = True

    # Tree structure
    parent: Optional[SearchNode] = None
    children: List[SearchNode] = field(default_factory=list)
    depth: int = 0

    # Action cycle tracking
    attempt_count: int = 0       # total implementation attempts
    stagnant_count: int = 0      # consecutive non-improving attempts
    best_cycle_speedup: float = 0.0  # best speedup in current action cycle

    @property
    def is_frontier(self) -> bool:
        """Can be selected for the next action cycle."""
        if not self.is_open:
            return False
        if self.attempt_count >= MAX_NODE_EXPANSIONS:
            return False
        # Parent must have a solution (or be root)
        if self.parent is None:
            return True
        return not self.parent.is_open

    @property
    def speedup(self) -> float:
        if self.candidate and self.candidate.is_viable():
            return self.candidate.speedup
        return 0.0

    @property
    def parent_code(self) -> str:
        """Base code to generate from (parent's solution)."""
        if self.parent and self.parent.candidate:
            return self.parent.candidate.code
        return ""


class SearchTree:
    """Manages the K-Search optimization tree with WM integration."""

    def __init__(self):
        self.root: Optional[SearchNode] = None
        self.all_nodes: List[SearchNode] = []
        self.global_best: Optional[SearchNode] = None
        self._next_id = 0

    def _gen_id(self) -> str:
        self._next_id += 1
        return f"n{self._next_id}"

    def create_root(self, candidate: KernelCandidate) -> SearchNode:
        """Create the root node (naive kernel)."""
        node = SearchNode(
            node_id=self._gen_id(),
            strategy_name="root",
            strategy_description="Naive reference kernel",
            candidate=candidate,
            is_open=False,  # root is always closed
            depth=0,
        )
        self.root = node
        self.all_nodes.append(node)
        return node

    def add_strategy_node(
        self, parent: SearchNode, name: str, description: str,
        score: float = 0.5, difficulty: int = 3,
    ) -> SearchNode:
        """Add an OPEN strategy node (proposed by WM)."""
        node = SearchNode(
            node_id=self._gen_id(),
            strategy_name=name,
            strategy_description=description,
            score=score,
            difficulty=difficulty,
            parent=parent,
            depth=parent.depth + 1,
            is_open=True,
        )
        parent.children.append(node)
        self.all_nodes.append(node)
        return node

    def close_node(self, node: SearchNode, candidate: KernelCandidate):
        """Close a node by attaching its best solution."""
        node.candidate = candidate
        node.is_open = False
        if candidate.is_viable():
            if self.global_best is None or candidate.speedup > self.global_best.speedup:
                self.global_best = node
                logger.info("NEW GLOBAL BEST: %.3fx [%s] (depth=%d)",
                            candidate.speedup, node.strategy_name, node.depth)

    def choose_top_n_frontier(self, n: int) -> List[SearchNode]:
        """Deterministic selection: argmax by score among frontier nodes.

        K-Search selection: sort by (-score, difficulty, -speedup).
        Difficulty gating: skip difficulty > MAX_DIFFICULTY unless we have
        a good enough solution (DIFFICULTY_UNLOCK_SPEEDUP).
        """
        max_diff = MAX_DIFFICULTY
        if self.global_best and self.global_best.speedup >= DIFFICULTY_UNLOCK_SPEEDUP:
            max_diff = 5  # unlock hard strategies once we have a good base

        frontier = [nd for nd in self.all_nodes
                    if nd.is_frontier and nd.difficulty <= max_diff]

        # K-Search sorting: best score first, then easier first, then faster first
        frontier.sort(key=lambda nd: (-nd.score, nd.difficulty))
        return frontier[:n]

    def to_compact_json(self) -> str:
        """Serialize tree for WM prompt (compact representation)."""
        nodes = []
        for nd in self.all_nodes:
            entry = {
                "id": nd.node_id,
                "parent": nd.parent.node_id if nd.parent else None,
                "strategy": nd.strategy_name,
                "description": nd.strategy_description[:100],
                "score": round(nd.score, 2),
                "difficulty": nd.difficulty,
                "status": "CLOSED" if not nd.is_open else "OPEN",
            }
            if nd.candidate and nd.candidate.is_viable():
                entry["speedup"] = round(nd.speedup, 3)
                m = nd.candidate.metrics or {}
                cm = m.get("_compiler", {})
                entry["regs"] = cm.get("registers_per_thread", 0)
                entry["vec_ld_pct"] = cm.get("vectorized_load_pct", 0)
                entry["sass_insts"] = cm.get("sass_total_instructions", 0)
            if nd.attempt_count > 0:
                entry["attempts"] = nd.attempt_count
                entry["stagnant"] = nd.stagnant_count
            nodes.append(entry)
        return json.dumps(nodes, indent=1)

    def apply_wm_edits(self, edits: dict):
        """Apply Insert/Update/Delete operations from the world model."""
        id_map = {nd.node_id: nd for nd in self.all_nodes}

        # Updates
        for upd in edits.get("updates", []):
            node = id_map.get(upd.get("node_id"))
            if node and node.is_open:
                if "score" in upd:
                    old = node.score
                    node.score = max(0.0, min(1.0, float(upd["score"])))
                    logger.info("  WM update [%s] %s: score %.2f -> %.2f (%s)",
                                node.node_id, node.strategy_name,
                                old, node.score,
                                upd.get("reason", "")[:60])
                if "difficulty" in upd:
                    node.difficulty = int(upd["difficulty"])

        # Inserts — add new OPEN strategy nodes
        for ins in edits.get("inserts", []):
            parent = id_map.get(ins.get("parent_id"))
            if not parent:
                # Default to global best or root
                parent = self.global_best or self.root
            if parent:
                name = ins.get("name", "wm_proposed")
                desc = ins.get("what", ins.get("description", ""))
                new_node = self.add_strategy_node(
                    parent, name, desc,
                    score=float(ins.get("score", 0.5)),
                    difficulty=int(ins.get("difficulty", 3)),
                )
                id_map[new_node.node_id] = new_node
                logger.info("  WM insert [%s] %s (score=%.2f, d=%d) under [%s]",
                            new_node.node_id, name, new_node.score,
                            new_node.difficulty, parent.node_id)

        # Deletes — only open leaves with no children
        for dl in edits.get("deletes", []):
            node = id_map.get(dl.get("node_id"))
            if node and node.is_open and not node.children:
                if node.parent:
                    node.parent.children.remove(node)
                self.all_nodes.remove(node)
                logger.info("  WM delete [%s] %s: %s",
                            node.node_id, node.strategy_name,
                            dl.get("reason", "")[:60])

    def has_frontier(self) -> bool:
        return any(nd.is_frontier for nd in self.all_nodes)

    def stats(self) -> str:
        total = len(self.all_nodes)
        open_n = sum(1 for nd in self.all_nodes if nd.is_open)
        frontier = sum(1 for nd in self.all_nodes if nd.is_frontier)
        closed = total - open_n
        max_depth = max((nd.depth for nd in self.all_nodes), default=0)
        best = self.global_best.speedup if self.global_best else 0.0
        return (f"nodes={total} open={open_n} frontier={frontier} "
                f"closed={closed} depth={max_depth} best={best:.3f}x")

    def get_best_n(self, n: int) -> List[SearchNode]:
        """Get n best viable CLOSED nodes."""
        viable = [nd for nd in self.all_nodes
                  if nd.candidate and nd.candidate.is_viable()]
        viable.sort(key=lambda nd: -nd.speedup)
        return viable[:n]


class BeamSearch:
    """
    K-Search tree search for CUDA kernel optimization.

    Algorithm:
      0. World model proposes scored optimization strategies
      1. Deterministic selection: pick top-N frontier nodes by WM score
      2. Action cycles: generate + refine up to K attempts per strategy
      3. Close nodes, attach best solutions
      4. World model evolves: Insert children / Update scores / Prune dead ends
      5. Repeat until budget or no frontier nodes remain
      6. Combine top-2 from tree → final kernel
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
        input_bytes = n * 2 * 2 + hidden * 2
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
    for(int i=0;i<warmup;++i) launch_fused_add_rmsnorm_nvfp4(di[i%nbufs],dr[i%nbufs],dw,dro,dq,ds,rows,hidden,s);
    cudaStreamSynchronize(s);
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
        input_bytes = n * 2 * 2
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
        input_bytes = n * 2
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
        """Measure reference kernel timing with the SAME harness used for candidates."""
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
        safe_strat = re.sub(r'[^a-zA-Z0-9_-]', '_', candidate.strategy)
        name    = f"{safe_strat}_r{candidate.round_num}_{int(time.time())}_{id(candidate)}"

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
            candidate.compile_ok = True
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
        """Profile all candidates in parallel using a thread pool."""
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

    # ── Results formatting for WM ─────────────────────────────────────────

    def _format_round_results(self, results: list) -> str:
        """Format action cycle results for the WM evolve prompt."""
        lines = []
        for node, candidate in results:
            status = "CLOSED" if not node.is_open else "OPEN"
            if candidate and candidate.is_viable():
                m = candidate.metrics or {}
                cm = m.get("_compiler", {})
                lines.append(
                    f"- [{node.node_id}] {node.strategy_name}: "
                    f"speedup={candidate.speedup:.3f}x "
                    f"regs={cm.get('registers_per_thread', '?')} "
                    f"vec_ld%={cm.get('vectorized_load_pct', '?')} "
                    f"sass={cm.get('sass_total_instructions', '?')} "
                    f"status={status}")
            elif candidate:
                err = (candidate.compile_error or "unknown")[:100]
                lines.append(
                    f"- [{node.node_id}] {node.strategy_name}: "
                    f"FAILED ({err}) status={status}")
            else:
                lines.append(
                    f"- [{node.node_id}] {node.strategy_name}: "
                    f"no result status={status}")
        return "\n".join(lines) if lines else "(no results)"

    def _format_failed_nodes(self, nodes: list) -> str:
        """Format failed nodes for the WM note_failure prompt."""
        lines = []
        for node in nodes:
            err = ""
            if node.candidate:
                err = (node.candidate.compile_error or "unknown")[:150]
            lines.append(
                f"- [{node.node_id}] {node.strategy_name} "
                f"(attempts={node.attempt_count}, "
                f"difficulty={node.difficulty}): {err}")
        return "\n".join(lines) if lines else "(none)"

    # ── Main search loop ──────────────────────────────────────────────────

    def run(self) -> KernelCandidate:
        """Execute K-Search: WM-guided tree search with action cycles."""
        env           = self.env
        problem_shape = env.problem_shapes[0]

        # ── Baselines ──────────────────────────────────────────────────────
        baseline_us = env.baseline_us_reported
        search_naive_us = self.measure_search_baseline(problem_shape)
        if search_naive_us:
            logger.info("Naive reference kernel (harness): %.3f us", search_naive_us)
        logger.info("FlashInfer baseline (speedup denominator): %.3f us", baseline_us)

        logger.info("="*60)
        logger.info("K-Search: kernel=%s beam_width=%d rounds=%d stagnation_K=%d",
                    env.kernel_name, self.beam_w, self.rounds, STAGNATION_K)
        logger.info("="*60)

        # ── World Model Init ──────────────────────────────────────────────
        hw_summary = (
            f"NVIDIA B200 (sm_100a, Blackwell) — 8 TB/s HBM3e, 142 SMs, "
            f"228KB smem/SM, 255 regs/thread, 126MB L2 (cycled = cold)")
        wm = WorldModel(
            engine=self.engine,
            kernel_src=env.kernel_src,
            kernel_type=env.kernel_type,
            problem_shape=problem_shape,
            hw_summary=hw_summary,
        )

        wm_strategies = wm.init_strategies(n=self.beam_w * 2)

        # Fallback to old decompose if WM fails
        if not wm_strategies:
            logger.warning("WM init failed — falling back to freeform decompose")
            raw = self.engine.run_decompose()
            wm_strategies = [
                {"name": s.get("name", s) if isinstance(s, dict) else s,
                 "what": s.get("what", "") if isinstance(s, dict) else "",
                 "score": 0.5, "difficulty": 3, "rationale": ""}
                for s in raw
            ]

        # ── Build search tree ─────────────────────────────────────────────
        tree = SearchTree()

        # Root = naive kernel (always CLOSED)
        naive_candidate = KernelCandidate(
            code=env.kernel_src_raw, strategy="naive_reference",
            round_num=0, compile_ok=True, correct=True,
            speedup=baseline_us / search_naive_us if search_naive_us else 0.5,
        )
        root = tree.create_root(naive_candidate)

        # Add WM-proposed strategies as OPEN children of root
        for s in wm_strategies:
            tree.add_strategy_node(
                root, s["name"], s["what"],
                score=s["score"], difficulty=s["difficulty"],
            )

        logger.info("Tree initialized: %s", tree.stats())
        env.current_round = 0

        # ── Action cycles ─────────────────────────────────────────────────
        # Each round: pick top-N frontier nodes, run action cycle, evolve WM
        total_rounds = self.rounds + 1  # +1 because round 0 is first generation
        kernel_slice = env.kernel_src

        for round_num in range(total_rounds):
            env.current_round = round_num

            if env.over_budget():
                logger.warning("Budget exhausted at round %d", round_num)
                break
            if not tree.has_frontier():
                logger.warning("No frontier nodes — stopping")
                break

            logger.info("--- Round %d (%s) ---", round_num, tree.stats())

            # ── Select top-N frontier nodes (deterministic argmax by score)
            chosen = tree.choose_top_n_frontier(self.beam_w)
            if not chosen:
                logger.warning("No selectable frontier nodes")
                break

            for nd in chosen:
                logger.info("  Selected [%s] %s (score=%.2f d=%d attempts=%d)",
                            nd.node_id, nd.strategy_name,
                            nd.score, nd.difficulty, nd.attempt_count)

            # ── Phase 1: Generate first attempts (parallel) ───────────────
            # Nodes that haven't been tried yet → generate from parent code
            to_generate_nodes = [nd for nd in chosen if nd.attempt_count == 0]
            # Nodes with a previous best → refine it
            to_refine_nodes = [nd for nd in chosen if nd.attempt_count > 0
                               and nd.candidate and nd.candidate.is_viable()]
            # Nodes that failed before → retry generation with error context
            to_retry_nodes = [nd for nd in chosen if nd.attempt_count > 0
                              and (not nd.candidate or not nd.candidate.is_viable())]

            profile_fn = self._make_inner_profile_fn(problem_shape, baseline_us)
            round_results = []  # list of (node, candidate)

            # Generate fresh beams
            if to_generate_nodes:
                strategies = []
                for nd in to_generate_nodes:
                    strategies.append({
                        "name": nd.strategy_name,
                        "what": nd.strategy_description,
                    })

                # Use parent code as base (root's code = naive kernel)
                # Group by parent to batch efficiently
                parent_groups = {}
                for nd, strat in zip(to_generate_nodes, strategies):
                    base = nd.parent_code or kernel_slice
                    parent_groups.setdefault(id(base), (base, []))[1].append(
                        (nd, strat))

                for _, (base_code, items) in parent_groups.items():
                    strats = [s for _, s in items]
                    nodes = [n for n, _ in items]
                    candidates = self.engine.run_generate_beams(
                        strategies=strats,
                        kernel_slice=base_code,
                        round_num=round_num,
                        profile_fn=profile_fn,
                    )
                    round_results.extend(zip(nodes, candidates))

            # Refine existing viable candidates
            if to_refine_nodes:
                parents = []
                for nd in to_refine_nodes:
                    c = nd.candidate
                    c.prev_metrics = c.metrics
                    c.best_code = c.code
                    c.best_speedup = c.speedup
                    parents.append(c)
                refined = self.engine.run_refine_beams(
                    parents, round_num, profile_fn=profile_fn)
                round_results.extend(zip(to_refine_nodes, refined))

            # Retry failed nodes with error context
            if to_retry_nodes:
                for nd in to_retry_nodes:
                    err = (nd.candidate.compile_error if nd.candidate
                           else "Unknown failure")
                    strat = {
                        "name": nd.strategy_name,
                        "what": (nd.strategy_description +
                                 f"\n\nPrevious attempt FAILED:\n{err[:400]}\n"
                                 "Fix the bug. Try a different implementation."),
                    }
                    base_code = nd.parent_code or kernel_slice
                    cands = self.engine.run_generate_beams(
                        strategies=[strat],
                        kernel_slice=base_code,
                        round_num=round_num,
                        profile_fn=profile_fn,
                    )
                    if cands:
                        round_results.append((nd, cands[0]))

            # ── Profile unprofiled candidates ─────────────────────────────
            for nd, c in round_results:
                if c.code and not c.metrics:
                    self._profile_candidate(c, problem_shape, baseline_us)

            # ── Update tree nodes with results ────────────────────────────
            newly_closed = []
            failed_nodes = []

            for nd, child_c in round_results:
                env.optimization_history.record(child_c)
                nd.attempt_count += 1
                logger.info("  %s", child_c.summary())

                if child_c.is_viable():
                    improved = child_c.speedup > nd.best_cycle_speedup + 0.02
                    if improved:
                        nd.best_cycle_speedup = child_c.speedup
                        nd.candidate = child_c
                        nd.stagnant_count = 0
                        logger.info("    IMPROVED -> %.3fx (attempts=%d)",
                                    child_c.speedup, nd.attempt_count)
                    else:
                        nd.stagnant_count += 1
                        # Keep better candidate
                        if not nd.candidate or child_c.speedup > nd.candidate.speedup:
                            nd.candidate = child_c
                        logger.info("    STAGNANT (%d/%d) %.3fx",
                                    nd.stagnant_count, STAGNATION_K,
                                    child_c.speedup)
                else:
                    nd.stagnant_count += 1
                    if not nd.candidate:
                        nd.candidate = child_c  # keep even failed for error tracking
                    logger.info("    FAILED (attempts=%d stagnant=%d)",
                                nd.attempt_count, nd.stagnant_count)

                # Check stagnation → close the node
                if nd.stagnant_count >= STAGNATION_K or nd.attempt_count >= MAX_NODE_EXPANSIONS:
                    if nd.candidate and nd.candidate.is_viable():
                        tree.close_node(nd, nd.candidate)
                        newly_closed.append(nd)
                        logger.info("    CLOSED [%s] at %.3fx after %d attempts",
                                    nd.strategy_name, nd.speedup, nd.attempt_count)
                    else:
                        nd.is_open = False  # dead node
                        failed_nodes.append(nd)
                        logger.info("    DEAD [%s] after %d failed attempts",
                                    nd.strategy_name, nd.attempt_count)

            # ── World Model Evolution ─────────────────────────────────────
            if newly_closed or failed_nodes:
                tree_json = tree.to_compact_json()

                # Evolve: insert children of successful nodes, update scores
                if newly_closed:
                    results_summary = self._format_round_results(
                        [(nd, nd.candidate) for nd in newly_closed])
                    edits = wm.evolve(tree_json, results_summary)
                    tree.apply_wm_edits(edits)

                # Handle failures: downgrade, propose alternatives
                if failed_nodes:
                    failed_summary = self._format_failed_nodes(failed_nodes)
                    edits = wm.note_failure(tree_json, failed_summary)
                    tree.apply_wm_edits(edits)

                logger.info("  After WM evolution: %s", tree.stats())

        # ── Final: Combine top-2 from entire tree ─────────────────────────
        best_nodes = tree.get_best_n(max(self.beam_w, 4))
        survivors = [nd.candidate for nd in best_nodes]

        logger.info("K-Search complete: %s", tree.stats())
        for i, nd in enumerate(best_nodes[:4]):
            logger.info("  Top-%d: [%s] %s %.3fx (depth=%d)",
                        i + 1, nd.node_id, nd.strategy_name,
                        nd.speedup, nd.depth)

        top_for_combine = self.selector.select_for_combination(survivors)
        logger.info("Combining %d top beams", len(top_for_combine))

        if len(top_for_combine) >= 2:
            final = self.engine.combine(top_for_combine)
            m     = self._profile_candidate(final, problem_shape, baseline_us)
            best_survivor = (max(survivors, key=lambda c: c.speedup)
                             if survivors else None)
            if best_survivor and final.speedup < best_survivor.speedup * 0.95:
                logger.warning("Combination regressed (%.3fx < %.3fx) — reverting",
                               final.speedup, best_survivor.speedup)
                final = best_survivor
        else:
            final = (top_for_combine[0] if top_for_combine
                     else survivors[0] if survivors
                     else KernelCandidate(code="", strategy="none"))

        logger.info("="*60)
        logger.info("Search complete: %s", final.summary())
        logger.info("Total API cost: $%.4f", env.total_api_cost_usd)
        logger.info("="*60)

        self.engine.close()
        return final
