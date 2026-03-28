"""
bottleneck_classifier.py — Classify kernel bottleneck from NCU metrics.
"""

from __future__ import annotations
from enum import Enum
from .metrics import KernelMetrics


class Bottleneck(str, Enum):
    MEMORY_BOUND  = "memory_bound"
    COMPUTE_BOUND = "compute_bound"
    SYNC_BOUND    = "sync_bound"
    LATENCY_BOUND = "latency_bound"
    UNKNOWN       = "unknown"


class BottleneckClassifier:

    def __init__(self, config: dict):
        bc = config.get("bottleneck", {})
        self.mem_threshold     = bc.get("memory_bound_threshold_pct",  70.0)
        self.compute_threshold = bc.get("compute_bound_threshold_pct", 60.0)
        self.sync_threshold    = bc.get("sync_bound_threshold_pct",    30.0)
        self.latency_threshold = bc.get("latency_bound_threshold_pct", 40.0)

    def classify(self, metrics: KernelMetrics) -> Bottleneck:
        if metrics.mem_throughput_pct >= self.mem_threshold:
            return Bottleneck.MEMORY_BOUND
        if metrics.compute_throughput_pct >= self.compute_threshold:
            return Bottleneck.COMPUTE_BOUND
        if metrics.stall_barrier >= self.sync_threshold:
            return Bottleneck.SYNC_BOUND
        if metrics.stall_memory >= self.latency_threshold:
            return Bottleneck.LATENCY_BOUND

        # Low utilization on both axes → kernel is latency-bound
        # (too small to saturate hardware, or stalled waiting for data)
        if metrics.mem_throughput_pct < 30 and metrics.compute_throughput_pct < 30:
            return Bottleneck.LATENCY_BOUND

        return Bottleneck.UNKNOWN

    def classify_all(self, candidates_metrics: list) -> dict:
        groups = {b: [] for b in Bottleneck}
        for candidate, metrics in candidates_metrics:
            groups[self.classify(metrics)].append((candidate, metrics))
        return {k: v for k, v in groups.items() if v}

    def actionable_advice(self, metrics: KernelMetrics) -> str:
        b = self.classify(metrics)
        advice = {
            Bottleneck.MEMORY_BOUND: (
                f"Memory bound at {metrics.mem_throughput_pct:.1f}% bandwidth. "
                "CUDAMaster Diagnosis: If writing heavily to HBM without reading back, L2 cache allocation thrashing is occurring. "
                "If reading a shared weight vector across blocks, L1 read bandwidth is saturated. "
                "Consider RAG queries: Cache-Streaming Stores (st.global.cs), or Multi-Row Processing."
            ),
            Bottleneck.COMPUTE_BOUND: (
                f"Compute bound at {metrics.compute_throughput_pct:.1f}% throughput. "
                f"Occupancy={metrics.sm_occupancy:.1f}%. "
                "CUDAMaster Diagnosis: SM instructions (FADD/FMUL/Bitwise) are dominating. "
                "Consider RAG queries: Blackwell PTX Intrinsics, Thread Coarsening, Fast Math."
            ),
            Bottleneck.SYNC_BOUND: (
                f"Sync bound: {metrics.stall_barrier:.1f}% barrier stalls. "
                f"SM occupancy={metrics.sm_occupancy:.1f}%. "
                "Try: warp shuffles, eliminate __syncthreads, warp-level reductions."
            ),
            Bottleneck.LATENCY_BOUND: (
                f"Latency bound: {metrics.stall_memory:.1f}% memory stalls. "
                "Try: software pipeline (cp.async), async prefetch, increase ILP."
            ),
            Bottleneck.UNKNOWN: (
                f"Bottleneck unclear. mem={metrics.mem_throughput_pct:.1f}% "
                f"compute={metrics.compute_throughput_pct:.1f}% "
                f"stalls: mem={metrics.stall_memory:.1f}% bar={metrics.stall_barrier:.1f}%"
            ),
        }
        return advice[b]

    def roofline_efficiency(self, metrics: KernelMetrics, hw_spec: dict) -> dict:
        return {
            "mem_efficiency_pct":     metrics.mem_throughput_pct,
            "compute_efficiency_pct": metrics.compute_throughput_pct,
            "headroom_mem_pct":       100.0 - metrics.mem_throughput_pct,
            "headroom_compute_pct":   100.0 - metrics.compute_throughput_pct,
            "bottleneck":             self.classify(metrics).value,
            "hw_mem_bw_tbs":          hw_spec["memory"]["hbm_bandwidth_tbs"],
            "hw_fp32_tflops":         hw_spec["compute"]["fp32_tflops"],
            "hw_nvfp4_tflops":        hw_spec["compute"]["nvfp4_tflops"],
        }
