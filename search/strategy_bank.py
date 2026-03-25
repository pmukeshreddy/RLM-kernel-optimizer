"""
strategy_bank.py — Predefined optimization strategies with metadata.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Strategy:
    name: str
    display_name: str
    description: str
    targets_bottleneck: list
    priority: int
    requires: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)

    def applies_to(self, bottleneck: str) -> bool:
        return bottleneck in self.targets_bottleneck or not self.targets_bottleneck


STRATEGY_BANK: dict = {
    "vectorize_loads": Strategy(
        name="vectorize_loads",
        display_name="Float4 Vectorized Loads",
        description="Replace scalar loads with 128-bit float4/uint4 vectorized transactions",
        targets_bottleneck=["memory_bound", "latency_bound"],
        priority=10,
    ),
    "tma_prefetch": Strategy(
        name="tma_prefetch",
        display_name="TMA Async Prefetch",
        description="Use Blackwell TMA engine for async bulk copy with double buffering",
        targets_bottleneck=["memory_bound", "latency_bound"],
        priority=9,
    ),
    "warp_reduction": Strategy(
        name="warp_reduction",
        display_name="Warp Shuffle Reduction",
        description="Replace __syncthreads reductions with __shfl_xor_sync",
        targets_bottleneck=["sync_bound"],
        priority=8,
    ),
    "fuse_passes": Strategy(
        name="fuse_passes",
        display_name="Fuse Memory Passes",
        description="Combine multiple global memory passes into a single kernel loop",
        targets_bottleneck=["memory_bound"],
        priority=7,
        conflicts=["async_pipeline"],
    ),
    "register_tiling": Strategy(
        name="register_tiling",
        display_name="Register-Level Tiling",
        description="Process multiple elements per thread using register arrays for ILP",
        targets_bottleneck=["compute_bound", "latency_bound"],
        priority=6,
    ),
    "async_pipeline": Strategy(
        name="async_pipeline",
        display_name="Async Software Pipeline",
        description="Overlap memory and compute using cp.async with double buffering",
        targets_bottleneck=["latency_bound", "memory_bound"],
        priority=5,
        conflicts=["fuse_passes"],
    ),
}


def select_strategies(
    bottleneck: str,
    missing_opts: list,
    tried: list,
    beam_width: int = 4,
) -> list:
    candidates = []
    for name, s in STRATEGY_BANK.items():
        if name in tried:
            continue
        if name not in missing_opts and missing_opts:
            continue
        score = s.priority
        if s.applies_to(bottleneck):
            score += 5
        candidates.append((score, name))
    candidates.sort(key=lambda x: -x[0])
    return [name for _, name in candidates[:beam_width]]


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGY_BANK:
        raise ValueError(f"Unknown strategy: {name}")
    return STRATEGY_BANK[name]
