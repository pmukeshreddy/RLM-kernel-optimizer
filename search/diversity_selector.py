"""
diversity_selector.py — Cluster-based beam diversity preservation.
Keeps the best candidate from each bottleneck cluster rather than top-K globally.
"""

from __future__ import annotations
import logging

from profiler.bottleneck_classifier import BottleneckClassifier, Bottleneck
from profiler.metrics import KernelMetrics

logger = logging.getLogger(__name__)


class DiversitySelector:
    """
    Selects survivors while preserving diversity across bottleneck types.
    Avoids collapsing all beams to the same optimization direction.
    """

    def __init__(self, config: dict):
        self.classifier    = BottleneckClassifier(config)
        self.mode          = config["beam"].get("diversity_mode", "bottleneck_cluster")
        self.combine_top_k = config["beam"].get("combine_top_k", 2)

    def select_survivors(
        self,
        candidates_with_metrics: list,
        max_survivors: int = 4,
    ) -> list:
        """Select diverse survivors; at most max_survivors, one per bottleneck cluster."""
        if self.mode == "bottleneck_cluster":
            return self._cluster_select(candidates_with_metrics, max_survivors)
        return self._top_k_select(candidates_with_metrics, max_survivors)

    def _cluster_select(self, candidates_with_metrics: list, max_survivors: int) -> list:
        clusters = {}
        for candidate, metrics in candidates_with_metrics:
            # Only filter on compile_ok — correctness check runs later in run.py
            if not candidate.compile_ok:
                continue
            bottleneck = self.classifier.classify(metrics)
            clusters.setdefault(bottleneck, []).append((candidate, metrics))

        survivors = []
        for bottleneck, group in clusters.items():
            best_c, best_m = max(group, key=lambda x: x[1].speedup)
            best_c.bottleneck = bottleneck.value
            best_c.metrics    = best_m.to_dict()
            best_c.speedup    = best_m.speedup
            survivors.append(best_c)
            logger.info("Cluster %s: best=%s speedup=%.3fx",
                        bottleneck.value, best_c.strategy, best_m.speedup)

        survivors.sort(key=lambda c: -c.speedup)
        return survivors[:max_survivors]

    def _top_k_select(self, candidates_with_metrics: list, max_survivors: int) -> list:
        viable = [(c, m) for c, m in candidates_with_metrics if c.compile_ok]
        viable.sort(key=lambda x: -x[1].speedup)
        survivors = []
        for candidate, metrics in viable[:max_survivors]:
            candidate.bottleneck = self.classifier.classify(metrics).value
            candidate.metrics    = metrics.to_dict()
            candidate.speedup    = metrics.speedup
            survivors.append(candidate)
        return survivors

    def select_for_combination(self, survivors: list) -> list:
        """Select top-K for combination, preferring orthogonal bottleneck types."""
        if len(survivors) <= self.combine_top_k:
            return survivors

        selected = []
        seen_bottlenecks = set()
        for c in sorted(survivors, key=lambda x: -x.speedup):
            if c.bottleneck not in seen_bottlenecks:
                selected.append(c)
                seen_bottlenecks.add(c.bottleneck)
            if len(selected) >= self.combine_top_k:
                break

        for c in sorted(survivors, key=lambda x: -x.speedup):
            if c not in selected:
                selected.append(c)
            if len(selected) >= self.combine_top_k:
                break

        return selected
