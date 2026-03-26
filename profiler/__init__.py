"""profiler — NCU + hybrid kernel profiler for RLM beam search."""
from .ncu_runner import NCURunner
from .hybrid_profiler import HybridProfiler
from .bottleneck_classifier import BottleneckClassifier, Bottleneck
from .metrics import KernelMetrics, metrics_from_dict
from .roofline import operational_intensity, roofline_bound, efficiency_report

__all__ = [
    "NCURunner", "HybridProfiler", "BottleneckClassifier", "Bottleneck",
    "KernelMetrics", "metrics_from_dict",
    "operational_intensity", "roofline_bound", "efficiency_report",
]
