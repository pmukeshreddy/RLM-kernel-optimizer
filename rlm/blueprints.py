"""
blueprints.py (Legacy) — Kept only for Roofline feedback logic.
Structural blueprints have been moved to the BM25 RAG Knowledge Base.
"""

def get_roofline_feedback(kernel_type: str, problem_shape: tuple,
                          timing_us: float, speedup: float) -> str:
    """Calculate roofline efficiency for profiler feedback."""
    if kernel_type == "add_rmsnorm":
        rows, hidden = problem_shape
        n = rows * hidden
        # Minimum bytes (single-pass, no re-read)
        min_bytes = n * 2 + n * 2 + hidden * 2 + n * 2 + n // 2 + n // 16
        # Naive bytes (with re-read)
        naive_bytes = min_bytes + n * 2
    elif kernel_type == "silu_mul":
        if len(problem_shape) == 3:
            n = problem_shape[0] * problem_shape[1] * problem_shape[2]
        else:
            n = problem_shape[0]
        min_bytes = n * 2 * 2 + n // 2 + n // 16
        naive_bytes = min_bytes
    elif kernel_type == "nvfp4_quantize":
        if len(problem_shape) == 2:
            n = problem_shape[0] * problem_shape[1]
        else:
            n = problem_shape[0]
        min_bytes = n * 2 + n // 2 + n // 16
        naive_bytes = min_bytes
    else:
        return ""

    if timing_us <= 0:
        return ""

    achieved_bw = naive_bytes / timing_us / 1e6  # TB/s
    peak_bw = 8.0  # TB/s for B200
    bw_util = achieved_bw / peak_bw * 100
    theoretical_min = min_bytes / (peak_bw * 1e6)
    realistic_min = min_bytes / (4.0 * 1e6)  # ~50% of peak for small kernels
    efficiency = theoretical_min / timing_us * 100

    lines = [
        f"Roofline: {achieved_bw:.2f} TB/s achieved ({bw_util:.0f}% of 8 TB/s peak HBM)",
        f"  Theoretical min (peak BW):     {theoretical_min:.3f} us",
        f"  Realistic min (~50% peak):     {realistic_min:.3f} us",
        f"  Your timing:                   {timing_us:.3f} us",
        f"  Roofline efficiency:           {efficiency:.0f}%",
    ]

    return "\n".join(lines)
