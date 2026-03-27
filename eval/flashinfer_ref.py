"""
flashinfer_ref.py — FlashInfer-based reference outputs and baseline timing.

Uses FlashInfer's production CUDA kernels on B200 as the ground-truth
reference for WaferBench NVFP4, matching the official evaluation methodology.

Matches KernelArena bench_sustained convention:
  - 500 warmup, 100 timed reps, L2 cache cycling via input buffer rotation
  - silu_mul uses fused silu_and_mul_scaled_nvfp4_experts_quantize (not 2 calls)
"""

from __future__ import annotations
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import flashinfer
    _HAS_FLASHINFER = True
except ImportError:
    _HAS_FLASHINFER = False
    logger.warning("torch/flashinfer not available — will fall back to reference kernels")

# Check for the fused silu_mul expert API (KernelArena reference)
_HAS_FUSED_SILU = False
if _HAS_FLASHINFER:
    _HAS_FUSED_SILU = hasattr(flashinfer, "silu_and_mul_scaled_nvfp4_experts_quantize")
    if not _HAS_FUSED_SILU:
        logger.warning(
            "flashinfer.silu_and_mul_scaled_nvfp4_experts_quantize not found — "
            "silu_mul baseline will use unfused fallback (NOT comparable to KernelArena)"
        )

# Timing constants matching ThunderKittens 2.0 / WaferBench convention
_WARMUP_ITERS = 500
_BENCH_ITERS  = 100
# L2 cache cycling — must match benchmark.py to keep baseline/candidate symmetric
_L2_CYCLE_BUFS = 4


def available() -> bool:
    return _HAS_FLASHINFER and torch.cuda.is_available()


_jit_warmed_up = False

def _jit_warmup():
    """Trigger FlashInfer JIT compilation before any timing.

    FlashInfer uses Triton/TVM backends that JIT-compile on first call.
    This inflates the first baseline measurement if not pre-warmed.
    """
    global _jit_warmed_up
    if _jit_warmed_up:
        return
    logger.info("FlashInfer JIT warmup (first call triggers compilation)...")
    try:
        # Small tensors to trigger JIT with minimal time
        x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
        r = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
        w = torch.ones(256, dtype=torch.bfloat16, device="cuda")
        flashinfer.add_rmsnorm_fp4quant(x, r, w, eps=1e-6)
        gs = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        flashinfer.fp4_quantize(x, global_scale=gs)
        # Warm fused silu_mul expert API (KernelArena reference)
        if _HAS_FUSED_SILU:
            x3d = torch.randn(1, 4, 512, dtype=torch.bfloat16, device="cuda")
            mask = torch.full((1,), 4, dtype=torch.int64, device="cuda")
            flashinfer.silu_and_mul_scaled_nvfp4_experts_quantize(x3d, mask, gs)
        else:
            combined = torch.cat([x, r], dim=-1)
            flashinfer.silu_and_mul(combined)
        torch.cuda.synchronize()
    except Exception as e:
        logger.warning("JIT warmup partial failure (ok): %s", e)
    _jit_warmed_up = True
    logger.info("FlashInfer JIT warmup complete")


def measure_baseline(kernel_type: str, shape: tuple) -> Optional[float]:
    """Measure FlashInfer baseline timing in microseconds.

    Returns None if FlashInfer is not available.
    """
    if not available():
        return None

    _jit_warmup()

    try:
        if kernel_type == "add_rmsnorm":
            return _baseline_add_rmsnorm(shape)
        elif kernel_type == "silu_mul":
            return _baseline_silu_mul(shape)
        elif kernel_type == "nvfp4_quantize":
            return _baseline_nvfp4_quantize(shape)
        else:
            logger.warning("Unknown kernel_type for FlashInfer baseline: %s", kernel_type)
            return None
    except Exception as e:
        logger.warning("FlashInfer baseline measurement failed: %s", e)
        return None


def generate_reference(kernel_type: str, shape: tuple, seed: int = 42) -> Optional[dict]:
    """Generate reference output tensors using FlashInfer.

    Returns dict with input/output torch tensors, or None if unavailable.
    """
    if not available():
        return None

    try:
        if kernel_type == "add_rmsnorm":
            return _reference_add_rmsnorm(shape, seed)
        elif kernel_type == "silu_mul":
            return _reference_silu_mul(shape, seed)
        elif kernel_type == "nvfp4_quantize":
            return _reference_nvfp4_quantize(shape, seed)
        else:
            return None
    except Exception as e:
        logger.warning("FlashInfer reference generation failed: %s", e)
        return None


# ── Add + RMSNorm + FP4 Quant ────────────────────────────────────────────────

def _baseline_add_rmsnorm(shape: tuple) -> float:
    rows, hidden = shape
    # L2 cycling: multiple input buffers, cycled per iteration
    inps, ress = [], []
    for i in range(_L2_CYCLE_BUFS):
        torch.manual_seed(i)
        inps.append(torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda"))
        ress.append(torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda"))
    w = torch.ones(hidden, dtype=torch.bfloat16, device="cuda")

    def run(buf_idx):
        flashinfer.add_rmsnorm_fp4quant(inps[buf_idx], ress[buf_idx], w, eps=1e-6)

    return _time_fn(run)


def _reference_add_rmsnorm(shape: tuple, seed: int) -> dict:
    rows, hidden = shape
    torch.manual_seed(seed)
    inp = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    res = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    w   = torch.ones(hidden, dtype=torch.bfloat16, device="cuda") + \
          torch.rand(hidden, dtype=torch.bfloat16, device="cuda") * 0.1

    # FlashInfer's fused_add_rmsnorm modifies inp in-place (residual_out = inp + res)
    inp_copy = inp.clone()
    res_copy = res.clone()
    flashinfer.fused_add_rmsnorm(inp_copy, res_copy, w, eps=1e-6)

    return {
        "input": inp, "residual": res, "weight": w,
        "residual_out": res_copy, "norm_out": inp_copy,
    }


# ── SiLU × Mul + FP4 Quant ──────────────────────────────────────────────────

def _baseline_silu_mul(shape: tuple) -> float:
    b, m, k = shape
    global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    if _HAS_FUSED_SILU:
        # KernelArena reference: single fused kernel (3D expert-batched, swizzled)
        # Input: (B, M, 2*K) bf16 — gate and up concatenated on last dim
        # mask: (B,) int — per-expert token count (all M for uniform batches)
        xs = []
        for i in range(_L2_CYCLE_BUFS):
            torch.manual_seed(i)
            xs.append(torch.randn(b, m, 2 * k, dtype=torch.bfloat16, device="cuda"))
        mask = torch.full((b,), m, dtype=torch.int64, device="cuda")

        def run(buf_idx):
            flashinfer.silu_and_mul_scaled_nvfp4_experts_quantize(
                xs[buf_idx], mask, global_scale
            )

        return _time_fn(run)
    else:
        # Fallback: two separate calls (NOT comparable to KernelArena)
        logger.warning("Using unfused silu_mul baseline — speedup NOT comparable to KernelArena")
        combineds = []
        for i in range(_L2_CYCLE_BUFS):
            torch.manual_seed(i)
            gate = torch.randn(b * m, k, dtype=torch.bfloat16, device="cuda")
            up   = torch.randn(b * m, k, dtype=torch.bfloat16, device="cuda")
            combineds.append(torch.cat([gate, up], dim=-1))

        def run(buf_idx):
            out = flashinfer.silu_and_mul(combineds[buf_idx])
            flashinfer.fp4_quantize(out, global_scale=global_scale)

        return _time_fn(run)


def _reference_silu_mul(shape: tuple, seed: int) -> dict:
    b, m, k = shape
    torch.manual_seed(seed)
    gate = torch.randn(b * m, k, dtype=torch.bfloat16, device="cuda")
    up   = torch.randn(b * m, k, dtype=torch.bfloat16, device="cuda")
    combined = torch.cat([gate, up], dim=-1)  # (B*M, 2*K)
    out  = flashinfer.silu_and_mul(combined)
    return {"gate": gate.flatten(), "up": up.flatten(), "output": out.flatten()}


# ── NVFP4 Quantize ──────────────────────────────────────────────────────────

def _baseline_nvfp4_quantize(shape: tuple) -> float:
    m, k = shape
    inps = []
    for i in range(_L2_CYCLE_BUFS):
        torch.manual_seed(i)
        inps.append(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
    global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    def run(buf_idx):
        flashinfer.fp4_quantize(inps[buf_idx], global_scale=global_scale)

    return _time_fn(run)


def _reference_nvfp4_quantize(shape: tuple, seed: int) -> dict:
    m, k = shape
    torch.manual_seed(seed)
    inp = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    packed, scales = flashinfer.fp4_quantize(inp, global_scale=global_scale)
    return {"input": inp, "packed": packed, "scales": scales}


# ── Timing utility ───────────────────────────────────────────────────────────

def _time_fn(fn, warmup: int = _WARMUP_ITERS, iters: int = _BENCH_ITERS) -> float:
    """Time a function with L2 cache cycling. Returns microseconds per call.

    fn(buf_idx) is called with a rotating buffer index to prevent L2 cache
    hits from inflating performance — matching benchmark.py's candidate timing.
    """
    nbufs = _L2_CYCLE_BUFS
    
    # Pre-warmup
    for i in range(warmup):
        fn(i % nbufs)
    torch.cuda.synchronize()

    # Capture CUDA Graph containing all iterations
    # Stream-specific warmup for graph capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(iters):
            fn(i % nbufs)
    s.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(iters):
            fn(i % nbufs)

    # Timed graph replay
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    g.replay()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    return ms * 1000.0 / iters  # convert ms → µs per iteration
