"""
flashinfer_ref.py — FlashInfer-based reference outputs and baseline timing.

Uses FlashInfer's production CUDA kernels on B200 as the ground-truth
reference for WaferBench NVFP4, matching the official evaluation methodology.
"""

from __future__ import annotations
import logging
import subprocess
import tempfile
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

# Timing constants matching ThunderKittens 2.0 / WaferBench convention
_WARMUP_ITERS = 500
_BENCH_ITERS  = 100


def available() -> bool:
    return _HAS_FLASHINFER and torch.cuda.is_available()


def measure_baseline(kernel_type: str, shape: tuple) -> Optional[float]:
    """Measure FlashInfer baseline timing in microseconds.

    Returns None if FlashInfer is not available.
    """
    if not available():
        return None

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
    torch.manual_seed(0)
    inp = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    res = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    w   = torch.ones(hidden, dtype=torch.bfloat16, device="cuda")

    def run():
        # Single fused call matching KernelArena: add + rmsnorm + fp4quant
        flashinfer.add_rmsnorm_fp4quant(inp, res, w, eps=1e-6)

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
    torch.manual_seed(0)
    # silu_and_mul expects single concatenated tensor of shape (N, 2*K)
    # where first half is gate, second half is up
    gate = torch.randn(b * m, k, dtype=torch.bfloat16, device="cuda")
    up   = torch.randn(b * m, k, dtype=torch.bfloat16, device="cuda")
    combined = torch.cat([gate, up], dim=-1)  # (B*M, 2*K)
    global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    def run():
        # silu_and_mul + fp4_quantize (no single fused API available)
        out = flashinfer.silu_and_mul(combined)
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
    torch.manual_seed(0)
    inp = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    def run():
        flashinfer.fp4_quantize(inp, global_scale=global_scale)

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
    """Time a function using CUDA events. Returns microseconds per call."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    return ms * 1000.0 / iters  # convert ms → µs per iteration
